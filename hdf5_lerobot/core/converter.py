"""Main conversion engine: HDF5 → LeRobot v2.1 dataset.

This module contains the generic ``convert_dataset`` orchestrator.
Schema-specific behaviour is injected via the *hooks* parameter.
"""

import logging
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from tqdm import tqdm

from ..hooks import DefaultHooks
from ..language import extract_language_instructions as _builtin_extract_language
from .features import build_info_json
from .hdf5_utils import (
    decode_compressed_image,
    get_image_dimensions_from_compressed,
    read_concatenated,
)
from .hub import push_dataset_to_hub
from .io_utils import load_json, load_yaml, natural_sort_key, write_json, write_jsonl
from .video import FFmpegVideoWriter

logger = logging.getLogger(__name__)


def convert_dataset(
    hdf5_dir: str,
    schema_path: str,
    mapping_path: str,
    output_dir: str,
    hooks: object | None = None,
    push_to_hub: bool = False,
    repo_id_override: str | None = None,
    hub_token: str | None = None,
    private: bool = False,
    *,
    schema: dict | None = None,
    mapping: dict | None = None,
):
    """Convert HDF5 episode files to a LeRobot v2.1 dataset.

    Parameters
    ----------
    hdf5_dir : str
        Directory containing ``.hdf5`` episode files.
    schema_path : str
        Path to ``schema.json``.  Ignored if *schema* is provided directly.
    mapping_path : str
        Path to ``mapping.yaml``.  Ignored if *mapping* is provided directly.
    output_dir : str
        Output directory for the LeRobot dataset.
    hooks : object, optional
        An object implementing any subset of the ``ConverterHooks`` protocol.
        Defaults to ``DefaultHooks()`` (no-op).
    push_to_hub : bool
        Upload to HuggingFace Hub after conversion.
    repo_id_override : str, optional
        Override ``dataset.repo_id`` from the mapping config.
    hub_token : str, optional
        HuggingFace API token.
    private : bool
        Create the HuggingFace repo as private.
    schema : dict, optional
        Pre-loaded schema dict (avoids re-reading *schema_path*).
    mapping : dict, optional
        Pre-loaded mapping dict (avoids re-reading *mapping_path*).
    """
    if hooks is None:
        hooks = DefaultHooks()

    if schema is None:
        schema = load_json(schema_path)
    if mapping is None:
        mapping = load_yaml(mapping_path)

    output_dir = Path(output_dir)

    # --- Discover HDF5 files ---
    hdf5_files = sorted(
        [f for f in Path(hdf5_dir).glob("*.hdf5")],
        key=lambda p: natural_sort_key(p.name),
    )
    if not hdf5_files:
        hdf5_files = sorted(
            [f for f in Path(hdf5_dir).glob("**/*.hdf5")],
            key=lambda p: natural_sort_key(p.name),
        )
    logger.info(f"Found {len(hdf5_files)} HDF5 files in {hdf5_dir}")
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found in {hdf5_dir}")

    # --- Load external language instructions if needed ---
    lang_cfg = mapping.get("language_instructions", {})
    external_instructions = None
    if lang_cfg.get("source") == "external_json":
        ext_path = lang_cfg["external_json"]["path"]
        external_instructions = load_json(ext_path)

    # --- Determine FPS ---
    fps_cfg = mapping["dataset"]["fps"]
    if fps_cfg["source"] == "fixed":
        fps = fps_cfg["fixed_value"]
    elif fps_cfg["source"] == "hdf5":
        with h5py.File(hdf5_files[0], "r") as h5f:
            fps = int(h5f[fps_cfg["hdf5_path"]][()])
    else:
        raise ValueError(f"Unknown fps source: {fps_cfg['source']}")
    logger.info(f"Dataset FPS: {fps}")

    # --- Detect image dimensions from first episode ---
    image_dims: dict[str, tuple[int, int, int]] = {}
    img_cfg = mapping["images"]
    with h5py.File(hdf5_files[0], "r") as h5f:
        for cam in img_cfg["cameras"]:
            raw_row = h5f[cam["hdf5_path"]][0]
            if img_cfg.get("compressed", False):
                comp_lens = h5f[img_cfg["compressed_len_path"]][:]
                byte_len = int(comp_lens[cam["compressed_len_index"], 0])
                img_bytes = raw_row[:byte_len].tobytes()
            else:
                h_raw = img_cfg.get("raw_height", 480)
                w_raw = img_cfg.get("raw_width", 640)
                c_raw = img_cfg.get("raw_channels", 3)
                img_bytes = raw_row.reshape(h_raw, w_raw, c_raw)
                image_dims[cam["key"]] = (h_raw, w_raw, c_raw)
                continue
            h, w, c = get_image_dimensions_from_compressed(img_bytes)
            image_dims[cam["key"]] = (h, w, c)
    logger.info(f"Detected image dims: {image_dims}")

    # --- First pass: collect tasks, filter episodes, count frames ---
    task_to_index: dict[str, int] = {}
    task_counter = 0
    episode_infos: list[dict] = []
    global_frame_count = 0
    filter_cfg = mapping.get("filtering", {})

    logger.info("Pass 1/2: Scanning episodes for tasks and frame counts...")
    for file_idx, hdf5_path in enumerate(tqdm(hdf5_files, desc="Scanning")):
        with h5py.File(hdf5_path, "r") as h5f:
            # --- Hook-based filtering (runs first) ---
            if not hooks.filter_episode(h5f, mapping):
                logger.debug(f"Skipping {hdf5_path.name} (hook filter)")
                continue

            # --- YAML-based filtering ---
            if filter_cfg.get("skip_on_error", False):
                err_path = filter_cfg.get("error_flag_path", "/metadata/data_collection_error")
                if err_path in h5f:
                    if bool(h5f[err_path][()]):
                        logger.debug(f"Skipping {hdf5_path.name} (data_collection_error=True)")
                        continue

            # Determine number of frames from first observation component
            first_obs_path = mapping["observation_state"]["components"][0]["hdf5_path"]
            num_frames = h5f[first_obs_path].shape[0]

            min_frames = filter_cfg.get("min_frames", 0)
            if num_frames < min_frames:
                logger.debug(f"Skipping {hdf5_path.name} ({num_frames} < {min_frames} frames)")
                continue

            # --- Extract language instructions (to discover unique tasks) ---
            instructions = hooks.extract_language_instructions(
                h5f, num_frames, lang_cfg,
                episode_id=hdf5_path.stem,
                external_instructions=external_instructions,
            )
            if instructions is None:
                instructions = _builtin_extract_language(
                    h5f, num_frames, lang_cfg,
                    episode_id=hdf5_path.stem,
                    external_instructions=external_instructions,
                )
            unique_instructions = list(dict.fromkeys(instructions))  # preserve order, dedup
            task_indices_for_episode = []
            for instr in unique_instructions:
                if instr not in task_to_index:
                    task_to_index[instr] = task_counter
                    task_counter += 1
                task_indices_for_episode.append(task_to_index[instr])

            # Per-frame task indices
            frame_task_indices = [task_to_index[inst] for inst in instructions]

            episode_infos.append({
                "hdf5_path": str(hdf5_path),
                "num_frames": num_frames,
                "tasks": unique_instructions,
                "task_indices": task_indices_for_episode,
                "frame_task_indices": frame_task_indices,
                "frame_instructions": instructions,
                "global_start_index": global_frame_count,
            })
            global_frame_count += num_frames

    total_episodes = len(episode_infos)
    total_tasks = len(task_to_index)
    logger.info(f"Kept {total_episodes} episodes, {global_frame_count} frames, {total_tasks} unique tasks")

    if total_episodes == 0:
        raise RuntimeError("No valid episodes found after filtering!")

    # --- Build info.json ---
    info = build_info_json(
        mapping, schema, fps,
        total_episodes, global_frame_count, total_tasks, image_dims,
    )

    # --- Write meta/ ---
    meta_dir = output_dir / "meta"
    os.makedirs(meta_dir, exist_ok=True)
    write_json(str(meta_dir / "info.json"), info)

    # tasks.jsonl
    tasks_records = []
    for task_str, task_idx in sorted(task_to_index.items(), key=lambda x: x[1]):
        tasks_records.append({"task_index": task_idx, "task": task_str})
    write_jsonl(str(meta_dir / "tasks.jsonl"), tasks_records)

    # --- Second pass: write parquet + video for each episode ---
    chunks_size = mapping["dataset"]["chunks_size"]
    episodes_records = []
    logger.info("Pass 2/2: Writing parquet files and encoding videos...")
    for ep_idx, ep_info in enumerate(tqdm(episode_infos, desc="Converting")):
        chunk_idx = ep_idx // chunks_size
        chunk_dir = f"chunk-{chunk_idx:03d}"

        # --- Read HDF5 data ---
        with h5py.File(ep_info["hdf5_path"], "r") as h5f:
            num_frames = ep_info["num_frames"]

            # Observation state [T, obs_dim]
            obs_state = read_concatenated(h5f, mapping["observation_state"]["components"])
            obs_state = hooks.transform_observation_state(obs_state, h5f, mapping)

            # Action [T, act_dim]
            action = read_concatenated(h5f, mapping["action"]["components"])
            action = hooks.transform_action(action, h5f, mapping)

            # Extra observations
            extra_data = {}
            for extra in mapping.get("extra_observations", []):
                arr = h5f[extra["hdf5_path"]][:].astype(np.float32)
                extra_data[extra["feature_name"]] = arr

            # Hook: extra parquet columns
            hook_extra_cols = hooks.extra_parquet_columns(h5f, num_frames, mapping)

            # --- Build parquet rows ---
            rows = []
            for t in range(num_frames):
                row = {
                    "timestamp": np.float32(t / fps),
                    "frame_index": t,
                    "episode_index": ep_idx,
                    "index": ep_info["global_start_index"] + t,
                    "task_index": ep_info["frame_task_indices"][t],
                    "observation.state": obs_state[t].tolist(),
                    "action": action[t].tolist(),
                }
                for feat_name, arr in extra_data.items():
                    row[feat_name] = arr[t].tolist()

                # Per-frame language instruction
                if lang_cfg.get("per_frame", False):
                    row["language_instruction"] = ep_info["frame_instructions"][t]

                # Video frame references
                for cam in img_cfg["cameras"]:
                    key = f"observation.images.{cam['key']}"
                    row[key] = [
                        {"path": f"videos/{chunk_dir}/{key}/episode_{ep_idx:06d}.mp4",
                         "timestamp": float(t / fps)}
                    ]

                # Hook: extra columns
                for col_name, col_values in hook_extra_cols.items():
                    row[col_name] = col_values[t]

                rows.append(row)

            # --- Write parquet ---
            parquet_dir = output_dir / "data" / chunk_dir
            os.makedirs(parquet_dir, exist_ok=True)
            parquet_path = parquet_dir / f"episode_{ep_idx:06d}.parquet"
            df = pd.DataFrame(rows)
            df.to_parquet(str(parquet_path), engine="pyarrow", index=False)

            # --- Encode videos ---
            comp_lens_data = None
            if img_cfg.get("compressed", False) and img_cfg.get("compressed_len_path"):
                comp_lens_data = h5f[img_cfg["compressed_len_path"]][:]

            for cam in img_cfg["cameras"]:
                cam_key = f"observation.images.{cam['key']}"
                video_dir = output_dir / "videos" / chunk_dir / cam_key
                video_path = video_dir / f"episode_{ep_idx:06d}.mp4"
                h_img, w_img, _ = image_dims[cam["key"]]

                writer = FFmpegVideoWriter(
                    str(video_path), fps, h_img, w_img,
                    codec=img_cfg.get("video_codec", "libx264"),
                    pix_fmt=img_cfg.get("video_pix_fmt", "yuv420p"),
                    crf=img_cfg.get("video_crf", 23),
                )

                raw_images = h5f[cam["hdf5_path"]]
                for t in range(num_frames):
                    raw_row = raw_images[t]
                    if img_cfg.get("compressed", False):
                        byte_len = int(comp_lens_data[cam["compressed_len_index"], t])
                        img_bytes = raw_row[:byte_len].tobytes()
                        frame = decode_compressed_image(img_bytes)
                    else:
                        frame = raw_row.reshape(h_img, w_img, -1)

                    # Ensure RGB uint8
                    if frame.ndim == 2:
                        frame = np.stack([frame] * 3, axis=-1)
                    if frame.shape[-1] == 1:
                        frame = np.concatenate([frame] * 3, axis=-1)
                    frame = frame.astype(np.uint8)
                    if frame.shape[-1] == 4:  # RGBA -> RGB
                        frame = frame[..., :3]

                    # Hook: per-frame transform
                    frame = hooks.transform_frame(frame, cam["key"])

                    writer.write_frame(frame)
                writer.close()

        # --- episodes.jsonl record ---
        episodes_records.append({
            "episode_index": ep_idx,
            "tasks": ep_info["tasks"],
            "length": num_frames,
        })

    # --- Write episodes.jsonl ---
    write_jsonl(str(meta_dir / "episodes.jsonl"), episodes_records)

    logger.info(f"Done! LeRobot v2.1 dataset written to {output_dir}")
    logger.info("Note: stats were not computed. Use LeRobot to compute them:")
    logger.info(f"  from lerobot.datasets.lerobot_dataset import LeRobotDataset")
    logger.info(f"  ds = LeRobotDataset('repo_id', local_files_only=True, root='{output_dir}')")
    logger.info(f"  Episodes: {total_episodes}")
    logger.info(f"  Frames:   {global_frame_count}")
    logger.info(f"  Tasks:    {total_tasks}")
    logger.info(f"  FPS:      {fps}")

    if push_to_hub:
        repo_id = repo_id_override or mapping["dataset"].get("repo_id")
        if not repo_id:
            raise ValueError(
                "No repo_id specified. Provide --repo-id or set dataset.repo_id "
                "in mapping_config.yaml."
            )
        push_dataset_to_hub(
            output_dir=output_dir,
            repo_id=repo_id,
            token=hub_token,
            private=private,
        )
