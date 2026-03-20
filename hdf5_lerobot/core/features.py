"""Feature construction: build LeRobot info.json features from mapping config."""


def build_feature_names(components: list[dict], schema_datasets: dict) -> list[str]:
    """Build ordered list of motor/joint names from component definitions."""
    names = []
    for comp in components:
        ds = schema_datasets.get(comp["hdf5_path"])
        if ds is None:
            raise ValueError(f"HDF5 path {comp['hdf5_path']} not found in schema")
        n_cols = ds["shape"][-1]
        prefix = comp.get("names_prefix", "dim")
        sl = comp.get("slice")
        if sl:
            n_cols = sl[1] - sl[0]
            start = sl[0]
        else:
            start = 0
        for i in range(n_cols):
            names.append(f"{prefix}_{i + start}")
    return names


def get_total_dim(components: list[dict], schema_datasets: dict) -> int:
    """Get total concatenated dimension."""
    total = 0
    for comp in components:
        ds = schema_datasets.get(comp["hdf5_path"])
        sl = comp.get("slice")
        if sl:
            total += sl[1] - sl[0]
        else:
            total += ds["shape"][-1]
    return total


def build_info_json(
    mapping: dict,
    schema: dict,
    fps: int,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    image_dims: dict[str, tuple[int, int, int]],  # cam_key -> (H, W, C)
) -> dict:
    """Build meta/info.json for LeRobot v2.1."""
    ds = schema["datasets"]
    obs_dim = get_total_dim(mapping["observation_state"]["components"], ds)
    act_dim = get_total_dim(mapping["action"]["components"], ds)

    obs_names = build_feature_names(mapping["observation_state"]["components"], ds)
    act_names = build_feature_names(mapping["action"]["components"], ds)

    # --- Construct features dict ---
    features = {}

    # Image features (video type)
    for cam in mapping["images"]["cameras"]:
        key = cam["key"]
        h, w, c = image_dims[key]
        features[f"observation.images.{key}"] = {
            "dtype": "video",
            "shape": [h, w, c],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": float(fps),
                "video.codec": mapping["images"].get("video_codec", "libx264").replace("lib", ""),
                "video.pix_fmt": mapping["images"].get("video_pix_fmt", "yuv420p"),
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    # Observation state
    features["observation.state"] = {
        "dtype": "float32",
        "shape": [obs_dim],
        "names": {"motors": obs_names},
    }

    # Action
    features["action"] = {
        "dtype": "float32",
        "shape": [act_dim],
        "names": {"motors": act_names},
    }

    # Extra observations
    for extra in mapping.get("extra_observations", []):
        path = extra["hdf5_path"]
        ed = ds.get(path)
        if ed:
            dim = ed["shape"][-1] if ed["ndims"] >= 2 else 1
            features[extra["feature_name"]] = {
                "dtype": extra.get("dtype", "float32"),
                "shape": [dim],
                "names": None,
            }

    # Language instruction (per-frame string column)
    lang_cfg = mapping.get("language_instructions", {})
    if lang_cfg.get("per_frame", False):
        features["language_instruction"] = {
            "dtype": "string",
            "shape": [1],
            "names": None,
        }

    # Built-in index columns
    features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    features["frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["episode_index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["task_index"] = {"dtype": "int64", "shape": [1], "names": None}

    num_cameras = len(mapping["images"]["cameras"])
    chunks = (total_episodes + mapping["dataset"]["chunks_size"] - 1) // mapping["dataset"]["chunks_size"]

    info = {
        "codebase_version": mapping["dataset"].get("codebase_version", "v2.1"),
        "robot_type": mapping["dataset"].get("robot_type", "unknown"),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * num_cameras,
        "total_chunks": chunks,
        "chunks_size": mapping["dataset"]["chunks_size"],
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    return info
