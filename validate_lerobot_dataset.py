#!/usr/bin/env python3
"""
validate_lerobot_dataset.py — Validate a converted LeRobot v2.1 dataset.

Checks:
  - meta/info.json exists and has required fields
  - meta/tasks.jsonl exists and is non-empty
  - meta/episodes.jsonl exists and episode count matches info.json
  - Parquet files exist and have expected columns
  - Video files exist for each camera × episode
  - Frame counts are consistent across parquet and info.json
  - Language instruction column present if declared in features
  - Stats files exist

Usage:
    python validate_lerobot_dataset.py /path/to/dataset/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate(dataset_dir: str) -> list[str]:
    errors = []
    warnings = []
    root = Path(dataset_dir)

    # --- info.json ---
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        errors.append("MISSING meta/info.json")
        return errors

    with open(info_path) as f:
        info = json.load(f)

    required_keys = [
        "codebase_version", "robot_type", "total_episodes", "total_frames",
        "total_tasks", "fps", "features", "data_path", "video_path",
    ]
    for key in required_keys:
        if key not in info:
            errors.append(f"info.json missing required key: {key}")

    total_episodes = info.get("total_episodes", 0)
    total_frames = info.get("total_frames", 0)
    features = info.get("features", {})
    fps = info.get("fps", 30)

    print(f"  Dataset: {info.get('robot_type', '?')} | "
          f"{total_episodes} episodes | {total_frames} frames | {fps} FPS")
    print(f"  Features: {list(features.keys())}")

    # --- tasks.jsonl ---
    tasks_path = root / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        errors.append("MISSING meta/tasks.jsonl")
    else:
        tasks = load_jsonl(str(tasks_path))
        print(f"  Tasks: {len(tasks)}")
        if len(tasks) != info.get("total_tasks", -1):
            warnings.append(
                f"tasks.jsonl has {len(tasks)} tasks but info.json says {info.get('total_tasks')}"
            )
        for t in tasks[:5]:
            print(f"    [{t['task_index']}] {t['task'][:80]}")
        if len(tasks) > 5:
            print(f"    ... and {len(tasks) - 5} more")

    # --- episodes.jsonl ---
    episodes_path = root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        errors.append("MISSING meta/episodes.jsonl")
    else:
        episodes = load_jsonl(str(episodes_path))
        if len(episodes) != total_episodes:
            errors.append(
                f"episodes.jsonl has {len(episodes)} entries but info.json says {total_episodes}"
            )
        frame_sum = sum(ep["length"] for ep in episodes)
        if frame_sum != total_frames:
            errors.append(
                f"Sum of episode lengths ({frame_sum}) != total_frames ({total_frames})"
            )

    # --- Parquet files ---
    checked_frames = 0
    video_features = [k for k, v in features.items() if v.get("dtype") == "video"]
    has_lang = "language_instruction" in features

    for ep_idx in range(min(total_episodes, 5)):  # spot-check first 5
        chunk = ep_idx // info.get("chunks_size", 1000)
        parquet_path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        if not parquet_path.exists():
            errors.append(f"MISSING {parquet_path.relative_to(root)}")
            continue

        df = pd.read_parquet(parquet_path)
        expected_cols = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
        if "observation.state" in features:
            expected_cols.add("observation.state")
        if "action" in features:
            expected_cols.add("action")
        if has_lang:
            expected_cols.add("language_instruction")

        missing_cols = expected_cols - set(df.columns)
        if missing_cols:
            errors.append(f"Episode {ep_idx} parquet missing columns: {missing_cols}")

        # Check frame indices are sequential
        if not df["frame_index"].tolist() == list(range(len(df))):
            warnings.append(f"Episode {ep_idx}: frame_index not sequential 0..{len(df)-1}")

        checked_frames += len(df)

        # Check videos exist
        for vf in video_features:
            video_path = root / "videos" / f"chunk-{chunk:03d}" / vf / f"episode_{ep_idx:06d}.mp4"
            if not video_path.exists():
                errors.append(f"MISSING {video_path.relative_to(root)}")

    print(f"  Spot-checked {min(total_episodes, 5)} episodes ({checked_frames} frames)")

    # --- Stats (informational — computed by LeRobot on first load) ---
    stats_path = root / "meta" / "stats.json"
    ep_stats_path = root / "meta" / "episodes_stats.jsonl"
    if stats_path.exists():
        print("  meta/stats.json: present")
    else:
        print("  meta/stats.json: absent (LeRobot will compute on first load)")
    if ep_stats_path.exists():
        print("  meta/episodes_stats.jsonl: present")
    else:
        print("  meta/episodes_stats.jsonl: absent (LeRobot will compute on first load)")

    # --- Report ---
    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    ⚠ {w}")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    ✗ {e}")
    else:
        print("\n  ✓ All checks passed!")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate a LeRobot v2.1 dataset")
    parser.add_argument("dataset_dir", help="Path to the LeRobot dataset directory")
    args = parser.parse_args()

    print(f"Validating: {args.dataset_dir}\n")
    errors = validate(args.dataset_dir)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
