#!/usr/bin/env python3
"""Backward-compatible entry point for HDF5 → LeRobot conversion.

This shim preserves the original CLI interface::

    python hdf5_to_lerobot.py \\
        --hdf5-dir /path/to/hdf5_files/ \\
        --schema schema.json \\
        --mapping mapping_config.yaml \\
        --output-dir /path/to/output_dataset/

For the new CLI with schema-bundle support, use::

    python -m hdf5_lerobot convert --schema-name picknplace --hdf5-dir ... --output-dir ...
"""

import argparse
import logging

from hdf5_lerobot.core.converter import convert_dataset
from hdf5_lerobot.hooks import DefaultHooks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Convert custom HDF5 robot datasets to LeRobot v2.1 format."
    )
    parser.add_argument(
        "--hdf5-dir", required=True,
        help="Directory containing .hdf5 episode files",
    )
    parser.add_argument(
        "--schema", required=True,
        help="Path to schema.json (HDF5 structure reference)",
    )
    parser.add_argument(
        "--mapping", required=True,
        help="Path to mapping_config.yaml (HDF5→LeRobot field mapping)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for the LeRobot dataset",
    )
    parser.add_argument(
        "--push-to-hub", action="store_true",
        help="Upload to HuggingFace Hub after conversion",
    )
    parser.add_argument(
        "--repo-id",
        help="HuggingFace repo ID (overrides dataset.repo_id in mapping config)",
    )
    parser.add_argument(
        "--hub-token",
        help="HuggingFace API token (default: cached token from 'huggingface-cli login')",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create the HuggingFace repo as private",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    convert_dataset(
        hdf5_dir=args.hdf5_dir,
        schema_path=args.schema,
        mapping_path=args.mapping,
        output_dir=args.output_dir,
        hooks=DefaultHooks(),
        push_to_hub=args.push_to_hub,
        repo_id_override=args.repo_id,
        hub_token=args.hub_token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
