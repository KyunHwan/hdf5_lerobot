"""CLI entry point for hdf5_lerobot.

Supports two subcommands:
    convert       — Convert HDF5 dataset to LeRobot v2.1
    list-schemas  — List available schema bundles
"""

import argparse
import logging
import sys

from .core.converter import convert_dataset
from .core.io_utils import load_json, load_yaml
from .hooks import DefaultHooks
from .registry import list_schemas, load_hooks_from_file, load_schema_bundle

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hdf5_lerobot",
        description="Convert custom HDF5 robot datasets to LeRobot v2.1 format.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- convert ---
    conv = subparsers.add_parser(
        "convert",
        help="Convert HDF5 dataset to LeRobot v2.1",
    )
    conv.add_argument(
        "--hdf5-dir", required=True,
        help="Directory containing .hdf5 episode files",
    )
    conv.add_argument(
        "--output-dir", required=True,
        help="Output directory for the LeRobot dataset",
    )

    # Schema source: either --schema-name OR --schema + --mapping
    source = conv.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--schema-name",
        help="Name of a schema bundle in schemas/ (e.g. 'picknplace')",
    )
    source.add_argument(
        "--schema",
        help="Path to schema.json (use with --mapping)",
    )
    conv.add_argument(
        "--mapping",
        help="Path to mapping.yaml (required when using --schema)",
    )
    conv.add_argument(
        "--hooks",
        help="Path to custom hooks.py (only with --schema/--mapping mode)",
    )
    conv.add_argument(
        "--schemas-dir",
        help="Custom schemas directory (default: schemas/ in repo root)",
    )

    # Hub options
    conv.add_argument(
        "--push-to-hub", action="store_true",
        help="Upload to HuggingFace Hub after conversion",
    )
    conv.add_argument(
        "--repo-id",
        help="HuggingFace repo ID (overrides dataset.repo_id in mapping config)",
    )
    conv.add_argument(
        "--hub-token",
        help="HuggingFace API token",
    )
    conv.add_argument(
        "--private", action="store_true",
        help="Create the HuggingFace repo as private",
    )
    conv.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    # --- list-schemas ---
    ls = subparsers.add_parser(
        "list-schemas",
        help="List available schema bundles",
    )
    ls.add_argument(
        "--schemas-dir",
        help="Custom schemas directory",
    )

    return parser


def _cmd_convert(args):
    """Handle the 'convert' subcommand."""
    from pathlib import Path

    schemas_dir = Path(args.schemas_dir) if args.schemas_dir else None

    if args.schema_name:
        # Mode 1: Load from schema bundle
        schema, mapping, hooks = load_schema_bundle(args.schema_name, schemas_dir)
        schema_path = ""
        mapping_path = ""
    else:
        # Mode 2: Explicit paths
        if not args.mapping:
            print("Error: --mapping is required when using --schema", file=sys.stderr)
            sys.exit(1)
        schema_path = args.schema
        mapping_path = args.mapping
        schema = load_json(schema_path)
        mapping = load_yaml(mapping_path)

        if args.hooks:
            hooks = load_hooks_from_file(args.hooks)
        else:
            hooks = DefaultHooks()

    convert_dataset(
        hdf5_dir=args.hdf5_dir,
        schema_path=schema_path,
        mapping_path=mapping_path,
        output_dir=args.output_dir,
        hooks=hooks,
        push_to_hub=args.push_to_hub,
        repo_id_override=args.repo_id,
        hub_token=args.hub_token,
        private=args.private,
        schema=schema,
        mapping=mapping,
    )


def _cmd_list_schemas(args):
    """Handle the 'list-schemas' subcommand."""
    from pathlib import Path

    schemas_dir = Path(args.schemas_dir) if args.schemas_dir else None
    names = list_schemas(schemas_dir)

    if not names:
        print("No schema bundles found.")
        print("Create a directory under schemas/ with at least a mapping.yaml file.")
        return

    print(f"Available schemas ({len(names)}):")
    for name in names:
        print(f"  - {name}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if hasattr(args, "log_level"):
        logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.command == "convert":
        _cmd_convert(args)
    elif args.command == "list-schemas":
        _cmd_list_schemas(args)


if __name__ == "__main__":
    main()
