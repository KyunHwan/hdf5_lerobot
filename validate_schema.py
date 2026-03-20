#!/usr/bin/env python3
"""Derive an HDF5 schema from a single reference file and validate other files against it."""

import argparse
import json
import os
import sys
from collections import defaultdict

import h5py


def build_schema(filepath):
    """Extract a schema from a single reference HDF5 file."""
    groups = []
    datasets = {}

    def visitor(name, obj):
        path = f"/{name}"
        if isinstance(obj, h5py.Group):
            groups.append(path)
        elif isinstance(obj, h5py.Dataset):
            datasets[path] = {
                "dtype_kind": obj.dtype.kind,
                "dtype_itemsize": obj.dtype.itemsize,
                "ndims": len(obj.shape),
                "shape": obj.shape,
            }

    with h5py.File(filepath, "r") as f:
        f.visititems(visitor)

    groups.sort()
    return {
        "reference_file": os.path.basename(filepath),
        "groups": groups,
        "datasets": datasets,
    }


def save_schema(schema, output_path):
    """Save the schema to a JSON file."""
    serializable = {
        "reference_file": schema["reference_file"],
        "groups": schema["groups"],
        "datasets": {
            path: {
                "dtype_kind": info["dtype_kind"],
                "dtype_itemsize": info["dtype_itemsize"],
                "ndims": info["ndims"],
                "shape": list(info["shape"]),
            }
            for path, info in schema["datasets"].items()
        },
    }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_schema(filepath):
    """Load a schema from a JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    # Convert shape lists back to tuples
    for info in data["datasets"].values():
        info["shape"] = tuple(info["shape"])
    return data


def format_schema(schema):
    """Format the schema as markdown text."""
    kind_map = {"f": "float", "i": "int", "S": "string", "b": "bool", "u": "uint", "O": "object"}
    lines = []
    lines.append(f"## Schema (derived from `{schema['reference_file']}`)")
    lines.append("")
    lines.append(f"### Groups ({len(schema['groups'])})")
    lines.append("")
    for g in schema["groups"]:
        lines.append(f"- `{g}`")
    lines.append("")
    lines.append(f"### Datasets ({len(schema['datasets'])})")
    lines.append("")
    lines.append("| Dataset | Dtype | Shape |")
    lines.append("|---------|-------|-------|")
    for path in sorted(schema["datasets"]):
        info = schema["datasets"][path]
        kind_str = kind_map.get(info["dtype_kind"], info["dtype_kind"])
        dtype_str = f"{kind_str}{info['dtype_itemsize'] * 8}"
        shape_str = "scalar" if info["ndims"] == 0 else f"{info['shape']}"
        lines.append(f"| `{path}` | {dtype_str} | {shape_str} |")
    return "\n".join(lines)


def validate_file(filepath, schema):
    """Validate a file against the schema. Returns (missing, extra, dtype_mismatches, ndims_mismatches)."""
    missing = []
    extra = []
    dtype_mismatches = []
    ndims_mismatches = []

    try:
        with h5py.File(filepath, "r") as f:
            actual_datasets = set()

            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    actual_datasets.add(f"/{name}")

            f.visititems(visitor)

            expected = set(schema["datasets"].keys())
            missing = sorted(expected - actual_datasets)
            extra = sorted(actual_datasets - expected)

            for path in sorted(expected & actual_datasets):
                expected_info = schema["datasets"][path]
                ds = f[path]

                kind_ok = ds.dtype.kind == expected_info["dtype_kind"]
                size_ok = ds.dtype.kind in ("O",) or ds.dtype.itemsize == expected_info["dtype_itemsize"]
                if not kind_ok or not size_ok:
                    dtype_mismatches.append(
                        (path, expected_info["dtype_kind"], expected_info["dtype_itemsize"],
                         ds.dtype.kind, ds.dtype.itemsize)
                    )

                if len(ds.shape) != expected_info["ndims"]:
                    ndims_mismatches.append(
                        (path, expected_info["ndims"], len(ds.shape))
                    )

    except Exception as e:
        missing.append(f"OPEN_ERROR: {e}")

    return missing, extra, dtype_mismatches, ndims_mismatches


def find_hdf5_files(folder):
    """Find all .hdf5 files in a folder (non-recursive)."""
    files = []
    for name in sorted(os.listdir(folder)):
        if name.endswith(".hdf5"):
            files.append(os.path.join(folder, name))
    return files


def write_report(output_path, schema, results, ref_path, total_files):
    """Write the validation report as a markdown file."""
    # Aggregate: problem -> list of files
    missing_datasets = defaultdict(list)     # dataset_path -> [filenames]
    extra_datasets = defaultdict(list)       # dataset_path -> [filenames]
    dtype_issues = defaultdict(list)         # (path, expected, got) -> [filenames]
    ndims_issues = defaultdict(list)         # (path, expected, got) -> [filenames]

    passed_files = []
    failed_files = []

    for filepath, (missing, extra, dtype_mm, ndims_mm) in results:
        basename = os.path.basename(filepath)
        has_errors = missing or dtype_mm or ndims_mm

        if has_errors:
            failed_files.append(basename)
        else:
            passed_files.append(basename)

        for ds in missing:
            missing_datasets[ds].append(basename)
        for ds in extra:
            extra_datasets[ds].append(basename)
        for path, exp_kind, exp_size, got_kind, got_size in dtype_mm:
            key = (path, f"{exp_kind}{exp_size}", f"{got_kind}{got_size}")
            dtype_issues[key].append(basename)
        for path, exp_ndims, got_ndims in ndims_mm:
            key = (path, exp_ndims, got_ndims)
            ndims_issues[key].append(basename)

    lines = []
    lines.append("# HDF5 Schema Validation Report")
    lines.append("")
    lines.append(f"- **Reference file:** `{os.path.basename(ref_path)}`")
    lines.append(f"- **Total files:** {total_files}")
    lines.append(f"- **Validated:** {len(results)}")
    lines.append(f"- **Passed:** {len(passed_files)}")
    lines.append(f"- **Failed:** {len(failed_files)}")
    lines.append("")

    # Schema section
    lines.append(format_schema(schema))
    lines.append("")

    # Missing datasets section
    if missing_datasets:
        lines.append("---")
        lines.append("")
        lines.append("## Missing Datasets")
        lines.append("")
        lines.append("Datasets present in the schema but missing from these files.")
        lines.append("")
        for ds_path in sorted(missing_datasets):
            files = missing_datasets[ds_path]
            lines.append(f"### `{ds_path}`")
            lines.append("")
            lines.append(f"Missing from **{len(files)}** file(s):")
            lines.append("")
            for f in files:
                lines.append(f"- `{f}`")
            lines.append("")

    # Extra datasets section
    if extra_datasets:
        lines.append("---")
        lines.append("")
        lines.append("## Extra Datasets (not in schema)")
        lines.append("")
        lines.append("Datasets found in files but not present in the reference schema.")
        lines.append("")
        for ds_path in sorted(extra_datasets):
            files = extra_datasets[ds_path]
            lines.append(f"### `{ds_path}`")
            lines.append("")
            lines.append(f"Found in **{len(files)}** file(s):")
            lines.append("")
            for f in files:
                lines.append(f"- `{f}`")
            lines.append("")

    # Dtype mismatches section
    if dtype_issues:
        lines.append("---")
        lines.append("")
        lines.append("## Dtype Mismatches")
        lines.append("")
        for (path, expected, got), files in sorted(dtype_issues.items()):
            lines.append(f"### `{path}` (expected `{expected}`, got `{got}`)")
            lines.append("")
            lines.append(f"Affects **{len(files)}** file(s):")
            lines.append("")
            for f in files:
                lines.append(f"- `{f}`")
            lines.append("")

    # Ndims mismatches section
    if ndims_issues:
        lines.append("---")
        lines.append("")
        lines.append("## Ndims Mismatches")
        lines.append("")
        for (path, expected, got), files in sorted(ndims_issues.items()):
            lines.append(f"### `{path}` (expected {expected}D, got {got}D)")
            lines.append("")
            lines.append(f"Affects **{len(files)}** file(s):")
            lines.append("")
            for f in files:
                lines.append(f"- `{f}`")
            lines.append("")

    # Failed files list
    if failed_files:
        lines.append("---")
        lines.append("")
        lines.append("## Failed Files")
        lines.append("")
        for f in failed_files:
            lines.append(f"- `{f}`")
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Validate HDF5 files against a schema derived from one file.")
    parser.add_argument("folder", help="Folder containing HDF5 files")
    parser.add_argument(
        "--reference",
        default=None,
        help="Specific file to use as schema reference (default: first .hdf5 file found)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write schema.json and validation_report.md (default: same as folder)",
    )
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a directory", file=sys.stderr)
        sys.exit(1)

    all_files = find_hdf5_files(folder)
    if not all_files:
        print(f"Error: no .hdf5 files found in {folder}", file=sys.stderr)
        sys.exit(1)

    # Determine reference file
    if args.reference:
        ref_path = os.path.join(folder, args.reference) if not os.path.isabs(args.reference) else args.reference
        if not os.path.isfile(ref_path):
            print(f"Error: reference file {ref_path} not found", file=sys.stderr)
            sys.exit(1)
    else:
        ref_path = all_files[0]

    # Output directory
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else folder
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reference file: {os.path.basename(ref_path)}")
    print(f"Total HDF5 files: {len(all_files)}")

    # Build schema and save to JSON
    schema = build_schema(ref_path)

    schema_path = os.path.join(output_dir, "schema.json")
    save_schema(schema, schema_path)
    print(f"Schema saved to: {schema_path}")

    # Validate all other files
    files_to_validate = [f for f in all_files if f != ref_path]
    results = []
    passed = 0
    failed = 0

    print(f"Validating {len(files_to_validate)} files...")

    for filepath in files_to_validate:
        missing, extra, dtype_mm, ndims_mm = validate_file(filepath, schema)
        results.append((filepath, (missing, extra, dtype_mm, ndims_mm)))
        if missing or dtype_mm or ndims_mm:
            failed += 1
        else:
            passed += 1

    # Write report
    report_path = os.path.join(output_dir, "validation_report.md")
    write_report(report_path, schema, results, ref_path, len(all_files))

    print(f"\nSUMMARY: {passed}/{passed + failed} files passed")
    if failed:
        print(f"         {failed} file(s) failed")
    print(f"\nReport written to: {report_path}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
