# hdf5_lerobot

Convert custom HDF5 robot datasets to [LeRobot v2.1](https://github.com/huggingface/lerobot) format.

**hdf5_lerobot** takes your robot's HDF5 episode files — containing joint positions, actions, camera images, and task metadata — and produces a standards-compliant LeRobot dataset with Parquet data files, MP4 videos, and JSON metadata. The output is ready for training with LeRobot or sharing on HuggingFace Hub.

## Features

- **Schema bundles** — Package your HDF5 structure definition, field mapping, and custom logic together as a reusable schema bundle in `schemas/`
- **Pluggable hooks** — Inject custom Python logic for filtering episodes, transforming observations/actions, processing image frames, and adding derived columns
- **6 language instruction strategies** — Extract task descriptions from fixed strings, HDF5 datasets, HDF5 attributes, filename regex, external JSON files, or subtask status fields
- **Video encoding** — Encode camera streams to MP4 via FFmpeg with configurable codec, pixel format, and quality
- **HuggingFace Hub integration** — Push converted datasets directly to HuggingFace Hub
- **Validation tools** — Validate HDF5 file consistency and verify the converted LeRobot dataset structure
- **Two-pass conversion** — First pass discovers tasks and counts frames; second pass writes data efficiently

## Prerequisites

- **Python 3.10+**
- **FFmpeg** — must be available on your `PATH` (used for video encoding)
- **uv** (recommended) or **pip** for package management

Install FFmpeg if you don't have it:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Verify
ffmpeg -version
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd hdf5_lerobot

# Option 1: Using uv (recommended)
uv pip install -e .

# Option 2: Using pip
pip install -e .
```

This installs the `hdf5-lerobot` CLI command and all Python dependencies: `h5py`, `numpy`, `pandas`, `pyarrow`, `pyyaml`, `Pillow`, `tqdm`, and `huggingface_hub`.

## Quick Start

If you already understand your HDF5 file structure, here's the fastest path to a converted dataset:

```bash
# 1. Generate a schema.json from a reference HDF5 file
python validate_schema.py /path/to/hdf5_files/ --output-dir schemas/my_robot/

# 2. Create your mapping config (copy and edit the example)
cp schemas/picknplace/mapping.yaml schemas/my_robot/mapping.yaml
# Edit schemas/my_robot/mapping.yaml to match your HDF5 paths

# 3. Convert
hdf5-lerobot convert \
    --schema-name my_robot \
    --hdf5-dir /path/to/hdf5_files/ \
    --output-dir /path/to/output/

# 4. Validate the output
python validate_lerobot_dataset.py /path/to/output/
```

## Project Structure

```
hdf5_lerobot/
├── hdf5_lerobot/                  # Main Python package
│   ├── __init__.py
│   ├── __main__.py                # Enables `python -m hdf5_lerobot`
│   ├── cli.py                     # CLI: `convert` and `list-schemas` commands
│   ├── registry.py                # Schema bundle discovery and loading
│   ├── hooks.py                   # Hook protocol (ConverterHooks) and DefaultHooks
│   ├── language.py                # 6 built-in language instruction extraction strategies
│   └── core/
│       ├── converter.py           # Main orchestrator — two-pass conversion engine
│       ├── features.py            # Builds info.json feature definitions from mapping config
│       ├── hdf5_utils.py          # HDF5 reading: image decoding, dataset concatenation
│       ├── io_utils.py            # YAML/JSON I/O and natural sort utilities
│       ├── video.py               # FFmpegVideoWriter — subprocess-based MP4 encoding
│       └── hub.py                 # HuggingFace Hub upload
├── schemas/                       # Schema bundles (one subdirectory per robot/dataset type)
│   ├── README.md                  # Guide for creating new schema bundles
│   └── picknplace/                # Example: bimanual pick-and-place robot
│       ├── schema.json            # HDF5 structure reference (auto-generated)
│       ├── mapping.yaml           # HDF5 → LeRobot field mapping configuration
│       └── hooks.py               # Custom conversion hooks (optional)
├── test_data/                     # Sample HDF5 file for testing
├── hdf5_to_lerobot.py            # Legacy entry point (backward-compatible)
├── validate_schema.py             # Generate schema.json and validate HDF5 consistency
├── validate_lerobot_dataset.py    # Validate converted LeRobot dataset structure
└── pyproject.toml                 # Package metadata and dependencies
```

## End-to-End Walkthrough: Converting Your Custom Dataset

This section walks through every step of converting a custom HDF5 robot dataset to LeRobot format.

### Step 1: Inspect Your HDF5 Files

Before configuring anything, understand what's inside your HDF5 files. Use Python with h5py to explore:

```python
import h5py

# Open a representative episode file
with h5py.File("my_episode_001.hdf5", "r") as f:
    # Print all groups and datasets
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
        else:
            print(f"  {name}/ (group)")
    f.visititems(print_structure)
```

Example output for a bimanual robot:

```
  action/joint_pos/left: shape=(553, 6), dtype=float32
  action/joint_pos/right: shape=(553, 6), dtype=float32
  action/hand_joint_pos/left: shape=(553, 6), dtype=float32
  action/hand_joint_pos/right: shape=(553, 6), dtype=float32
  observation/joint_pos/left: shape=(553, 6), dtype=float32
  observation/joint_pos/right: shape=(553, 6), dtype=float32
  observation/hand_joint_pos/left: shape=(553, 6), dtype=float32
  observation/hand_joint_pos/right: shape=(553, 6), dtype=float32
  observation/images/head: shape=(553, 921600), dtype=uint8
  observation/images/left: shape=(553, 921600), dtype=uint8
  compressed_image_len: shape=(3, 553), dtype=int32
  metadata/HZ: shape=(), dtype=float64
  metadata/data_collection_error: shape=(), dtype=bool
```

Key things to identify:
- **Observation paths** — which datasets hold joint positions, end-effector poses, etc.
- **Action paths** — which datasets hold action targets
- **Image paths** — where camera images are stored
- **Image format** — are images JPEG/PNG-compressed bytes (variable length per frame) or raw pixel arrays?
- **FPS** — is there a metadata field with the recording frequency?
- **Frame count** — the first dimension (T) of time-series datasets
- **Task/language info** — any task descriptions, subtask statuses, or language annotations

### Step 2: Generate schema.json

Run `validate_schema.py` on your HDF5 directory. It picks the first file as a reference, extracts the full HDF5 structure, and saves it as `schema.json`:

```bash
# Generate schema from all HDF5 files in a directory
python validate_schema.py /path/to/hdf5_files/ --output-dir schemas/my_robot/

# Or specify a particular reference file
python validate_schema.py /path/to/hdf5_files/ \
    --reference my_best_episode.hdf5 \
    --output-dir schemas/my_robot/
```

This produces two files:
- `schemas/my_robot/schema.json` — the HDF5 structure (groups, datasets, dtypes, shapes)
- `validation_report.md` — a report showing which files match the reference and which have discrepancies

Review the validation report. Files with missing datasets, dtype mismatches, or shape mismatches may cause conversion errors and should be investigated or excluded.

### Step 3: Create mapping.yaml

The mapping file tells the converter how your HDF5 fields map to LeRobot features. Copy the example and edit it:

```bash
cp schemas/picknplace/mapping.yaml schemas/my_robot/mapping.yaml
```

Edit each section to match your HDF5 structure. Here's a complete annotated walkthrough:

#### Dataset metadata

```yaml
dataset:
  repo_id: "my_org/my_robot_dataset"       # HuggingFace repo ID (used in info.json)
  robot_type: "my_custom_robot"             # Free-text label for your robot
  codebase_version: "v2.1"                  # Target LeRobot format version
  chunks_size: 1000                         # Episodes per chunk directory

  fps:
    source: "hdf5"                          # "fixed" or "hdf5"
    hdf5_path: "/metadata/HZ"              # Path to FPS value in HDF5 (when source=hdf5)
    # fixed_value: 30                       # Use this instead when source=fixed
```

#### Observation state

List the HDF5 datasets to concatenate into `observation.state`. Order matters — this defines the feature vector layout:

```yaml
observation_state:
  components:
    - hdf5_path: "/observation/joint_pos/left"
      slice: null                           # null = all columns; or [start, end]
      names_prefix: "left_joint"            # Generates left_joint_0, left_joint_1, ...
    - hdf5_path: "/observation/joint_pos/right"
      slice: null
      names_prefix: "right_joint"
```

If your dataset has shape `[T, 7]` but you only want the first 6 columns, use `slice: [0, 6]`.

#### Actions

Same structure as observation_state. These are concatenated into the `action` feature:

```yaml
action:
  components:
    - hdf5_path: "/action/joint_pos/left"
      slice: null
      names_prefix: "left_joint"
    - hdf5_path: "/action/joint_pos/right"
      slice: null
      names_prefix: "right_joint"
```

#### Extra observations (optional)

Additional per-frame numeric features beyond `observation.state`. Each becomes its own column:

```yaml
extra_observations:
  - feature_name: "observation.xpos.left"
    hdf5_path: "/observation/xpos/left"
    dtype: "float32"
  - feature_name: "observation.quaternion.left"
    hdf5_path: "/observation/quaternion/left"
    dtype: "float32"
```

#### Camera images

Define your cameras and image format:

```yaml
images:
  # For JPEG/PNG-compressed images stored as byte arrays:
  compressed: true
  compressed_len_path: "/compressed_image_len"  # [num_cameras, num_frames] byte lengths

  # For raw (uncompressed) pixel arrays, use:
  # compressed: false
  # raw_height: 480
  # raw_width: 640
  # raw_channels: 3

  cameras:
    - key: "head"                           # Becomes observation.images.head
      hdf5_path: "/observation/images/head"
      compressed_len_index: 0               # Row index in compressed_image_len
    - key: "left"
      hdf5_path: "/observation/images/left"
      compressed_len_index: 1
    - key: "right"
      hdf5_path: "/observation/images/right"
      compressed_len_index: 2

  # Video encoding settings
  video_codec: "libx264"                    # libx264, libsvtav1, or libx265
  video_pix_fmt: "yuv420p"
  video_crf: 23                             # 0-51, lower = better quality, larger files
```

**Compressed images**: Each row in the image dataset is a flat byte array containing the JPEG/PNG bytes followed by padding. The `compressed_len_path` dataset stores the actual byte length per frame so the converter knows where the image data ends.

**Raw images**: Each row is a flat array of `height * width * channels` uint8 values that gets reshaped to `[H, W, C]`.

#### Language instructions

Choose one of 6 strategies for extracting task descriptions:

```yaml
language_instructions:
  source: "fixed"                           # See "Language Instruction Strategies" below
  fixed:
    instruction: "pick up the red block"
  per_frame: true                           # Write language_instruction column in parquet
  use_task_index: true                      # Always writes task_index (for documentation)
```

See the [Language Instruction Strategies](#language-instruction-strategies) section for all options.

#### Filtering (optional)

Skip bad episodes automatically:

```yaml
filtering:
  skip_on_error: true                       # Skip if error flag is true
  error_flag_path: "/metadata/data_collection_error"
  min_frames: 10                            # Skip episodes shorter than 10 frames
```

### Step 4: (Optional) Write Custom Hooks

If your conversion needs logic that YAML can't express, create `schemas/my_robot/hooks.py`:

```python
import numpy as np

class Hooks:
    def filter_episode(self, h5f, mapping):
        """Return False to skip this episode."""
        if "/metadata/quality_score" in h5f:
            return float(h5f["/metadata/quality_score"][()]) >= 0.5
        return True

    def transform_observation_state(self, obs, h5f, mapping):
        """Normalize joint positions to [-1, 1]."""
        joint_limits = np.array([3.14, 3.14, 3.14, 2.0, 2.0, 2.0])
        obs[:, :6] = obs[:, :6] / joint_limits
        return obs
```

Only define the methods you need — undefined methods fall back to no-op defaults. See the [Hook System](#hook-system) section for all available hooks.

### Step 5: Run the Conversion

```bash
# Using a schema bundle (recommended)
hdf5-lerobot convert \
    --schema-name my_robot \
    --hdf5-dir /path/to/hdf5_files/ \
    --output-dir /path/to/output/

# Or using explicit file paths
hdf5-lerobot convert \
    --schema schemas/my_robot/schema.json \
    --mapping schemas/my_robot/mapping.yaml \
    --hooks schemas/my_robot/hooks.py \
    --hdf5-dir /path/to/hdf5_files/ \
    --output-dir /path/to/output/
```

The converter runs in two passes:
1. **Pass 1 (Scanning)** — Filters episodes, extracts language instructions, discovers unique tasks, counts frames
2. **Pass 2 (Converting)** — Reads HDF5 data, writes Parquet files, encodes MP4 videos

### Step 6: Validate the Output

```bash
python validate_lerobot_dataset.py /path/to/output/
```

This checks:
- `meta/info.json` exists with all required fields
- `meta/tasks.jsonl` task count matches info.json
- `meta/episodes.jsonl` episode count and frame totals are consistent
- Parquet files exist with expected columns and sequential frame indices
- Video files exist for every camera × episode combination
- Language instruction column is present (if declared in features)

### Step 7: (Optional) Push to HuggingFace Hub

```bash
# During conversion
hdf5-lerobot convert \
    --schema-name my_robot \
    --hdf5-dir /path/to/hdf5_files/ \
    --output-dir /path/to/output/ \
    --push-to-hub \
    --repo-id my_org/my_robot_dataset \
    --hub-token hf_xxxxx \
    --private

# Or authenticate beforehand
huggingface-cli login
hdf5-lerobot convert \
    --schema-name my_robot \
    --hdf5-dir /path/to/hdf5_files/ \
    --output-dir /path/to/output/ \
    --push-to-hub
```

The `--repo-id` flag overrides `dataset.repo_id` from mapping.yaml. If neither is set, the upload will fail with an error.

### Loading the Dataset in LeRobot

After conversion, load the dataset in LeRobot:

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# From local files
ds = LeRobotDataset("my_org/my_robot_dataset", local_files_only=True, root="/path/to/output/")

# From HuggingFace Hub (if uploaded)
ds = LeRobotDataset("my_org/my_robot_dataset")
```

Note: Statistics (`meta/stats.json`) are not computed during conversion. LeRobot computes them automatically on first load.

## CLI Reference

### `hdf5-lerobot convert`

Convert HDF5 episode files to a LeRobot v2.1 dataset.

```
hdf5-lerobot convert [options]

Required:
  --hdf5-dir PATH           Directory containing .hdf5 episode files
  --output-dir PATH         Output directory for the LeRobot dataset

Schema source (mutually exclusive, one required):
  --schema-name NAME        Name of a schema bundle in schemas/ (e.g. "picknplace")
  --schema PATH             Path to schema.json (must be used with --mapping)

Optional:
  --mapping PATH            Path to mapping.yaml (required with --schema)
  --hooks PATH              Path to custom hooks.py (only with --schema/--mapping mode)
  --schemas-dir PATH        Custom schemas directory (default: schemas/ in repo root)

Hub options:
  --push-to-hub             Upload to HuggingFace Hub after conversion
  --repo-id ID              HuggingFace repo ID (overrides dataset.repo_id in config)
  --hub-token TOKEN         HuggingFace API token
  --private                 Create the HuggingFace repo as private

Logging:
  --log-level LEVEL         DEBUG, INFO (default), WARNING, or ERROR
```

### `hdf5-lerobot list-schemas`

List available schema bundles:

```
hdf5-lerobot list-schemas [--schemas-dir PATH]
```

### Alternative entry points

```bash
# Module invocation (equivalent to hdf5-lerobot CLI)
python -m hdf5_lerobot convert --schema-name picknplace ...
python -m hdf5_lerobot list-schemas

# Legacy script (backward-compatible, requires explicit paths)
python hdf5_to_lerobot.py \
    --hdf5-dir /path/to/files/ \
    --schema schemas/picknplace/schema.json \
    --mapping schemas/picknplace/mapping.yaml \
    --output-dir /path/to/output/
```

## Configuration Reference: mapping.yaml

Complete field reference for every section of the mapping configuration file.

### `dataset`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `repo_id` | string | Yes | HuggingFace repository ID (e.g. `"my_org/dataset_name"`) |
| `robot_type` | string | Yes | Free-text robot type label |
| `codebase_version` | string | No | LeRobot format version (default: `"v2.1"`) |
| `chunks_size` | int | Yes | Number of episodes per chunk directory |
| `fps.source` | `"fixed"` \| `"hdf5"` | Yes | How to determine frames per second |
| `fps.fixed_value` | int | When `source=fixed` | Static FPS value |
| `fps.hdf5_path` | string | When `source=hdf5` | HDF5 path to scalar FPS value |

### `observation_state`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `components` | list | Yes | HDF5 datasets to concatenate into `observation.state` |
| `components[].hdf5_path` | string | Yes | HDF5 dataset path (e.g. `"/observation/joint_pos/left"`) |
| `components[].slice` | `[start, end]` \| `null` | No | Column range to extract (default: all columns) |
| `components[].names_prefix` | string | No | Prefix for feature names (generates `prefix_0`, `prefix_1`, ...) |

Components are concatenated left-to-right along axis 1 to produce a single `[T, obs_dim]` array.

### `action`

Same structure as `observation_state`. Components are concatenated to produce `[T, act_dim]`.

### `extra_observations`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `[].feature_name` | string | Yes | Output column name (e.g. `"observation.xpos.left"`) |
| `[].hdf5_path` | string | Yes | HDF5 dataset path |
| `[].dtype` | string | No | NumPy dtype (default: `"float32"`) |

Each entry becomes a separate feature column in the Parquet output. Shape is inferred from the HDF5 dataset.

### `images`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `compressed` | bool | Yes | `true` if images are JPEG/PNG bytes; `false` if raw pixels |
| `compressed_len_path` | string | When compressed | HDF5 path to `[num_cameras, T]` byte length array |
| `raw_height` | int | When not compressed | Image height in pixels |
| `raw_width` | int | When not compressed | Image width in pixels |
| `raw_channels` | int | When not compressed | Number of color channels (typically 3) |
| `cameras` | list | Yes | Camera definitions |
| `cameras[].key` | string | Yes | Camera name (becomes `observation.images.<key>`) |
| `cameras[].hdf5_path` | string | Yes | HDF5 dataset path |
| `cameras[].compressed_len_index` | int | When compressed | Row index in the byte length array |
| `video_codec` | string | No | FFmpeg codec: `"libx264"` (default), `"libsvtav1"`, `"libx265"` |
| `video_pix_fmt` | string | No | Pixel format (default: `"yuv420p"`) |
| `video_crf` | int | No | Quality: 0 (lossless) to 51 (worst), default 23 |

### `language_instructions`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | string | Yes | Extraction strategy (see below) |
| `per_frame` | bool | No | Write `language_instruction` column per parquet row |
| `use_task_index` | bool | No | Populate `task_index` column (always happens) |

Strategy-specific fields are documented in the [Language Instruction Strategies](#language-instruction-strategies) section.

### `filtering`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `skip_on_error` | bool | No | Skip episodes where error flag is `true` |
| `error_flag_path` | string | No | HDF5 path to boolean error flag (default: `"/metadata/data_collection_error"`) |
| `min_frames` | int | No | Skip episodes with fewer than this many frames |

## Language Instruction Strategies

The converter supports 6 built-in strategies for extracting language/task instructions. Set `language_instructions.source` in mapping.yaml to one of:

### `"fixed"` — Same instruction for all frames and episodes

```yaml
language_instructions:
  source: "fixed"
  fixed:
    instruction: "pick up the red block and place it on the target"
  per_frame: true
```

Use when every episode performs the same task.

### `"hdf5_dataset"` — Per-frame strings from an HDF5 dataset

```yaml
language_instructions:
  source: "hdf5_dataset"
  hdf5_dataset:
    path: "/language/instruction"           # String array of shape [T]
  per_frame: true
```

Use when your data collection system stores a language instruction string per frame. The dataset should be a 1D array of strings (or bytes) with length matching the number of frames.

### `"hdf5_attribute"` — Episode-level string from an HDF5 attribute

```yaml
language_instructions:
  source: "hdf5_attribute"
  hdf5_attribute:
    group: "/metadata"                      # Group or dataset containing the attribute
    attribute: "task_description"            # Attribute name
  per_frame: true
```

Use when each episode has a single task description stored as an HDF5 attribute. The same string is replicated for all frames.

### `"filename_regex"` — Extract from the HDF5 filename

```yaml
language_instructions:
  source: "filename_regex"
  filename_regex:
    pattern: "task_(.*?)_episode"           # Regex with one capture group
  per_frame: true
```

Use when the task name is embedded in the filename. For a file named `task_pick_red_block_episode_001.hdf5`, the regex above captures `"pick_red_block"`. If the regex doesn't match, the instruction defaults to `"unknown task"`.

### `"external_json"` — Load from a sidecar JSON file

```yaml
language_instructions:
  source: "external_json"
  external_json:
    path: "instructions.json"               # Path to JSON file
  per_frame: true
```

The JSON file maps episode IDs (filename stems without `.hdf5`) to instruction strings:

```json
{
  "episode_001": "pick up the red block",
  "episode_002": "stack the blue cube on the green cube",
  "episode_003": "push the cylinder to the right"
}
```

Use when instructions are annotated separately from the HDF5 files (e.g., by a human annotator).

### `"subtask_status"` — Derive from subtask status datasets

```yaml
language_instructions:
  source: "subtask_status"
  subtask_status:
    task_group: "/tasks/pick_and_place"
    subtask_names:
      pick: "pick up the object"
      place: "place the object"
      place_1: "place the object at target location"
    default_instruction: "pick and place the object"
  per_frame: true
```

This reads status datasets at `{task_group}/subtasks/{subtask_name}/status`. For each frame, if a subtask's status is `"active"`, `"in_progress"`, `"running"`, or `"started"`, the corresponding instruction is assigned. If no subtask is active, the `default_instruction` is used.

Use when your data collection system records per-frame subtask statuses — the instruction changes dynamically as the robot transitions between subtasks within an episode.

## Hook System

Hooks let you inject custom Python logic at specific points in the conversion pipeline. Create a `hooks.py` file in your schema bundle with a `Hooks` class:

```python
# schemas/my_robot/hooks.py
class Hooks:
    # Define only the methods you need.
    # Undefined methods use no-op defaults.
    ...
```

### Available Hooks

#### `filter_episode(self, h5f, mapping) -> bool`

Called after the HDF5 file is opened, before any data is read. Return `False` to skip this episode entirely.

**Parameters:**
- `h5f` — open `h5py.File` object
- `mapping` — the full mapping config dict

**Example: Skip episodes with low quality scores**

```python
def filter_episode(self, h5f, mapping):
    if "/metadata/quality_score" in h5f:
        return float(h5f["/metadata/quality_score"][()]) >= 0.5
    return True
```

#### `extract_language_instructions(self, h5f, num_frames, lang_cfg, episode_id, external_instructions) -> list[str] | None`

Override language instruction extraction. Return a list of `num_frames` strings, or `None` to fall back to the built-in strategy from mapping.yaml.

**Parameters:**
- `h5f` — open `h5py.File` object
- `num_frames` — number of frames in this episode
- `lang_cfg` — the `language_instructions` section from mapping.yaml
- `episode_id` — filename stem (e.g. `"episode_001"`)
- `external_instructions` — loaded JSON dict (if using `external_json` strategy), else `None`

**Example: Custom instruction logic**

```python
def extract_language_instructions(self, h5f, num_frames, lang_cfg,
                                   episode_id, external_instructions):
    task = h5f["/metadata/task_name"][()].decode("utf-8")
    variant = h5f["/metadata/variant"][()].decode("utf-8")
    instruction = f"{task} ({variant})"
    return [instruction] * num_frames
```

#### `transform_observation_state(self, obs, h5f, mapping) -> np.ndarray`

Post-process the concatenated observation array before writing to Parquet.

**Parameters:**
- `obs` — `np.ndarray` of shape `[T, obs_dim]`
- `h5f` — open `h5py.File` object
- `mapping` — the full mapping config dict

**Example: Normalize joint positions**

```python
def transform_observation_state(self, obs, h5f, mapping):
    import numpy as np
    # Normalize first 6 columns (joint angles) to [-1, 1]
    joint_limits = np.array([3.14, 3.14, 3.14, 2.0, 2.0, 2.0])
    obs[:, :6] = obs[:, :6] / joint_limits
    return obs
```

#### `transform_action(self, action, h5f, mapping) -> np.ndarray`

Post-process the concatenated action array before writing to Parquet.

**Parameters:**
- `action` — `np.ndarray` of shape `[T, act_dim]`
- `h5f` — open `h5py.File` object
- `mapping` — the full mapping config dict

**Example: Clip actions to safe range**

```python
def transform_action(self, action, h5f, mapping):
    import numpy as np
    return np.clip(action, -1.0, 1.0)
```

#### `transform_frame(self, frame, cam_key) -> np.ndarray`

Post-process an image frame before video encoding. Called once per frame per camera.

**Parameters:**
- `frame` — `np.ndarray` of shape `[H, W, C]`, dtype `uint8`, RGB
- `cam_key` — camera name string (e.g. `"head"`, `"left"`)

**Example: Crop and resize**

```python
def transform_frame(self, frame, cam_key):
    # Crop center 80% of the image
    h, w = frame.shape[:2]
    margin_h, margin_w = h // 10, w // 10
    frame = frame[margin_h:h-margin_h, margin_w:w-margin_w]
    return frame
```

> **Note:** If you resize frames, the video dimensions will differ from what's auto-detected. Ensure your info.json dimensions match the post-hook frame size.

#### `extra_parquet_columns(self, h5f, num_frames, mapping) -> dict[str, list]`

Add custom columns to the Parquet output. Return a dict mapping column names to lists of length `num_frames`.

**Parameters:**
- `h5f` — open `h5py.File` object
- `num_frames` — number of frames in this episode
- `mapping` — the full mapping config dict

**Example: Add reward column**

```python
def extra_parquet_columns(self, h5f, num_frames, mapping):
    rewards = h5f["/labels/rewards"][:num_frames].tolist()
    return {"reward": rewards}
```

## Output Format

The converter produces a LeRobot v2.1 dataset with this directory structure:

```
output_dir/
├── meta/
│   ├── info.json                          # Dataset metadata and feature definitions
│   ├── tasks.jsonl                        # Task index → task string mapping
│   └── episodes.jsonl                     # Per-episode metadata (tasks, length)
├── data/
│   └── chunk-000/                         # Episodes 0-999 (chunk size configurable)
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        ├── observation.images.head/
        │   ├── episode_000000.mp4
        │   ├── episode_000001.mp4
        │   └── ...
        ├── observation.images.left/
        │   └── ...
        └── observation.images.right/
            └── ...
```

### meta/info.json

Contains dataset-wide metadata and feature definitions:

```json
{
  "codebase_version": "v2.1",
  "robot_type": "bimanual_custom",
  "total_episodes": 500,
  "total_frames": 125000,
  "total_tasks": 3,
  "total_videos": 1500,
  "total_chunks": 1,
  "chunks_size": 1000,
  "fps": 30,
  "splits": {"train": "0:500"},
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
  "features": {
    "observation.images.head": {
      "dtype": "video",
      "shape": [480, 640, 3],
      "video_info": {"video.fps": 30.0, "video.codec": "x264", ...}
    },
    "observation.state": {
      "dtype": "float32",
      "shape": [24],
      "names": {"motors": ["left_joint_0", "left_joint_1", ...]}
    },
    "action": {"dtype": "float32", "shape": [24], "names": {"motors": [...]}},
    "timestamp": {"dtype": "float32", "shape": [1]},
    "frame_index": {"dtype": "int64", "shape": [1]},
    "episode_index": {"dtype": "int64", "shape": [1]},
    "index": {"dtype": "int64", "shape": [1]},
    "task_index": {"dtype": "int64", "shape": [1]},
    "language_instruction": {"dtype": "string", "shape": [1]}
  }
}
```

### Parquet columns

Each episode Parquet file contains one row per frame with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | float32 | Frame time in seconds (`frame_index / fps`) |
| `frame_index` | int64 | Frame number within episode (0-indexed) |
| `episode_index` | int64 | Episode number (0-indexed) |
| `index` | int64 | Global frame index across all episodes |
| `task_index` | int64 | Index into `tasks.jsonl` |
| `observation.state` | list[float32] | Concatenated observation vector |
| `action` | list[float32] | Concatenated action vector |
| `observation.images.<key>` | list[dict] | Video frame reference (`{"path": ..., "timestamp": ...}`) |
| `language_instruction` | string | Per-frame task description (if `per_frame: true`) |
| Extra observations | list[float32] | Additional features from `extra_observations` |

### meta/tasks.jsonl

One JSON object per line mapping task indices to task strings:

```json
{"task_index": 0, "task": "pick up the object"}
{"task_index": 1, "task": "place the object"}
```

### meta/episodes.jsonl

One JSON object per line with episode metadata:

```json
{"episode_index": 0, "tasks": ["pick up the object", "place the object"], "length": 553}
{"episode_index": 1, "tasks": ["pick up the object"], "length": 412}
```

## Validation Tools

### validate_schema.py — HDF5 Consistency Validation

Generates a schema from a reference HDF5 file and validates all other files against it.

```bash
# Basic usage (first file is reference)
python validate_schema.py /path/to/hdf5_files/

# Specify reference file and output directory
python validate_schema.py /path/to/hdf5_files/ \
    --reference best_episode.hdf5 \
    --output-dir schemas/my_robot/
```

**Outputs:**
- `schema.json` — HDF5 structure extracted from the reference file
- `validation_report.md` — Markdown report with:
  - Summary (total files, passed, failed)
  - Full schema table (all datasets with dtypes and shapes)
  - Missing datasets — datasets in schema but absent from files
  - Extra datasets — datasets in files but not in schema
  - Dtype mismatches — datasets with wrong data types
  - Ndims mismatches — datasets with wrong number of dimensions

### validate_lerobot_dataset.py — Output Dataset Validation

Validates the structure and consistency of a converted LeRobot dataset.

```bash
python validate_lerobot_dataset.py /path/to/output/
```

**Checks performed:**
- `meta/info.json` exists with required keys: `codebase_version`, `robot_type`, `total_episodes`, `total_frames`, `total_tasks`, `fps`, `features`, `data_path`, `video_path`
- `meta/tasks.jsonl` exists and task count matches `info.json`
- `meta/episodes.jsonl` exists, episode count matches, and sum of episode lengths equals `total_frames`
- Spot-checks the first 5 episodes:
  - Parquet file exists with expected columns
  - Frame indices are sequential (0 to N-1)
  - Video files exist for every camera
- Reports whether `meta/stats.json` and `meta/episodes_stats.jsonl` exist (these are computed by LeRobot on first load, not by this converter)

**Exit code:** 0 if all checks pass, 1 if any errors are found.

## HuggingFace Hub Integration

Upload your converted dataset to HuggingFace Hub for sharing and remote access.

**Authentication** — either pass `--hub-token` or log in beforehand:

```bash
huggingface-cli login
```

**Upload during conversion:**

```bash
hdf5-lerobot convert \
    --schema-name my_robot \
    --hdf5-dir /path/to/files/ \
    --output-dir /path/to/output/ \
    --push-to-hub \
    --repo-id my_org/my_dataset \
    --private
```

The upload creates the dataset repository (if it doesn't exist) and uploads the entire output directory. The `--private` flag makes the repository private on HuggingFace Hub.

## Common Scenarios

### Single-arm robot with one camera

```yaml
dataset:
  repo_id: "my_org/single_arm_dataset"
  robot_type: "single_arm"
  codebase_version: "v2.1"
  chunks_size: 1000
  fps:
    source: "fixed"
    fixed_value: 15

observation_state:
  components:
    - hdf5_path: "/observation/joint_positions"
      names_prefix: "joint"

action:
  components:
    - hdf5_path: "/action/joint_positions"
      names_prefix: "joint"

images:
  compressed: false
  raw_height: 480
  raw_width: 640
  raw_channels: 3
  cameras:
    - key: "wrist"
      hdf5_path: "/observation/images/wrist_cam"
  video_codec: "libx264"
  video_pix_fmt: "yuv420p"
  video_crf: 23

language_instructions:
  source: "fixed"
  fixed:
    instruction: "reach the target position"
  per_frame: false
```

### Bimanual robot with multiple cameras (picknplace example)

See `schemas/picknplace/mapping.yaml` for the full working example. It demonstrates:
- Multiple observation/action component concatenation (left + right joints and hand joints)
- 3 compressed cameras (head, left, right)
- Subtask-based language instructions
- Error-based filtering

### Dataset with raw (uncompressed) images

```yaml
images:
  compressed: false
  raw_height: 480
  raw_width: 640
  raw_channels: 3
  cameras:
    - key: "front"
      hdf5_path: "/images/front_camera"
    - key: "side"
      hdf5_path: "/images/side_camera"
```

When `compressed: false`, each HDF5 image dataset row is treated as a flat array of `height * width * channels` uint8 values. The converter reshapes it to `[H, W, C]` before encoding to video.

### Dataset with no language instructions

Simply omit the `language_instructions` section from mapping.yaml, or set:

```yaml
language_instructions:
  source: "fixed"
  fixed:
    instruction: "perform task"
  per_frame: false
```

The converter always writes `task_index` to the Parquet output. With a fixed instruction and `per_frame: false`, every episode gets the same task index and no `language_instruction` column is written.

### Dataset with per-frame language instructions in HDF5

```yaml
language_instructions:
  source: "hdf5_dataset"
  hdf5_dataset:
    path: "/annotations/language"
  per_frame: true
```

The HDF5 dataset at `/annotations/language` should be a 1D array of strings with the same length as the number of frames. Each unique string becomes a task entry in `tasks.jsonl`.

### Adding custom derived columns via hooks

```python
# schemas/my_robot/hooks.py
import numpy as np

class Hooks:
    def extra_parquet_columns(self, h5f, num_frames, mapping):
        # Add reward signal
        rewards = h5f["/labels/rewards"][:num_frames].tolist()

        # Add a binary success flag per frame
        task_done = h5f["/labels/task_done"][:num_frames]
        success = [bool(task_done[i]) for i in range(num_frames)]

        return {
            "reward": rewards,
            "success": success,
        }
```

## Troubleshooting

### `ffmpeg: command not found`

FFmpeg is required for video encoding. Install it:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### `FileNotFoundError: No .hdf5 files found in ...`

The converter looks for `*.hdf5` files in the specified directory. Check:
- The path is correct
- Files have the `.hdf5` extension (not `.h5` or `.hdf`)
- Files are in the directory root (the converter checks flat first, then recursive)

### `ValueError: HDF5 path /some/path not found in schema`

The HDF5 path specified in mapping.yaml doesn't exist in schema.json. Either:
- Fix the path in mapping.yaml to match your actual HDF5 structure
- Regenerate schema.json from a reference file that contains the expected dataset

### `KeyError` when reading HDF5 datasets

A dataset path in your mapping.yaml doesn't exist in one or more HDF5 files. Use the filtering section to skip inconsistent files, or run `validate_schema.py` to identify which files are missing datasets.

### `RuntimeError: No valid episodes found after filtering!`

All episodes were filtered out. Check your filtering config:
- Is `skip_on_error: true` and all episodes have `data_collection_error=True`?
- Is `min_frames` set too high?
- Is your hook's `filter_episode()` returning `False` for everything?

Set `--log-level DEBUG` to see why each episode is being skipped.

### `RuntimeError: No Hugging Face token found`

Either pass `--hub-token hf_xxxxx` on the command line, or log in first:

```bash
huggingface-cli login
```

### Video encoding is slow

Video encoding speed depends on codec and resolution. Tips:
- Use `video_codec: "libx264"` (fastest of the three supported codecs)
- Increase `video_crf` (e.g., 28) for faster encoding at lower quality
- Ensure FFmpeg is compiled with hardware acceleration for your platform

### Parquet files are very large

Large observation vectors or many extra observation columns increase file size. Consider:
- Using `slice` in observation/action components to drop unneeded columns
- Reducing the number of `extra_observations`
- Increasing `chunks_size` doesn't affect file size per episode but changes directory layout

## License

See the repository for license information.
