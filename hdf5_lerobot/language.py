"""Built-in language instruction extraction strategies."""

import re

import h5py


def extract_language_instructions(
    h5f: h5py.File,
    num_frames: int,
    lang_cfg: dict,
    episode_id: str,
    external_instructions: dict | None = None,
) -> list[str]:
    """Return a list of num_frames instruction strings for this episode."""
    source = lang_cfg.get("source", "fixed")

    if source == "fixed":
        text = lang_cfg["fixed"]["instruction"]
        return [text] * num_frames

    elif source == "hdf5_dataset":
        path = lang_cfg["hdf5_dataset"]["path"]
        data = h5f[path][:]
        # Handle bytes vs str
        return [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in data[:num_frames]]

    elif source == "hdf5_attribute":
        cfg = lang_cfg["hdf5_attribute"]
        group = h5f[cfg["group"]]
        text = group.attrs[cfg["attribute"]]
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        return [text] * num_frames

    elif source == "filename_regex":
        pattern = lang_cfg["filename_regex"]["pattern"]
        match = re.search(pattern, episode_id)
        text = match.group(1) if match else "unknown task"
        return [text] * num_frames

    elif source == "external_json":
        text = external_instructions.get(episode_id, "unknown task")
        return [text] * num_frames

    elif source == "subtask_status":
        cfg = lang_cfg["subtask_status"]
        task_group = cfg["task_group"]
        subtask_names = cfg["subtask_names"]  # {hdf5_name: instruction_text}
        default = cfg.get("default_instruction", "perform task")

        instructions = [default] * num_frames
        # Read each subtask status and assign instruction where active
        for subtask_hdf5_name, instruction_text in subtask_names.items():
            status_path = f"{task_group}/subtasks/{subtask_hdf5_name}/status"
            if status_path in h5f:
                statuses = h5f[status_path][:]
                for i in range(min(num_frames, len(statuses))):
                    s = statuses[i]
                    if isinstance(s, bytes):
                        s = s.decode("utf-8").strip()
                    # "active" or "in_progress" or "running" → assign this subtask's instruction
                    if s.lower() in ("active", "in_progress", "running", "started"):
                        instructions[i] = instruction_text
        return instructions

    else:
        raise ValueError(f"Unknown language instruction source: {source}")
