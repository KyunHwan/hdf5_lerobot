"""Hugging Face Hub upload functionality."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def push_dataset_to_hub(
    output_dir: Path,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
) -> str:
    """Upload the converted LeRobot dataset to Hugging Face Hub.

    Returns the URL of the uploaded dataset.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    if api.token is None:
        raise RuntimeError(
            "No Hugging Face token found. Either pass --hub-token or run "
            "'huggingface-cli login' first."
        )

    logger.info(f"Creating/verifying dataset repo: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    logger.info(f"Uploading {output_dir} to {repo_id} ...")
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload LeRobot v2.1 dataset",
    )

    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info(f"Upload complete: {url}")
    return url
