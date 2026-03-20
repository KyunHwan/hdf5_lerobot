"""Hook protocol and default implementation for schema-specific conversion logic.

Schema bundles can provide a ``hooks.py`` file that defines a ``Hooks`` class.
The class can override any subset of the methods below.  Methods that are not
overridden fall back to the no-op defaults in ``DefaultHooks``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import h5py
import numpy as np


@runtime_checkable
class ConverterHooks(Protocol):
    """Optional hooks for schema-specific conversion logic.

    All methods have default no-op implementations in ``DefaultHooks``.
    A schema's ``hooks.py`` only needs to define the methods it wants to
    override.
    """

    def filter_episode(self, h5f: h5py.File, mapping: dict) -> bool:
        """Return True to KEEP the episode, False to skip it.

        Called after the HDF5 file is opened, before any data is read.
        The default implementation returns True (defer to YAML-based filtering).
        """
        ...

    def extract_language_instructions(
        self,
        h5f: h5py.File,
        num_frames: int,
        lang_cfg: dict,
        episode_id: str,
        external_instructions: dict | None,
    ) -> list[str] | None:
        """Return per-frame instruction strings, or None to use built-in logic.

        Returning ``None`` falls through to the standard strategy-based
        extraction in ``language.py``.  Return a list to fully override.
        """
        ...

    def transform_observation_state(
        self, obs: np.ndarray, h5f: h5py.File, mapping: dict,
    ) -> np.ndarray:
        """Post-process observation state array ``[T, obs_dim]`` before writing.

        Default: return *obs* unchanged.
        """
        ...

    def transform_action(
        self, action: np.ndarray, h5f: h5py.File, mapping: dict,
    ) -> np.ndarray:
        """Post-process action array ``[T, act_dim]`` before writing.

        Default: return *action* unchanged.
        """
        ...

    def transform_frame(
        self, frame: np.ndarray, cam_key: str,
    ) -> np.ndarray:
        """Post-process an image frame before video encoding.

        Default: return *frame* unchanged.  Useful for cropping, resizing,
        or color correction that varies by schema.
        """
        ...

    def extra_parquet_columns(
        self, h5f: h5py.File, num_frames: int, mapping: dict,
    ) -> dict[str, list]:
        """Return additional columns to include in the parquet file.

        Keys are column names, values are lists of length *num_frames*.
        Default: return empty dict.
        """
        ...


class DefaultHooks:
    """Default (no-op) implementation of all hooks."""

    def filter_episode(self, h5f: h5py.File, mapping: dict) -> bool:
        return True

    def extract_language_instructions(
        self,
        h5f: h5py.File,
        num_frames: int,
        lang_cfg: dict,
        episode_id: str,
        external_instructions: dict | None,
    ) -> list[str] | None:
        return None

    def transform_observation_state(
        self, obs: np.ndarray, h5f: h5py.File, mapping: dict,
    ) -> np.ndarray:
        return obs

    def transform_action(
        self, action: np.ndarray, h5f: h5py.File, mapping: dict,
    ) -> np.ndarray:
        return action

    def transform_frame(
        self, frame: np.ndarray, cam_key: str,
    ) -> np.ndarray:
        return frame

    def extra_parquet_columns(
        self, h5f: h5py.File, num_frames: int, mapping: dict,
    ) -> dict[str, list]:
        return {}
