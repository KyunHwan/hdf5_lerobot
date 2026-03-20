"""Filesystem-based schema bundle discovery and loading.

A schema bundle is a directory under ``schemas/`` containing at least a
``mapping.yaml``.  It may also contain ``schema.json`` and ``hooks.py``.
"""

import importlib.util
import logging
import sys
from pathlib import Path

from .core.io_utils import load_json, load_yaml
from .hooks import DefaultHooks

logger = logging.getLogger(__name__)

# Default schemas directory: <repo_root>/schemas/
_DEFAULT_SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"


def list_schemas(schemas_dir: Path | None = None) -> list[str]:
    """Return names of all available schema bundles."""
    d = schemas_dir or _DEFAULT_SCHEMAS_DIR
    if not d.is_dir():
        return []
    return sorted(
        p.name for p in d.iterdir()
        if p.is_dir() and (p / "mapping.yaml").exists()
    )


def load_schema_bundle(
    name: str,
    schemas_dir: Path | None = None,
) -> tuple[dict | None, dict, object]:
    """Load a schema bundle by name.

    Parameters
    ----------
    name : str
        Name of the schema directory (e.g. ``"picknplace"``).
    schemas_dir : Path, optional
        Custom schemas root directory.

    Returns
    -------
    schema : dict or None
        Parsed ``schema.json``, or ``None`` if not present.
    mapping : dict
        Parsed ``mapping.yaml``.
    hooks : object
        Instance of the bundle's ``Hooks`` class, or ``DefaultHooks()``.
    """
    d = (schemas_dir or _DEFAULT_SCHEMAS_DIR) / name
    if not d.is_dir():
        available = list_schemas(schemas_dir)
        raise ValueError(
            f"Schema '{name}' not found in {d.parent}. "
            f"Available: {available or '(none)'}"
        )

    mapping_path = d / "mapping.yaml"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing mapping.yaml in schema '{name}'")
    mapping = load_yaml(str(mapping_path))

    schema_path = d / "schema.json"
    schema = load_json(str(schema_path)) if schema_path.exists() else None

    hooks_path = d / "hooks.py"
    if hooks_path.exists():
        hooks = _load_hooks_module(hooks_path, name)
    else:
        hooks = DefaultHooks()

    return schema, mapping, hooks


def load_hooks_from_file(hooks_path: str | Path) -> object:
    """Load a standalone hooks.py file (not part of a schema bundle).

    Parameters
    ----------
    hooks_path : str or Path
        Path to a Python file defining a ``Hooks`` class.

    Returns
    -------
    hooks : object
        Instance of the file's ``Hooks`` class, or ``DefaultHooks()``.
    """
    path = Path(hooks_path)
    if not path.exists():
        raise FileNotFoundError(f"Hooks file not found: {hooks_path}")
    return _load_hooks_module(path, path.stem)


def _load_hooks_module(hooks_path: Path, schema_name: str) -> object:
    """Dynamically import a hooks.py and instantiate its Hooks class."""
    module_name = f"hdf5_lerobot_schema_{schema_name}_hooks"
    spec = importlib.util.spec_from_file_location(module_name, hooks_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    hooks_cls = getattr(mod, "Hooks", None)
    if hooks_cls is None:
        logger.warning(
            f"No 'Hooks' class found in {hooks_path}, using DefaultHooks"
        )
        return DefaultHooks()
    return hooks_cls()
