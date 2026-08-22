"""
Helper to import the *external* `samgeo` Python library from within the QGIS plugin.

Why this exists:
- When the plugin folder is named "samgeo" (as required by the official QGIS plugin repo),
  importing `samgeo` from plugin code resolves to the plugin package itself, shadowing the
  external `samgeo` library.

This module provides `get_samgeo()` which returns the external library module even when
shadowed by the plugin package name.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Optional

import importlib
import importlib.metadata
import importlib.util
import sys

_CACHED: Optional[ModuleType] = None


def _ensure_venv_on_path() -> bool:
    """Attempt to add the managed venv site-packages to sys.path.

    Returns:
        True if venv site-packages was added or already present.
    """
    try:
        from .core.venv_manager import ensure_venv_packages_available, get_venv_status

        is_ready, _ = get_venv_status()
        if is_ready:
            return ensure_venv_packages_available()
    except ImportError:
        try:
            from core.venv_manager import (
                ensure_venv_packages_available,
                get_venv_status,
            )

            is_ready, _ = get_venv_status()
            if is_ready:
                return ensure_venv_packages_available()
        except ImportError:
            pass
    return False


def _is_module_from_dir(mod: ModuleType, directory: Path) -> bool:
    try:
        mod_file = getattr(mod, "__file__", None)
        if not mod_file:
            return False
        return Path(mod_file).resolve().is_relative_to(directory.resolve())
    except Exception:
        return False


def _import_samgeo_without_plugin_shadow(plugin_pkg_dir: Path) -> Optional[ModuleType]:
    """
    Try to import `samgeo` while temporarily removing the plugin path from sys.path.
    """
    plugin_parent = plugin_pkg_dir.parent
    orig_sys_path = list(sys.path)
    orig_samgeo_mod = sys.modules.get("samgeo")

    try:
        sys.path = [p for p in sys.path if Path(p).resolve() != plugin_parent.resolve()]
        if "samgeo" in sys.modules:
            del sys.modules["samgeo"]

        imported = importlib.import_module("samgeo")
        if _is_module_from_dir(imported, plugin_pkg_dir):
            return None
        return imported
    except Exception:
        return None
    finally:
        sys.path = orig_sys_path
        if orig_samgeo_mod is not None:
            sys.modules["samgeo"] = orig_samgeo_mod


def _load_external_samgeo_from_dist(dist_name: str) -> Optional[ModuleType]:
    """
    Load the external samgeo package from an installed distribution.

    We load it under an alias module name (samgeo_external) to avoid conflicting with the
    plugin package (which may also be named `samgeo`).
    """
    try:
        dist = importlib.metadata.distribution(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None

    files = list(dist.files or [])
    init_rel = None
    for f in files:
        if str(f).replace("\\", "/").endswith("samgeo/__init__.py"):
            init_rel = f
            break
    if init_rel is None:
        return None

    init_path = Path(dist.locate_file(init_rel)).resolve()
    pkg_dir = init_path.parent

    alias_name = "samgeo_external"
    existing = sys.modules.get(alias_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(
        alias_name,
        str(init_path),
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[alias_name] = module
    spec.loader.exec_module(module)
    return module


def get_samgeo() -> ModuleType:
    """
    Return the external `samgeo` library module.
    """
    global _CACHED
    if _CACHED is not None:
        return _CACHED

    # Try to ensure managed venv packages are available
    _ensure_venv_on_path()

    plugin_dir = Path(__file__).resolve().parent

    # Normal import first (works when plugin package isn't named `samgeo`)
    try:
        imported = importlib.import_module("samgeo")
        if not _is_module_from_dir(imported, plugin_dir):
            _CACHED = imported
            return imported
    except Exception:  # nosec B110
        pass

    imported = _import_samgeo_without_plugin_shadow(plugin_dir)
    if imported is not None:
        _CACHED = imported
        return imported

    # Fallback: load from installed distribution metadata
    for dist_name in ("samgeo", "samgeo-py"):
        ext = _load_external_samgeo_from_dist(dist_name)
        if ext is not None:
            _CACHED = ext
            return ext

    py = getattr(sys, "executable", "python")
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    raise ImportError(
        "SamGeo plugin could not import the external 'samgeo' library.\n\n"
        "Please use the 'Install Dependencies' button in the SamGeo panel\n"
        "to install all required packages automatically.\n\n"
        f"QGIS Python:\n  executable: {py}\n  version: {ver}\n\n"
        "Alternatively, install manually in QGIS Python Console:\n"
        "  import sys, subprocess\n"
        "  subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'samgeo'])\n"
    )
