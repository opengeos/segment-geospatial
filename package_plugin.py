#!/usr/bin/env python3
"""Package a QGIS plugin directory for upload to the official QGIS plugin repository."""

from __future__ import annotations

import argparse
import os
import re
import zipfile
from pathlib import Path

EXCLUDE_PATTERNS = [
    r"^ui_.*\.py$",
    r"^resources_rc\.py$",
    r"^.*_rc\.py$",
    r"^.*\.pyc$",
    r"^.*\.pyo$",
    r"^.*\.bak$",
    r"^.*~$",
    r"^\..*\.swp$",
    r"^.*\.orig$",
    r"^package_plugin\.py$",
]

EXCLUDE_DIRS = {
    "__pycache__",
    "__MACOSX",
    ".git",
    ".svn",
    ".hg",
    ".github",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".eggs",
    "build",
    "dist",
    "node_modules",
    "scripts",
    "help",
}


def should_exclude_file(filename: str) -> bool:
    return any(re.match(pattern, filename) for pattern in EXCLUDE_PATTERNS)


def should_exclude_dir(dirname: str) -> bool:
    return dirname.startswith(".") or dirname in EXCLUDE_DIRS or dirname.endswith(".egg-info")


def get_version_from_metadata(plugin_dir: Path) -> str:
    metadata_file = plugin_dir / "metadata.txt"
    if metadata_file.exists():
        with metadata_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("version="):
                    return line.split("=", 1)[1].strip()
    return "unknown"


def package_plugin(source_dir: Path, output_path: Path | None, target_name: str) -> Path:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not source_dir.is_dir():
        raise ValueError(f"Source path is not a directory: {source_dir}")

    version = get_version_from_metadata(source_dir)
    if output_path is None:
        output_path = source_dir.parent / f"{target_name}-{version}.zip"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    print(f"Packaging plugin from: {source_dir}")
    print(f"Output zip file: {output_path}")
    print(f"Root folder name in zip: {target_name}")
    print(f"Plugin version: {version}")

    files_added = 0
    files_excluded = 0
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            dirs[:] = [d for d in dirs if not should_exclude_dir(d)]
            for file in files:
                file_path = Path(root) / file
                if should_exclude_file(file) or file.startswith("."):
                    files_excluded += 1
                    continue
                rel_path = file_path.relative_to(source_dir)
                archive_name = Path(target_name) / rel_path
                zipf.write(file_path, archive_name)
                files_added += 1

    print(f"Package created successfully: {output_path}")
    print(f"Files added: {files_added}")
    print(f"Files excluded: {files_excluded}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output path for the zip file")
    parser.add_argument("--source", "-s", type=Path, default=Path("."), help="Source plugin directory")
    parser.add_argument("--name", "-n", default=None, help="Target plugin folder name in the zip")
    args = parser.parse_args()

    source_dir = args.source.resolve()
    target_name = args.name or source_dir.name
    package_plugin(source_dir, args.output, target_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
