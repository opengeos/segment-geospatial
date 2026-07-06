"""Convert Jupyter notebooks in the docs folder to Markdown for Zensical.

Zensical does not yet support the mkdocs-jupyter plugin, so this script
converts every notebook under ``docs/examples`` and ``docs/workshops`` to a
Markdown file of the same name before the site is built. A download link to
the original notebook (which Zensical copies into the site as a static file)
is prepended to each page.

Usage:
    python scripts/convert_notebooks.py
"""

import pathlib
import sys

import nbformat
from nbconvert import MarkdownExporter

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOK_DIRS = ["docs/examples", "docs/workshops"]
REPO_URL = "https://github.com/opengeos/segment-geospatial"


def convert_notebook(nb_path: pathlib.Path) -> pathlib.Path:
    """Convert a single notebook to a Markdown file alongside it.

    Args:
        nb_path: Path to the ``.ipynb`` file to convert.

    Returns:
        pathlib.Path: Path to the generated ``.md`` file.
    """
    notebook = nbformat.read(nb_path, as_version=4)
    exporter = MarkdownExporter()
    body, _ = exporter.from_notebook_node(notebook)

    rel_path = nb_path.relative_to(ROOT)
    header = (
        f"[![Download notebook]"
        f"(https://img.shields.io/badge/Download-notebook-blue)]"
        f"({REPO_URL}/blob/main/{rel_path.as_posix()})\n\n"
    )

    md_path = nb_path.with_suffix(".md")
    md_path.write_text(header + body, encoding="utf-8")
    return md_path


def main() -> None:
    """Convert all notebooks in the configured docs directories.

    Conversion continues past individual failures so that one broken
    notebook does not hide the status of the others; the script exits
    with a non-zero status if any notebook failed to convert.
    """
    failures = []
    for dir_name in NOTEBOOK_DIRS:
        for nb_path in sorted((ROOT / dir_name).glob("*.ipynb")):
            try:
                md_path = convert_notebook(nb_path)
            except Exception as e:
                failures.append(nb_path)
                print(f"FAILED to convert {nb_path.relative_to(ROOT)}: {e}")
            else:
                print(f"Converted {nb_path.relative_to(ROOT)} -> {md_path.name}")

    if failures:
        print(f"{len(failures)} notebook(s) failed to convert.")
        sys.exit(1)


if __name__ == "__main__":
    main()
