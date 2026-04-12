"""Setup for cli-anything-samgeo."""

from setuptools import setup, find_namespace_packages

setup(
    name="cli-anything-samgeo",
    version="0.1.0",
    description="CLI harness for segment-geospatial (SAM for geospatial data)",
    author="Qiusheng Wu",
    author_email="giswqs@gmail.com",
    url="https://github.com/opengeos/segment-geospatial",
    packages=find_namespace_packages(include=["cli_anything.*"]),
    package_data={
        "cli_anything.samgeo": ["skills/*.md"],
    },
    install_requires=[
        "click>=8.0",
        "segment-geospatial",
    ],
    extras_require={
        "repl": ["prompt_toolkit>=3.0"],
        "dev": ["pytest>=7.0"],
    },
    entry_points={
        "console_scripts": [
            "cli-anything-samgeo=cli_anything.samgeo.samgeo_cli:main",
        ],
    },
    python_requires=">=3.10",
)
