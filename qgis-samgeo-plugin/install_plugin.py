#!/usr/bin/env python3
"""
Install SamGeo QGIS Plugin.

This script creates a symbolic link to the plugin in the QGIS plugins directory.
Run this script to install the plugin for development/testing.

Usage:
    python install_plugin.py
    python install_plugin.py --uninstall
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def get_qgis_plugins_dir():
    """Get the QGIS plugins directory based on the operating system."""
    home = Path.home()

    # Try different locations based on OS
    possible_dirs = [
        # Linux
        home / ".local/share/QGIS/QGIS3/profiles/default/python/plugins",
        # macOS
        home / "Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins",
        # Windows (via APPDATA)
        Path(os.environ.get("APPDATA", ""))
        / "QGIS/QGIS3/profiles/default/python/plugins",
    ]

    for plugins_dir in possible_dirs:
        if plugins_dir.parent.exists():
            return plugins_dir

    return None


def install_plugin(plugins_dir, plugin_dir):
    """Install the plugin by creating a symbolic link."""
    # The plugin folder name in QGIS - use 'samgeo_plugin' to avoid conflict with samgeo package
    target = plugins_dir / "samgeo_plugin"

    # Create plugins directory if it doesn't exist
    plugins_dir.mkdir(parents=True, exist_ok=True)

    # Remove existing installation
    if target.exists() or target.is_symlink():
        if target.is_symlink():
            target.unlink()
        else:
            shutil.rmtree(target)
        print(f"Removed existing installation at {target}")

    # Create symbolic link
    try:
        target.symlink_to(plugin_dir)
        print(f"Plugin installed successfully!")
        print(f"Plugin location: {target}")
        print(f"Linked to: {plugin_dir}")
    except OSError as e:
        # On Windows, symlinks may require admin privileges
        # Fall back to copying
        print(f"Could not create symlink: {e}")
        print("Copying plugin instead...")
        try:
            shutil.copytree(plugin_dir, target)
            print(f"Plugin copied to: {target}")
        except Exception as copy_exc:
            print(f"Failed to copy plugin: {copy_exc}")
            # Clean up partially copied directory
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            print("Cleaned up partially copied plugin directory due to error.")

    print()
    print("To use the plugin:")
    print("1. Restart QGIS")
    print("2. Go to Plugins > Manage and Install Plugins")
    print("3. Enable 'SamGeo' in the plugin manager")
    print()
    print("Note: Make sure you have samgeo installed in your QGIS Python environment:")
    print("  pip install segment-geospatial[samgeo3]")


def uninstall_plugin(plugins_dir):
    """Uninstall the plugin by removing the symbolic link or directory."""
    target = plugins_dir / "samgeo_plugin"

    if target.exists() or target.is_symlink():
        if target.is_symlink():
            target.unlink()
        else:
            shutil.rmtree(target)
        print(f"Plugin uninstalled successfully!")
        print(f"Removed: {target}")
    else:
        print("Plugin is not installed.")


def main():
    parser = argparse.ArgumentParser(description="Install/Uninstall SamGeo QGIS Plugin")
    parser.add_argument("--uninstall", action="store_true", help="Uninstall the plugin")
    parser.add_argument("--plugins-dir", type=str, help="Custom QGIS plugins directory")
    args = parser.parse_args()

    # Get plugin directory (where this script is located)
    script_dir = Path(__file__).parent.absolute()
    plugin_dir = script_dir  # The plugin IS this directory

    # Verify essential files exist
    required_files = ["__init__.py", "metadata.txt", "samgeo_plugin.py"]
    for f in required_files:
        if not (plugin_dir / f).exists():
            print(f"Error: Required file not found: {f}")
            sys.exit(1)

    # Get QGIS plugins directory
    if args.plugins_dir:
        plugins_dir = Path(args.plugins_dir)
    else:
        plugins_dir = get_qgis_plugins_dir()

    if plugins_dir is None:
        print("Error: Could not find QGIS plugins directory.")
        print("Please specify the directory with --plugins-dir")
        sys.exit(1)

    print(f"QGIS plugins directory: {plugins_dir}")

    if args.uninstall:
        uninstall_plugin(plugins_dir)
    else:
        install_plugin(plugins_dir, plugin_dir)


if __name__ == "__main__":
    main()
