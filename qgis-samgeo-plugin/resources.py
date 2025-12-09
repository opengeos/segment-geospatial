"""
Resource utilities for the SamGeo QGIS Plugin.
"""

import os

PLUGIN_DIR = os.path.dirname(__file__)


def get_icon_path(icon_name="icon.png"):
    """Get the path to an icon file.

    Args:
        icon_name: Name of the icon file.

    Returns:
        str: Full path to the icon file.
    """
    return os.path.join(PLUGIN_DIR, "icons", icon_name)
