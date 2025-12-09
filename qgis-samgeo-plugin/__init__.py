"""
SamGeo QGIS Plugin
A QGIS plugin for remote sensing image segmentation using SAM (Segment Anything Model).
"""


def classFactory(iface):
    """Load the SamGeoPlugin class.

    Args:
        iface: A QGIS interface instance.

    Returns:
        SamGeoPlugin: The plugin instance.
    """
    from .samgeo_plugin import SamGeoPlugin

    return SamGeoPlugin(iface)
