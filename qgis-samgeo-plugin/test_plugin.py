#!/usr/bin/env python3
"""
Test script for SamGeo QGIS Plugin.

This script can be run standalone to test basic functionality,
or in the QGIS Python console to test the full plugin.

Usage:
    # Standalone test
    python test_plugin.py

    # In QGIS Python Console
    exec(open('/path/to/qgis-samgeo-plugin/test_plugin.py').read())
"""

import os
import sys

# Get the plugin directory
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))

# Add plugin directory to path
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)


def test_samgeo_import():
    """Test that SamGeo can be imported."""
    print("[Test] SamGeo import...")
    try:
        from samgeo import SamGeo3

        print("  [OK] SamGeo3 imported successfully")
        return True
    except ImportError as e:
        print(f"  [FAIL] {e}")
        return False


def test_plugin_modules():
    """Test that plugin modules can be imported."""
    print("[Test] Plugin module imports...")
    try:
        # When run standalone, we need absolute imports
        # When loaded by QGIS, relative imports work
        try:
            # Try as QGIS plugin (relative imports)
            from samgeo.samgeo_plugin import SamGeoPlugin

            print("  [OK] samgeo_plugin module (as QGIS plugin)")
        except ImportError:
            # Run standalone (absolute imports)
            import samgeo_plugin

            print("  [OK] samgeo_plugin module (standalone)")

        # Test map tools - use absolute import for standalone
        try:
            from samgeo.map_tools import PointPromptTool, BoxPromptTool

            print("  [OK] map_tools module (as QGIS plugin)")
        except ImportError:
            import map_tools

            print("  [OK] map_tools module (standalone)")

        # Test resources
        try:
            from samgeo.resources import get_icon_path

            print("  [OK] resources module (as QGIS plugin)")
        except ImportError:
            import resources

            print("  [OK] resources module (standalone)")

        return True
    except ImportError as e:
        print(f"  [FAIL] {e}")
        return False


def test_qgis_imports():
    """Test that QGIS modules can be imported."""
    print("[Test] QGIS module imports...")
    try:
        from qgis.core import QgsApplication, QgsRasterLayer, QgsVectorLayer
        from qgis.PyQt.QtWidgets import QApplication

        print("  [OK] QGIS core modules imported")
        return True
    except ImportError as e:
        print(f"  [FAIL] {e}")
        print("  Note: QGIS modules require running in a QGIS environment")
        return False


def test_plugin_in_qgis():
    """Test the plugin within QGIS environment."""
    print("[Test] Plugin in QGIS...")
    try:
        from qgis.utils import iface

        if iface is None:
            print("  [SKIP] Not running in QGIS GUI")
            return False

        # Import the plugin
        from samgeo_plugin import SamGeoPlugin

        # Create plugin instance
        plugin = SamGeoPlugin(iface)
        print("  [OK] Plugin instance created")

        # Initialize GUI
        plugin.initGui()
        print("  [OK] Plugin GUI initialized")

        # Show the dock widget
        plugin.run()
        print("  [OK] Plugin dock widget opened")

        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def show_programmatic_usage():
    """Show examples of programmatic usage."""
    print("\n" + "=" * 60)
    print("Programmatic Usage Examples")
    print("=" * 60)

    print(
        """
# Initialize the model
from samgeo import SamGeo3

sam = SamGeo3(
    backend="meta",
    enable_inst_interactivity=True,
    confidence_threshold=0.5,
)

# Set an image
sam.set_image("/path/to/image.tif")

# Text-based segmentation
sam.generate_masks("building")
sam.save_masks("buildings.tif", unique=True)

# Point-based segmentation
point_coords = [[500, 300], [600, 400]]
point_labels = [1, 1]  # 1=foreground, 0=background
sam.generate_masks_by_points(point_coords, point_labels)
sam.save_masks("points_result.tif")

# Box-based segmentation
boxes = [[100, 100, 500, 500]]  # [xmin, ymin, xmax, ymax]
sam.generate_masks_by_boxes(boxes)
sam.save_masks("box_result.tif")

# Convert raster to vector
from samgeo import common
common.raster_to_vector("buildings.tif", "buildings.gpkg")
"""
    )


def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("SamGeo QGIS Plugin - Test Suite")
    print("=" * 60)
    print(f"Plugin directory: {PLUGIN_DIR}\n")

    results = []

    # Run tests
    results.append(("SamGeo Import", test_samgeo_import()))
    results.append(("Plugin Modules", test_plugin_modules()))
    results.append(("QGIS Imports", test_qgis_imports()))

    # Try QGIS-specific test
    try:
        from qgis.utils import iface

        if iface is not None:
            results.append(("Plugin in QGIS", test_plugin_in_qgis()))
    except ImportError:
        # QGIS is not available; skip QGIS-specific test.
        pass

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    # Show usage examples
    show_programmatic_usage()

    return passed == total


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
