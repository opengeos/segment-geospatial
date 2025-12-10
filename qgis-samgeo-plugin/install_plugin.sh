#!/bin/bash
# Install SamGeo QGIS Plugin
#
# This script creates a symbolic link to the plugin in the QGIS plugins directory.
# Run this script to install the plugin for development/testing.

# Determine QGIS plugins directory
if [ -d "$HOME/.local/share/QGIS/QGIS3/profiles/default/python/plugins" ]; then
    QGIS_PLUGINS_DIR="$HOME/.local/share/QGIS/QGIS3/profiles/default/python/plugins"
elif [ -d "$HOME/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins" ]; then
    QGIS_PLUGINS_DIR="$HOME/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins"
elif [ -d "$APPDATA/QGIS/QGIS3/profiles/default/python/plugins" ]; then
    QGIS_PLUGINS_DIR="$APPDATA/QGIS/QGIS3/profiles/default/python/plugins"
else
    echo "Could not find QGIS plugins directory."
    echo "Please set QGIS_PLUGINS_DIR environment variable."
    exit 1
fi

# Get script directory (the plugin directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$SCRIPT_DIR"

# Create plugins directory if it doesn't exist
mkdir -p "$QGIS_PLUGINS_DIR"

# Remove existing link/directory (use samgeo_plugin to avoid conflict with samgeo package)
if [ -L "$QGIS_PLUGINS_DIR/samgeo_plugin" ] || [ -d "$QGIS_PLUGINS_DIR/samgeo_plugin" ]; then
    rm -rf "$QGIS_PLUGINS_DIR/samgeo_plugin"
    echo "Removed existing plugin installation."
fi

# Create symbolic link
ln -s "$PLUGIN_DIR" "$QGIS_PLUGINS_DIR/samgeo_plugin"

if [ $? -eq 0 ]; then
    echo "Plugin installed successfully!"
    echo "Plugin location: $QGIS_PLUGINS_DIR/samgeo_plugin"
    echo ""
    echo "To use the plugin:"
    echo "1. Restart QGIS"
    echo "2. Go to Plugins > Manage and Install Plugins"
    echo "3. Enable 'SamGeo'"
    echo ""
    echo "Note: Make sure you have samgeo installed in your QGIS Python environment:"
    echo "  pip install segment-geospatial[samgeo3]"
else
    echo "Failed to create symbolic link."
    exit 1
fi
