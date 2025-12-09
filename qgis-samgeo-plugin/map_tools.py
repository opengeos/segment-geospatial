"""
Map tools for interactive segmentation prompts.
"""

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.core import QgsPointXY, QgsRectangle, QgsWkbTypes
from qgis.gui import QgsMapTool, QgsRubberBand, QgsVertexMarker


class PointPromptTool(QgsMapTool):
    """Map tool for adding point prompts."""

    point_added = pyqtSignal(object, bool)  # point, is_foreground

    def __init__(self, canvas, plugin, batch_mode=False):
        """Initialize the point prompt tool.

        Args:
            canvas: The QGIS map canvas.
            plugin: The parent plugin instance.
            batch_mode: If True, adds points to batch list instead of regular list.
        """
        super().__init__(canvas)
        self.canvas = canvas
        self.plugin = plugin
        self.is_foreground = True
        self.batch_mode = batch_mode
        self.markers = []

        # Set cursor
        self.setCursor(Qt.CrossCursor)

    def set_foreground(self, foreground):
        """Set whether we're adding foreground or background points."""
        self.is_foreground = foreground

    def canvasPressEvent(self, event):
        """Handle mouse press event."""
        pass

    def canvasReleaseEvent(self, event):
        """Handle mouse release event - add a point."""
        if event.button() == Qt.LeftButton:
            point = self.toMapCoordinates(event.pos())

            # Add visual marker
            marker = QgsVertexMarker(self.canvas)
            marker.setCenter(point)
            # Use blue for batch mode, green/red for regular mode
            if self.batch_mode:
                marker.setColor(QColor(0, 120, 255))
            else:
                marker.setColor(
                    QColor(0, 255, 0) if self.is_foreground else QColor(255, 0, 0)
                )
            marker.setIconType(QgsVertexMarker.ICON_CIRCLE)
            marker.setIconSize(10)
            marker.setPenWidth(3)
            self.markers.append(marker)

            # Add point to plugin
            if self.batch_mode:
                self.plugin.add_batch_point(point)
            else:
                self.plugin.add_point(point, self.is_foreground)

        elif event.button() == Qt.RightButton:
            # Right-click to finish
            self.deactivate()
            if self.batch_mode:
                self.plugin.batch_add_point_btn.setChecked(False)
            else:
                self.plugin.add_fg_point_btn.setChecked(False)
                self.plugin.add_bg_point_btn.setChecked(False)

    def clear_markers(self):
        """Clear all markers from the canvas."""
        for marker in self.markers:
            self.canvas.scene().removeItem(marker)
        self.markers = []

    def deactivate(self):
        """Deactivate the tool."""
        super().deactivate()


class BoxPromptTool(QgsMapTool):
    """Map tool for drawing box prompts."""

    box_drawn = pyqtSignal(object)  # QgsRectangle

    def __init__(self, canvas, plugin):
        """Initialize the box prompt tool.

        Args:
            canvas: The QGIS map canvas.
            plugin: The parent plugin instance.
        """
        super().__init__(canvas)
        self.canvas = canvas
        self.plugin = plugin

        # Rubber band for visual feedback
        self.rubber_band = QgsRubberBand(canvas, QgsWkbTypes.PolygonGeometry)
        self.rubber_band.setColor(QColor(0, 120, 255, 100))
        self.rubber_band.setStrokeColor(QColor(0, 120, 255))
        self.rubber_band.setWidth(2)

        self.start_point = None
        self.is_drawing = False

        # Set cursor
        self.setCursor(Qt.CrossCursor)

    def canvasPressEvent(self, event):
        """Handle mouse press event - start drawing."""
        if event.button() == Qt.LeftButton:
            self.start_point = self.toMapCoordinates(event.pos())
            self.is_drawing = True
            self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)

    def canvasMoveEvent(self, event):
        """Handle mouse move event - update rubber band."""
        if self.is_drawing and self.start_point is not None:
            current_point = self.toMapCoordinates(event.pos())
            self.update_rubber_band(self.start_point, current_point)

    def canvasReleaseEvent(self, event):
        """Handle mouse release event - finish drawing."""
        if event.button() == Qt.LeftButton and self.is_drawing:
            end_point = self.toMapCoordinates(event.pos())
            self.is_drawing = False

            # Create rectangle
            rect = QgsRectangle(self.start_point, end_point)

            # Update rubber band with final position
            self.update_rubber_band(self.start_point, end_point)

            # Set box in plugin
            self.plugin.set_box(rect)

            # Deactivate tool
            self.deactivate()
            self.plugin.draw_box_btn.setChecked(False)

        elif event.button() == Qt.RightButton:
            # Right-click to cancel
            self.is_drawing = False
            self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)
            self.deactivate()
            self.plugin.draw_box_btn.setChecked(False)

    def update_rubber_band(self, start_point, end_point):
        """Update the rubber band rectangle."""
        self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)

        # Create rectangle points
        self.rubber_band.addPoint(QgsPointXY(start_point.x(), start_point.y()), False)
        self.rubber_band.addPoint(QgsPointXY(end_point.x(), start_point.y()), False)
        self.rubber_band.addPoint(QgsPointXY(end_point.x(), end_point.y()), False)
        self.rubber_band.addPoint(QgsPointXY(start_point.x(), end_point.y()), True)

    def clear_rubber_band(self):
        """Clear the rubber band."""
        self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)

    def deactivate(self):
        """Deactivate the tool."""
        self.is_drawing = False
        super().deactivate()
