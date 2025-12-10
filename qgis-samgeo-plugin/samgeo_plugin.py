"""
SamGeo QGIS Plugin - Main Plugin Class
"""

import os

from qgis.PyQt.QtCore import Qt, QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import (
    QAction,
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QGroupBox,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QTabWidget,
    QTextEdit,
    QListWidget,
    QListWidgetItem,
)
from qgis.core import (
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsWkbTypes,
    Qgis,
    QgsMessageLog,
)

# Import the map tools
from .map_tools import PointPromptTool, BoxPromptTool


class SamGeoPlugin:
    """QGIS Plugin for remote sensing image segmentation using SamGeo."""

    def __init__(self, iface):
        """Initialize the plugin.

        Args:
            iface: A QGIS interface instance.
        """
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self.plugin_dir = os.path.dirname(__file__)

        # Initialize plugin attributes
        self.actions = []
        self.menu = "&SamGeo"
        self.toolbar = self.iface.addToolBar("SamGeo")
        self.toolbar.setObjectName("SamGeo")

        # Dock widget
        self.dock_widget = None

        # SamGeo model instance
        self.sam = None
        self.current_layer = None
        self.current_image_path = None

        # Point and box prompts
        self.point_coords = []
        self.point_labels = []
        self.box_coords = None

        # Batch point prompts
        self.batch_point_coords = []
        self.batch_point_coords_map = []  # Map coordinates for display

        # Map tools
        self.point_tool = None
        self.batch_point_tool = None
        self.box_tool = None
        self.previous_tool = None

    def tr(self, message):
        """Translate a message."""
        return QCoreApplication.translate("SamGeo", message)

    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None,
        checkable=False,
    ):
        """Add a toolbar icon to the toolbar."""
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)
        action.setCheckable(checkable)

        if status_tip is not None:
            action.setStatusTip(status_tip)
        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)
        if add_to_menu:
            self.iface.addPluginToRasterMenu(self.menu, action)

        self.actions.append(action)
        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""
        icon_path = os.path.join(self.plugin_dir, "icons", "icon.png")

        # Main action to open the plugin dock
        self.add_action(
            icon_path,
            text=self.tr("SamGeo Segmentation"),
            callback=self.run,
            parent=self.iface.mainWindow(),
            status_tip=self.tr("Open SamGeo Segmentation Panel"),
        )

    def unload(self):
        """Remove the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginRasterMenu(self.menu, action)
            self.iface.removeToolBarIcon(action)

        # Remove dock widget
        if self.dock_widget is not None:
            self.iface.removeDockWidget(self.dock_widget)

        # Remove toolbar
        del self.toolbar

        # Clean up model
        if self.sam is not None:
            del self.sam
            self.sam = None

    def run(self):
        """Run the plugin - show the dock widget."""
        if self.dock_widget is None:
            self.dock_widget = self.create_dock_widget()
            self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock_widget)
        else:
            self.dock_widget.show()

    def create_dock_widget(self):
        """Create the dock widget with all controls."""
        dock = QDockWidget("SamGeo Segmentation", self.iface.mainWindow())
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # Main widget
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # Tab widget for different modes
        tab_widget = QTabWidget()

        # === Model Settings Tab ===
        model_tab = QWidget()
        model_layout = QVBoxLayout()
        model_tab.setLayout(model_layout)

        # Backend selection
        backend_group = QGroupBox("Model Settings")
        backend_layout = QVBoxLayout()

        # Model version selection
        version_row = QHBoxLayout()
        version_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["SamGeo3 (SAM3)", "SamGeo2 (SAM2)", "SamGeo (SAM1)"])
        version_row.addWidget(self.model_combo)
        backend_layout.addLayout(version_row)

        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Backend:"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["meta", "transformers"])
        backend_row.addWidget(self.backend_combo)
        backend_layout.addLayout(backend_row)

        # Device selection
        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu"])
        device_row.addWidget(self.device_combo)
        backend_layout.addLayout(device_row)

        # Confidence threshold
        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Confidence:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setValue(0.5)
        self.conf_spin.setSingleStep(0.05)
        conf_row.addWidget(self.conf_spin)
        backend_layout.addLayout(conf_row)

        # Interactive mode checkbox
        self.interactive_check = QCheckBox(
            "Enable Interactive Mode (Point/Box Prompts)"
        )
        self.interactive_check.setChecked(True)
        backend_layout.addWidget(self.interactive_check)

        # Load model button
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        backend_layout.addWidget(self.load_model_btn)

        # Model status
        self.model_status = QLabel("Model: Not loaded")
        self.model_status.setStyleSheet("color: gray;")
        backend_layout.addWidget(self.model_status)

        backend_group.setLayout(backend_layout)
        model_layout.addWidget(backend_group)

        # Layer selection
        layer_group = QGroupBox("Input Layer")
        layer_layout = QVBoxLayout()

        layer_row = QHBoxLayout()
        self.layer_combo = QComboBox()
        self.refresh_layers()
        layer_row.addWidget(self.layer_combo)

        refresh_btn = QPushButton("↻")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.clicked.connect(self.refresh_layers)
        layer_row.addWidget(refresh_btn)
        layer_layout.addLayout(layer_row)

        self.set_layer_btn = QPushButton("Set Image from Layer")
        self.set_layer_btn.clicked.connect(self.set_image_from_layer)
        layer_layout.addWidget(self.set_layer_btn)

        # Or load from file
        file_row = QHBoxLayout()
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setPlaceholderText("Or select image file...")
        file_row.addWidget(self.image_path_edit)

        browse_btn = QPushButton("...")
        browse_btn.setMaximumWidth(30)
        browse_btn.clicked.connect(self.browse_image)
        file_row.addWidget(browse_btn)
        layer_layout.addLayout(file_row)

        self.set_file_btn = QPushButton("Set Image from File")
        self.set_file_btn.clicked.connect(self.set_image_from_file)
        layer_layout.addWidget(self.set_file_btn)

        self.image_status = QLabel("Image: Not set")
        self.image_status.setStyleSheet("color: gray;")
        layer_layout.addWidget(self.image_status)

        layer_group.setLayout(layer_layout)
        model_layout.addWidget(layer_group)

        model_layout.addStretch()
        tab_widget.addTab(model_tab, "Model")

        # === Text Prompts Tab ===
        text_tab = QWidget()
        text_layout = QVBoxLayout()
        text_tab.setLayout(text_layout)

        text_group = QGroupBox("Text-Based Segmentation")
        text_group_layout = QVBoxLayout()

        text_group_layout.addWidget(QLabel("Describe objects to segment:"))
        self.text_prompt_edit = QLineEdit()
        self.text_prompt_edit.setPlaceholderText("e.g., tree, building, road...")
        text_group_layout.addWidget(self.text_prompt_edit)

        # Size filters
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Min size:"))
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(0, 1000000)
        self.min_size_spin.setValue(0)
        size_row.addWidget(self.min_size_spin)

        size_row.addWidget(QLabel("Max size:"))
        self.max_size_spin = QSpinBox()
        self.max_size_spin.setRange(0, 10000000)
        self.max_size_spin.setValue(0)
        self.max_size_spin.setSpecialValueText("No limit")
        size_row.addWidget(self.max_size_spin)
        text_group_layout.addLayout(size_row)

        self.text_segment_btn = QPushButton("Segment by Text")
        self.text_segment_btn.clicked.connect(self.segment_by_text)
        text_group_layout.addWidget(self.text_segment_btn)

        # Text segmentation status
        self.text_status_label = QLabel("")
        text_group_layout.addWidget(self.text_status_label)

        text_group.setLayout(text_group_layout)
        text_layout.addWidget(text_group)

        text_layout.addStretch()
        tab_widget.addTab(text_tab, "Text")

        # === Interactive Tab ===
        interactive_tab = QWidget()
        interactive_layout = QVBoxLayout()
        interactive_tab.setLayout(interactive_layout)

        # Point prompts
        point_group = QGroupBox("Point Prompts")
        point_layout = QVBoxLayout()

        point_btn_row = QHBoxLayout()
        self.add_fg_point_btn = QPushButton("Add Foreground Points")
        self.add_fg_point_btn.setCheckable(True)
        self.add_fg_point_btn.clicked.connect(
            lambda: self.start_point_tool(foreground=True)
        )
        point_btn_row.addWidget(self.add_fg_point_btn)

        self.add_bg_point_btn = QPushButton("Add Background Points")
        self.add_bg_point_btn.setCheckable(True)
        self.add_bg_point_btn.clicked.connect(
            lambda: self.start_point_tool(foreground=False)
        )
        point_btn_row.addWidget(self.add_bg_point_btn)
        point_layout.addLayout(point_btn_row)

        self.points_list = QListWidget()
        self.points_list.setMaximumHeight(100)
        point_layout.addWidget(self.points_list)

        point_action_row = QHBoxLayout()
        clear_points_btn = QPushButton("Clear Points")
        clear_points_btn.clicked.connect(self.clear_points)
        point_action_row.addWidget(clear_points_btn)

        self.point_segment_btn = QPushButton("Segment by Points")
        self.point_segment_btn.clicked.connect(self.segment_by_points)
        point_action_row.addWidget(self.point_segment_btn)
        point_layout.addLayout(point_action_row)

        # Point segmentation status
        self.point_status_label = QLabel("")
        point_layout.addWidget(self.point_status_label)

        point_group.setLayout(point_layout)
        interactive_layout.addWidget(point_group)

        # Box prompts
        box_group = QGroupBox("Box Prompts")
        box_layout = QVBoxLayout()

        self.draw_box_btn = QPushButton("Draw Box")
        self.draw_box_btn.setCheckable(True)
        self.draw_box_btn.clicked.connect(self.start_box_tool)
        box_layout.addWidget(self.draw_box_btn)

        self.box_label = QLabel("Box: Not set")
        box_layout.addWidget(self.box_label)

        box_action_row = QHBoxLayout()
        clear_box_btn = QPushButton("Clear Box")
        clear_box_btn.clicked.connect(self.clear_box)
        box_action_row.addWidget(clear_box_btn)

        self.box_segment_btn = QPushButton("Segment by Box")
        self.box_segment_btn.clicked.connect(self.segment_by_box)
        box_action_row.addWidget(self.box_segment_btn)
        box_layout.addLayout(box_action_row)

        # Box segmentation status
        self.box_status_label = QLabel("")
        box_layout.addWidget(self.box_status_label)

        box_group.setLayout(box_layout)
        interactive_layout.addWidget(box_group)

        interactive_layout.addStretch()
        tab_widget.addTab(interactive_tab, "Interactive")

        # === Batch Tab ===
        batch_tab = QWidget()
        batch_layout = QVBoxLayout()
        batch_tab.setLayout(batch_layout)

        # Interactive Points for Batch
        batch_interactive_group = QGroupBox("Create Points Interactively")
        batch_interactive_layout = QVBoxLayout()

        batch_point_btn_row = QHBoxLayout()
        self.batch_add_point_btn = QPushButton("Add Points on Map")
        self.batch_add_point_btn.setCheckable(True)
        self.batch_add_point_btn.clicked.connect(self.start_batch_point_tool)
        batch_point_btn_row.addWidget(self.batch_add_point_btn)

        self.batch_clear_points_btn = QPushButton("Clear Points")
        self.batch_clear_points_btn.clicked.connect(self.clear_batch_points)
        batch_point_btn_row.addWidget(self.batch_clear_points_btn)
        batch_interactive_layout.addLayout(batch_point_btn_row)

        # Batch points list
        self.batch_points_list = QListWidget()
        self.batch_points_list.setMaximumHeight(80)
        batch_interactive_layout.addWidget(self.batch_points_list)

        self.batch_points_count_label = QLabel("Points: 0")
        batch_interactive_layout.addWidget(self.batch_points_count_label)

        batch_interactive_group.setLayout(batch_interactive_layout)
        batch_layout.addWidget(batch_interactive_group)

        # Or Load from File/Layer
        batch_file_group = QGroupBox("Or Load Points from File/Layer")
        batch_file_layout = QVBoxLayout()

        # Vector layer selection
        vector_layer_row = QHBoxLayout()
        vector_layer_row.addWidget(QLabel("Layer:"))
        self.vector_layer_combo = QComboBox()
        self.refresh_vector_layers()
        vector_layer_row.addWidget(self.vector_layer_combo)

        refresh_vector_btn = QPushButton("↻")
        refresh_vector_btn.setMaximumWidth(30)
        refresh_vector_btn.clicked.connect(self.refresh_vector_layers)
        vector_layer_row.addWidget(refresh_vector_btn)
        batch_file_layout.addLayout(vector_layer_row)

        # Or load from file
        vector_file_row = QHBoxLayout()
        self.vector_file_edit = QLineEdit()
        self.vector_file_edit.setPlaceholderText("Or select vector file...")
        vector_file_row.addWidget(self.vector_file_edit)

        browse_vector_btn = QPushButton("...")
        browse_vector_btn.setMaximumWidth(30)
        browse_vector_btn.clicked.connect(self.browse_vector_file)
        vector_file_row.addWidget(browse_vector_btn)
        batch_file_layout.addLayout(vector_file_row)

        # CRS selection
        crs_row = QHBoxLayout()
        crs_row.addWidget(QLabel("CRS:"))
        self.point_crs_edit = QLineEdit()
        self.point_crs_edit.setPlaceholderText("e.g., EPSG:4326 (auto-detect if empty)")
        crs_row.addWidget(self.point_crs_edit)
        batch_file_layout.addLayout(crs_row)

        batch_file_group.setLayout(batch_file_layout)
        batch_layout.addWidget(batch_file_group)

        # Batch Settings
        batch_settings_group = QGroupBox("Batch Settings")
        batch_settings_layout = QVBoxLayout()

        # Batch size filters
        batch_size_row = QHBoxLayout()
        batch_size_row.addWidget(QLabel("Min size:"))
        self.batch_min_size_spin = QSpinBox()
        self.batch_min_size_spin.setRange(0, 1000000)
        self.batch_min_size_spin.setValue(0)
        batch_size_row.addWidget(self.batch_min_size_spin)

        batch_size_row.addWidget(QLabel("Max size:"))
        self.batch_max_size_spin = QSpinBox()
        self.batch_max_size_spin.setRange(0, 10000000)
        self.batch_max_size_spin.setValue(0)
        self.batch_max_size_spin.setSpecialValueText("No limit")
        batch_size_row.addWidget(self.batch_max_size_spin)
        batch_settings_layout.addLayout(batch_size_row)

        # Output options for batch
        batch_output_row = QHBoxLayout()
        self.batch_output_edit = QLineEdit()
        self.batch_output_edit.setPlaceholderText("Output raster file (optional)...")
        batch_output_row.addWidget(self.batch_output_edit)

        browse_batch_output_btn = QPushButton("...")
        browse_batch_output_btn.setMaximumWidth(30)
        browse_batch_output_btn.clicked.connect(self.browse_batch_output)
        batch_output_row.addWidget(browse_batch_output_btn)
        batch_settings_layout.addLayout(batch_output_row)

        # Batch unique values
        self.batch_unique_check = QCheckBox("Unique values for each object")
        self.batch_unique_check.setChecked(True)
        batch_settings_layout.addWidget(self.batch_unique_check)

        batch_settings_group.setLayout(batch_settings_layout)
        batch_layout.addWidget(batch_settings_group)

        # Segment button
        self.batch_segment_btn = QPushButton("Run Batch Segmentation")
        self.batch_segment_btn.clicked.connect(self.segment_by_points_batch)
        batch_layout.addWidget(self.batch_segment_btn)

        # Batch status
        self.batch_status_label = QLabel("")
        batch_layout.addWidget(self.batch_status_label)

        # Info text
        batch_info = QLabel(
            "<i>Batch mode processes each point as a separate prompt, "
            "generating individual masks for each point.</i>"
        )
        batch_info.setWordWrap(True)
        batch_layout.addWidget(batch_info)

        batch_layout.addStretch()
        tab_widget.addTab(batch_tab, "Batch")

        # === Output Tab ===
        output_tab = QWidget()
        output_layout = QVBoxLayout()
        output_tab.setLayout(output_layout)

        output_group = QGroupBox("Output Settings")
        output_group_layout = QVBoxLayout()

        # Output format
        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("Format:"))
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(
            ["Raster (GeoTIFF)", "Vector (GeoPackage)", "Vector (Shapefile)"]
        )
        format_row.addWidget(self.output_format_combo)
        output_group_layout.addLayout(format_row)

        # Unique values
        self.unique_check = QCheckBox("Unique values for each object")
        self.unique_check.setChecked(True)
        output_group_layout.addWidget(self.unique_check)

        # Add to map
        self.add_to_map_check = QCheckBox("Add result to map")
        self.add_to_map_check.setChecked(True)
        output_group_layout.addWidget(self.add_to_map_check)

        # Output path
        output_path_row = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText(
            "Output file path (optional, uses temp file if empty)..."
        )
        output_path_row.addWidget(self.output_path_edit)

        output_browse_btn = QPushButton("...")
        output_browse_btn.setMaximumWidth(30)
        output_browse_btn.clicked.connect(self.browse_output)
        output_path_row.addWidget(output_browse_btn)
        output_group_layout.addLayout(output_path_row)

        self.save_btn = QPushButton("Save Masks")
        self.save_btn.clicked.connect(self.save_masks)
        output_group_layout.addWidget(self.save_btn)

        output_group.setLayout(output_group_layout)
        output_layout.addWidget(output_group)

        # Results info
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(150)
        results_layout.addWidget(self.results_text)

        results_group.setLayout(results_layout)
        output_layout.addWidget(results_group)

        output_layout.addStretch()
        tab_widget.addTab(output_tab, "Output")

        # Add tab widget to main layout
        main_layout.addWidget(tab_widget)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        dock.setWidget(main_widget)
        return dock

    def refresh_layers(self):
        """Refresh the list of raster layers."""
        self.layer_combo.clear()
        layers = QgsProject.instance().mapLayers().values()
        for layer in layers:
            if isinstance(layer, QgsRasterLayer):
                self.layer_combo.addItem(layer.name(), layer.id())

    def refresh_vector_layers(self):
        """Refresh the list of vector layers (for batch point mode)."""
        self.vector_layer_combo.clear()
        self.vector_layer_combo.addItem("-- Select from file instead --", None)
        layers = QgsProject.instance().mapLayers().values()
        for layer in layers:
            if isinstance(layer, QgsVectorLayer):
                # Only include point layers
                if layer.geometryType() == QgsWkbTypes.PointGeometry:
                    self.vector_layer_combo.addItem(layer.name(), layer.id())

    def browse_vector_file(self):
        """Browse for a vector file containing points."""
        file_path, _ = QFileDialog.getOpenFileName(
            self.iface.mainWindow(),
            "Select Vector File",
            "",
            "Vector files (*.geojson *.json *.shp *.gpkg *.kml);;All files (*.*)",
        )
        if file_path:
            self.vector_file_edit.setText(file_path)

    def browse_batch_output(self):
        """Browse for batch output file location."""
        file_path, _ = QFileDialog.getSaveFileName(
            self.iface.mainWindow(), "Save Batch Output", "", "GeoTIFF (*.tif)"
        )
        if file_path:
            self.batch_output_edit.setText(file_path)

    def browse_image(self):
        """Browse for an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self.iface.mainWindow(),
            "Select Image",
            "",
            "Images (*.tif *.tiff *.jpg *.jpeg *.png);;All files (*.*)",
        )
        if file_path:
            self.image_path_edit.setText(file_path)

    def browse_output(self):
        """Browse for output file location."""
        format_text = self.output_format_combo.currentText()
        if "GeoPackage" in format_text:
            filter_str = "GeoPackage (*.gpkg)"
        elif "Shapefile" in format_text:
            filter_str = "Shapefile (*.shp)"
        else:
            filter_str = "GeoTIFF (*.tif)"

        file_path, _ = QFileDialog.getSaveFileName(
            self.iface.mainWindow(), "Save Output", "", filter_str
        )
        if file_path:
            self.output_path_edit.setText(file_path)

    def load_model(self):
        """Load the SamGeo model."""
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate
            self.model_status.setText("Loading model...")
            self.model_status.setStyleSheet("color: orange;")
            QCoreApplication.processEvents()

            model_version = self.model_combo.currentText()
            backend = self.backend_combo.currentText()
            device = self.device_combo.currentText()
            if device == "auto":
                device = None

            confidence = self.conf_spin.value()
            enable_interactive = self.interactive_check.isChecked()

            # Import and initialize the appropriate model
            if "SamGeo3" in model_version:
                from samgeo import SamGeo3

                self.sam = SamGeo3(
                    backend=backend,
                    device=device,
                    confidence_threshold=confidence,
                    enable_inst_interactivity=enable_interactive,
                )
                model_name = "SamGeo3"
            elif "SamGeo2" in model_version:
                from samgeo import SamGeo2

                self.sam = SamGeo2(
                    device=device,
                )
                model_name = "SamGeo2"
            else:
                from samgeo import SamGeo

                self.sam = SamGeo(
                    device=device,
                )
                model_name = "SamGeo"

            self.model_status.setText(f"Model: {model_name} loaded")
            self.model_status.setStyleSheet("color: green;")
            self.log_message(f"{model_name} model loaded successfully")

        except Exception as e:
            self.model_status.setText("Model: Failed to load")
            self.model_status.setStyleSheet("color: red;")
            self.show_error(f"Failed to load model: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def set_image_from_layer(self):
        """Set the image from the selected QGIS layer."""
        if self.sam is None:
            self.show_error("Please load the model first.")
            return

        layer_id = self.layer_combo.currentData()
        if not layer_id:
            self.show_error("Please select a raster layer.")
            return

        layer = QgsProject.instance().mapLayer(layer_id)
        if not layer:
            self.show_error("Layer not found.")
            return

        # Get the layer's file path
        source = layer.source()
        if not os.path.exists(source):
            self.show_error(f"Layer source file not found: {source}")
            return

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.image_status.setText("Setting image...")
            QCoreApplication.processEvents()

            self.sam.set_image(source)
            self.current_layer = layer
            self.current_image_path = source

            self.image_status.setText(f"Image: {layer.name()}")
            self.image_status.setStyleSheet("color: green;")
            self.log_message(f"Image set from layer: {layer.name()}")

        except Exception as e:
            self.image_status.setText("Image: Failed to set")
            self.image_status.setStyleSheet("color: red;")
            self.show_error(f"Failed to set image: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def set_image_from_file(self):
        """Set the image from the file path."""
        if self.sam is None:
            self.show_error("Please load the model first.")
            return

        file_path = self.image_path_edit.text()
        if not file_path or not os.path.exists(file_path):
            self.show_error("Please select a valid image file.")
            return

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.image_status.setText("Setting image...")
            QCoreApplication.processEvents()

            self.sam.set_image(file_path)
            self.current_image_path = file_path
            self.current_layer = None

            self.image_status.setText(f"Image: {os.path.basename(file_path)}")
            self.image_status.setStyleSheet("color: green;")
            self.log_message(f"Image set from file: {file_path}")

            # Optionally add the layer to the map
            layer = QgsRasterLayer(file_path, os.path.basename(file_path))
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                self.current_layer = layer
                self.refresh_layers()

        except Exception as e:
            self.image_status.setText("Image: Failed to set")
            self.image_status.setStyleSheet("color: red;")
            self.show_error(f"Failed to set image: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def segment_by_text(self):
        """Segment the image using text prompt."""
        if self.sam is None:
            self.show_error("Please load the model first.")
            return

        if self.current_image_path is None:
            self.show_error("Please set an image first.")
            return

        prompt = self.text_prompt_edit.text().strip()
        if not prompt:
            self.show_error("Please enter a text prompt.")
            return

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.text_status_label.setText("Processing...")
            self.text_status_label.setStyleSheet("color: orange;")
            QCoreApplication.processEvents()

            min_size = self.min_size_spin.value()
            max_size = (
                self.max_size_spin.value() if self.max_size_spin.value() > 0 else None
            )

            self.sam.generate_masks(prompt, min_size=min_size, max_size=max_size)

            num_masks = len(self.sam.masks) if self.sam.masks else 0
            self.results_text.setText(
                f"Text Segmentation Results:\n"
                f"Prompt: {prompt}\n"
                f"Objects found: {num_masks}\n"
            )

            # Update status label
            if num_masks > 0:
                self.text_status_label.setText(
                    f"Found {num_masks} object(s). Go to Output tab to save."
                )
                self.text_status_label.setStyleSheet("color: green;")
            else:
                self.text_status_label.setText(
                    "No objects found. Try a different prompt."
                )
                self.text_status_label.setStyleSheet("color: orange;")

            self.log_message(f"Text segmentation complete. Found {num_masks} objects.")

        except Exception as e:
            self.text_status_label.setText("Segmentation failed!")
            self.text_status_label.setStyleSheet("color: red;")
            self.show_error(f"Segmentation failed: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def start_point_tool(self, foreground=True):
        """Start the point prompt tool."""
        if self.current_layer is None:
            self.show_error("Please set an image first.")
            self.add_fg_point_btn.setChecked(False)
            self.add_bg_point_btn.setChecked(False)
            return

        # Uncheck the other button
        if foreground:
            self.add_bg_point_btn.setChecked(False)
        else:
            self.add_fg_point_btn.setChecked(False)

        if self.point_tool is None:
            self.point_tool = PointPromptTool(self.canvas, self)

        self.point_tool.set_foreground(foreground)
        self.previous_tool = self.canvas.mapTool()
        self.canvas.setMapTool(self.point_tool)

    def add_point(self, point, foreground):
        """Add a point prompt."""
        # Convert map coordinates to pixel coordinates
        if self.current_layer is not None:
            extent = self.current_layer.extent()
            width = self.current_layer.width()
            height = self.current_layer.height()

            # Calculate pixel coordinates
            px = (point.x() - extent.xMinimum()) / extent.width() * width
            py = (extent.yMaximum() - point.y()) / extent.height() * height

            self.point_coords.append([px, py])
            self.point_labels.append(1 if foreground else 0)

            # Update list widget
            label_text = "FG" if foreground else "BG"
            item = QListWidgetItem(f"{label_text}: ({px:.1f}, {py:.1f})")
            item.setForeground(Qt.green if foreground else Qt.red)
            self.points_list.addItem(item)

    def clear_points(self):
        """Clear all point prompts."""
        self.point_coords = []
        self.point_labels = []
        self.points_list.clear()

        # Clear rubber bands from point tool
        if self.point_tool is not None:
            self.point_tool.clear_markers()

    def start_batch_point_tool(self):
        """Start the batch point tool for adding multiple points."""
        if self.current_layer is None:
            self.show_error("Please set an image first.")
            self.batch_add_point_btn.setChecked(False)
            return

        if self.batch_point_tool is None:
            self.batch_point_tool = PointPromptTool(self.canvas, self, batch_mode=True)

        self.batch_point_tool.set_foreground(True)  # All batch points are foreground
        self.previous_tool = self.canvas.mapTool()
        self.canvas.setMapTool(self.batch_point_tool)

    def add_batch_point(self, point):
        """Add a batch point prompt."""
        if self.current_layer is not None:
            extent = self.current_layer.extent()
            width = self.current_layer.width()
            height = self.current_layer.height()

            # Calculate pixel coordinates
            px = (point.x() - extent.xMinimum()) / extent.width() * width
            py = (extent.yMaximum() - point.y()) / extent.height() * height

            self.batch_point_coords.append([px, py])
            self.batch_point_coords_map.append([point.x(), point.y()])

            # Update list widget
            item = QListWidgetItem(
                f"Point {len(self.batch_point_coords)}: ({px:.1f}, {py:.1f})"
            )
            item.setForeground(Qt.green)
            self.batch_points_list.addItem(item)

            # Update count label
            self.batch_points_count_label.setText(
                f"Points: {len(self.batch_point_coords)}"
            )

    def clear_batch_points(self):
        """Clear all batch point prompts."""
        self.batch_point_coords = []
        self.batch_point_coords_map = []
        self.batch_points_list.clear()
        self.batch_points_count_label.setText("Points: 0")

        # Clear markers from batch point tool
        if self.batch_point_tool is not None:
            self.batch_point_tool.clear_markers()

    def segment_by_points(self):
        """Segment using point prompts."""
        if self.sam is None:
            self.show_error("Please load the model first.")
            return

        if self.current_image_path is None:
            self.show_error("Please set an image first.")
            return

        if not self.point_coords:
            self.show_error("Please add at least one point.")
            return

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.point_status_label.setText("Processing...")
            self.point_status_label.setStyleSheet("color: orange;")
            QCoreApplication.processEvents()

            import numpy as np

            point_coords = np.array(self.point_coords)
            point_labels = np.array(self.point_labels)

            # Use appropriate method based on model type
            if hasattr(self.sam, "generate_masks_by_points"):
                self.sam.generate_masks_by_points(
                    point_coords=point_coords.tolist(),
                    point_labels=point_labels.tolist(),
                )
            else:
                # Fallback for older SamGeo versions
                self.sam.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                )

            num_masks = len(self.sam.masks) if self.sam.masks else 0
            self.results_text.setText(
                f"Point Segmentation Results:\n"
                f"Points used: {len(self.point_coords)}\n"
                f"Objects found: {num_masks}\n"
            )

            # Update status label
            if num_masks > 0:
                self.point_status_label.setText(
                    f"Found {num_masks} object(s). Go to Output tab to save."
                )
                self.point_status_label.setStyleSheet("color: green;")
            else:
                self.point_status_label.setText(
                    "No objects found. Try different points."
                )
                self.point_status_label.setStyleSheet("color: orange;")

            self.log_message(f"Point segmentation complete. Found {num_masks} objects.")

            # Deactivate tool
            self.add_fg_point_btn.setChecked(False)
            self.add_bg_point_btn.setChecked(False)
            if self.previous_tool:
                self.canvas.setMapTool(self.previous_tool)

        except Exception as e:
            self.point_status_label.setText("Segmentation failed!")
            self.point_status_label.setStyleSheet("color: red;")
            self.show_error(f"Segmentation failed: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def start_box_tool(self):
        """Start the box prompt tool."""
        if self.current_layer is None:
            self.show_error("Please set an image first.")
            self.draw_box_btn.setChecked(False)
            return

        if self.box_tool is None:
            self.box_tool = BoxPromptTool(self.canvas, self)

        self.previous_tool = self.canvas.mapTool()
        self.canvas.setMapTool(self.box_tool)

    def set_box(self, rect):
        """Set the box prompt from a rectangle."""
        if self.current_layer is not None:
            extent = self.current_layer.extent()
            width = self.current_layer.width()
            height = self.current_layer.height()

            # Convert to pixel coordinates
            x1 = (rect.xMinimum() - extent.xMinimum()) / extent.width() * width
            y1 = (extent.yMaximum() - rect.yMaximum()) / extent.height() * height
            x2 = (rect.xMaximum() - extent.xMinimum()) / extent.width() * width
            y2 = (extent.yMaximum() - rect.yMinimum()) / extent.height() * height

            self.box_coords = [x1, y1, x2, y2]
            self.box_label.setText(f"Box: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f})")

    def clear_box(self):
        """Clear the box prompt."""
        self.box_coords = None
        self.box_label.setText("Box: Not set")

        if self.box_tool is not None:
            self.box_tool.clear_rubber_band()

    def segment_by_box(self):
        """Segment using box prompt."""
        if self.sam is None:
            self.show_error("Please load the model first.")
            return

        if self.current_image_path is None:
            self.show_error("Please set an image first.")
            return

        if self.box_coords is None:
            self.show_error("Please draw a box first.")
            return

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.box_status_label.setText("Processing...")
            self.box_status_label.setStyleSheet("color: orange;")
            QCoreApplication.processEvents()

            # Use appropriate method based on model type
            if hasattr(self.sam, "generate_masks_by_boxes"):
                self.sam.generate_masks_by_boxes(boxes=[self.box_coords])
            else:
                # Fallback for older SamGeo versions
                import numpy as np

                self.sam.predict(box=np.array(self.box_coords))

            num_masks = len(self.sam.masks) if self.sam.masks else 0
            self.results_text.setText(
                f"Box Segmentation Results:\n"
                f"Box: {self.box_coords}\n"
                f"Objects found: {num_masks}\n"
            )

            # Update status label
            if num_masks > 0:
                self.box_status_label.setText(
                    f"Found {num_masks} object(s). Go to Output tab to save."
                )
                self.box_status_label.setStyleSheet("color: green;")
            else:
                self.box_status_label.setText("No objects found. Try a different box.")
                self.box_status_label.setStyleSheet("color: orange;")

            self.log_message(f"Box segmentation complete. Found {num_masks} objects.")

            # Deactivate tool
            self.draw_box_btn.setChecked(False)
            if self.previous_tool:
                self.canvas.setMapTool(self.previous_tool)

        except Exception as e:
            self.box_status_label.setText("Segmentation failed!")
            self.box_status_label.setStyleSheet("color: red;")
            self.show_error(f"Segmentation failed: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def segment_by_points_batch(self):
        """Segment using batch point prompts from interactive points or vector file/layer."""
        if self.sam is None:
            self.show_error("Please load the model first.")
            return

        if self.current_image_path is None:
            self.show_error("Please set an image first.")
            return

        # Check if batch method is available
        if not hasattr(self.sam, "generate_masks_by_points_patch"):
            self.show_error(
                "Batch point mode requires SamGeo3 with enable_inst_interactivity=True.\n"
                "Please reload the model with the correct settings."
            )
            return

        # Check for interactive batch points first
        use_interactive_points = len(self.batch_point_coords) > 0

        # Get point source (interactive, layer, or file)
        point_source = None
        point_crs = None
        source_description = ""

        if use_interactive_points:
            # Use interactive points (already in pixel coordinates)
            point_source = self.batch_point_coords
            point_crs = None  # Already in pixel coordinates
            source_description = f"{len(self.batch_point_coords)} interactive points"
        else:
            # Check if a layer is selected
            layer_id = self.vector_layer_combo.currentData()
            if layer_id:
                layer = QgsProject.instance().mapLayer(layer_id)
                if layer and layer.isValid():
                    point_source = layer.source()
                    source_description = layer.name()
                    # Get CRS from layer
                    if layer.crs().isValid():
                        point_crs = layer.crs().authid()

            # If no layer, check for file path
            if point_source is None:
                vector_file = self.vector_file_edit.text().strip()
                if vector_file and os.path.exists(vector_file):
                    point_source = vector_file
                    source_description = os.path.basename(vector_file)
                else:
                    self.show_error(
                        "Please add points interactively or select a vector layer/file."
                    )
                    return

            # Get CRS from input field if specified
            crs_text = self.point_crs_edit.text().strip()
            if crs_text:
                point_crs = crs_text

        # Get output path
        output_path = self.batch_output_edit.text().strip()
        if not output_path:
            output_path = None  # Will only store in memory

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.batch_status_label.setText("Processing batch segmentation...")
            self.batch_status_label.setStyleSheet("color: orange;")
            QCoreApplication.processEvents()

            # Get size filters
            min_size = self.batch_min_size_spin.value()
            max_size = (
                self.batch_max_size_spin.value()
                if self.batch_max_size_spin.value() > 0
                else None
            )
            unique = self.batch_unique_check.isChecked()

            # Run batch segmentation
            self.sam.generate_masks_by_points_patch(
                point_coords_batch=point_source,
                point_crs=point_crs,
                output=output_path,
                unique=unique,
                min_size=min_size,
                max_size=max_size,
            )

            num_masks = len(self.sam.masks) if self.sam.masks else 0

            # Update results
            result_text = (
                f"Batch Point Segmentation Results:\n"
                f"Source: {source_description}\n"
                f"Objects found: {num_masks}\n"
            )
            if output_path:
                result_text += f"Output saved to: {output_path}\n"

            self.results_text.setText(result_text)

            # Update status label
            if num_masks > 0:
                self.batch_status_label.setText(
                    f"Found {num_masks} object(s). Go to Output tab to save."
                )
                self.batch_status_label.setStyleSheet("color: green;")
            else:
                self.batch_status_label.setText(
                    "No objects found. Try different points."
                )
                self.batch_status_label.setStyleSheet("color: orange;")

            self.log_message(
                f"Batch point segmentation complete. Found {num_masks} objects."
            )

            # Deactivate batch point tool if active
            self.batch_add_point_btn.setChecked(False)

            # Add output to map if saved and option is checked
            if output_path and self.add_to_map_check.isChecked():
                layer = QgsRasterLayer(output_path, os.path.basename(output_path))
                if layer.isValid():
                    QgsProject.instance().addMapLayer(layer)
                    self.results_text.append("Added result layer to map.")

        except Exception as e:
            self.batch_status_label.setText("Failed!")
            self.batch_status_label.setStyleSheet("color: red;")
            self.show_error(f"Batch segmentation failed: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def save_masks(self):
        """Save the segmentation masks."""
        if self.sam is None or self.sam.masks is None or len(self.sam.masks) == 0:
            self.show_error("No masks to save. Please run segmentation first.")
            return

        import tempfile

        output_path = self.output_path_edit.text().strip()
        format_text = self.output_format_combo.currentText()

        # Generate temp file path if not specified
        use_temp_file = False
        if not output_path:
            use_temp_file = True
            if "Raster" in format_text:
                temp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
                output_path = temp_file.name
                temp_file.close()
            elif "GeoPackage" in format_text:
                temp_file = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
                output_path = temp_file.name
                temp_file.close()
            else:  # Shapefile
                temp_dir = tempfile.mkdtemp()
                output_path = os.path.join(temp_dir, "masks.shp")

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            QCoreApplication.processEvents()

            unique = self.unique_check.isChecked()

            if "Raster" in format_text:
                # Save as raster
                self.sam.save_masks(output=output_path, unique=unique)

                if self.add_to_map_check.isChecked():
                    layer_name = (
                        "samgeo_masks"
                        if use_temp_file
                        else os.path.basename(output_path)
                    )
                    layer = QgsRasterLayer(output_path, layer_name)
                    if layer.isValid():
                        QgsProject.instance().addMapLayer(layer)
            else:
                # Save as vector - first save as raster, then convert
                temp_raster = tempfile.NamedTemporaryFile(
                    suffix=".tif", delete=False
                ).name
                try:
                    self.sam.save_masks(output=temp_raster, unique=unique)

                    # Convert raster to vector
                    from samgeo import common

                    common.raster_to_vector(temp_raster, output_path)

                    if self.add_to_map_check.isChecked():
                        layer_name = (
                            "samgeo_masks"
                            if use_temp_file
                            else os.path.basename(output_path)
                        )
                        layer = QgsVectorLayer(output_path, layer_name, "ogr")
                        if layer.isValid():
                            QgsProject.instance().addMapLayer(layer)
                finally:
                    if os.path.exists(temp_raster):
                        os.remove(temp_raster)
            self.results_text.append(f"\nSaved to: {output_path}")
            self.log_message(f"Masks saved to: {output_path}")

        except Exception as e:
            self.show_error(f"Failed to save masks: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def show_error(self, message):
        """Show an error message."""
        QMessageBox.critical(self.iface.mainWindow(), "SamGeo Error", message)
        self.log_message(message, level=Qgis.Critical)

    def log_message(self, message, level=Qgis.Info):
        """Log a message to QGIS."""
        QgsMessageLog.logMessage(message, "SamGeo", level)
