"""
SamGeo QGIS Plugin - Main Plugin Class
"""

import os

from qgis.PyQt.QtCore import Qt, QCoreApplication, QVariant
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import (
    QAction,
    QToolBar,
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
    QgsField,
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsWkbTypes,
    Qgis,
    QgsMessageLog,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
)

# Import the map tools
from .map_tools import PointPromptTool, BoxPromptTool
from .plugin_utils import (
    PROMPT_BATCH_POINT,
    PROMPT_BOX,
    PROMPT_FIELD,
    PROMPT_POINT,
    PROMPT_TEXT,
    SCALE_MAX,
    SCALE_MIN,
    SOURCE_FIELD,
    TILE_OVERLAP_DEFAULT,
    TILE_SIZE_DEFAULT,
    default_output_path,
    failed_tile_messages,
    is_geotiff,
    mask_attributes,
    model_constructor_kwargs,
    model_family,
    model_weight_options,
    output_extension,
    scaled_size,
    source_directory,
    tile_count,
    validate_tile_settings,
)

TOOLBAR_OBJECT_NAME = "SamGeo"
MENU_TITLE = "&SamGeo"


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
        self._remove_toolbars_by_object_name()
        self._remove_menus_by_title()
        self.toolbar = self.iface.addToolBar("SamGeo")
        self.toolbar.setObjectName(TOOLBAR_OBJECT_NAME)

        # Dock widget
        self.dock_widget = None

        # SamGeo model instance
        self.sam = None
        self.current_layer = None
        self.current_image_path = None
        # Prompt type and source image of the current masks, written to the
        # ``prompt`` and ``source`` fields. Recorded only after a successful
        # segmentation so a failed run or a new image does not relabel them.
        self.last_prompt_type = None
        self.mask_source_path = None
        # Set when tiled mode replaced the model's image with its last tile
        self.image_needs_reset = False

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

        # Dependency management
        self._deps_available = False
        self._deps_dock = None
        self._deps_install_worker = None

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

        # Clear GPU Memory action (menu only) - use QGIS refresh/clear icon
        gpu_icon = ":/images/themes/default/mActionRefresh.svg"
        self.add_action(
            gpu_icon,
            text=self.tr("Clear GPU Memory"),
            callback=self.clear_gpu_memory,
            parent=self.iface.mainWindow(),
            status_tip=self.tr("Release GPU memory and clear CUDA cache"),
            add_to_toolbar=False,
        )

        # Check for Updates action (menu only) - use QGIS download/update icon
        update_icon = ":/images/themes/default/mActionFileOpen.svg"
        self.add_action(
            update_icon,
            text=self.tr("Check for Updates..."),
            callback=self.show_update_checker,
            parent=self.iface.mainWindow(),
            status_tip=self.tr("Check for plugin updates from GitHub"),
            add_to_toolbar=False,
        )

        # About action (menu only) - use QGIS help/info icon
        about_icon = ":/images/themes/default/mActionHelpContents.svg"
        self.add_action(
            about_icon,
            text=self.tr("About SamGeo"),
            callback=self.show_about,
            parent=self.iface.mainWindow(),
            status_tip=self.tr("About SamGeo Plugin"),
            add_to_toolbar=False,
        )

    def _remove_toolbar(self, toolbar):
        """Detach and schedule deletion of a plugin toolbar widget."""
        if toolbar is None:
            return

        main_window = self.iface.mainWindow()
        actions = []
        try:
            actions = list(toolbar.actions())
        except Exception:
            pass  # nosec B110
        try:
            toolbar.clear()
        except Exception:
            pass  # nosec B110
        for action in actions:
            try:
                action.deleteLater()
            except Exception:
                pass  # nosec B110
        try:
            main_window.removeToolBar(toolbar)
        except Exception:
            pass  # nosec B110
        try:
            toolbar.hide()
        except Exception:
            pass  # nosec B110
        try:
            toolbar.setParent(None)
        except Exception:
            pass  # nosec B110
        try:
            toolbar.deleteLater()
        except Exception:
            pass  # nosec B110

    def _remove_toolbars_by_object_name(self):
        """Remove current or stale plugin toolbars from QGIS."""
        main_window = self.iface.mainWindow()
        for toolbar in main_window.findChildren(QToolBar, TOOLBAR_OBJECT_NAME):
            self._remove_toolbar(toolbar)

    def _plugin_menu_titles(self):
        """Return possible translated and untranslated plugin menu titles."""
        titles = {MENU_TITLE}
        translator = getattr(self, "tr", None)
        if callable(translator):
            try:
                titles.add(translator(MENU_TITLE))
            except Exception:
                pass  # nosec B110
        return titles

    def _remove_menu(self, menu):
        """Detach and schedule deletion of a plugin menu."""
        if menu is None:
            return

        main_window = self.iface.mainWindow()
        try:
            menu.clear()
        except Exception:
            pass  # nosec B110
        try:
            main_window.menuBar().removeAction(menu.menuAction())
        except Exception:
            pass  # nosec B110
        try:
            menu.setParent(None)
        except Exception:
            pass  # nosec B110
        try:
            menu.deleteLater()
        except Exception:
            pass  # nosec B110

    def _remove_menus_by_title(self):
        """Remove current or stale plugin menus from QGIS."""
        menu_bar = self.iface.mainWindow().menuBar()
        titles = self._plugin_menu_titles()
        for action in menu_bar.actions():
            menu = action.menu()
            if menu is not None and menu.title() in titles:
                self._remove_menu(menu)

    def unload(self):
        """Remove the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginRasterMenu(self.menu, action)
            self.iface.removeToolBarIcon(action)

        # Remove dock widget
        if self.dock_widget is not None:
            self.iface.removeDockWidget(self.dock_widget)

        # Clean up deps installer
        if self._deps_install_worker is not None:
            if self._deps_install_worker.isRunning():
                self._deps_install_worker.cancel()
                self._deps_install_worker.terminate()
                self._deps_install_worker.wait(5000)
            self._deps_install_worker = None

        if self._deps_dock is not None:
            self.iface.removeDockWidget(self._deps_dock)
            self._deps_dock = None

        # Remove toolbar
        del self.toolbar

        # Clean up model
        if self.sam is not None:
            del self.sam
            self.sam = None

        self._remove_toolbars_by_object_name()
        self._remove_menus_by_title()

    # ------------------------------------------------------------------
    # Dependency management
    # ------------------------------------------------------------------

    def _ensure_dependencies(self):
        """Check if dependencies are installed; show installer if not.

        Returns:
            True if dependencies are available, False if installer was shown.
        """
        if self._deps_available:
            return True

        try:
            from .core.venv_manager import (
                ensure_venv_packages_available,
                get_venv_status,
            )

            is_ready, message = get_venv_status()
            if is_ready:
                ensure_venv_packages_available()
                self._deps_available = True
                return True
        except Exception:  # nosec B110
            pass

        self._show_deps_install_dock()
        return False

    def _show_deps_install_dock(self):
        """Create and show the dependency installation dock widget."""
        if self._deps_dock is not None:
            self._deps_dock.show()
            self._deps_dock.raise_()
            return

        from .deps_install_dialog import DepsInstallDockWidget

        self._deps_dock = DepsInstallDockWidget(self.iface.mainWindow())
        self._deps_dock.install_requested.connect(self._on_install_requested)
        self._deps_dock.cancel_requested.connect(self._on_cancel_install)

        self.iface.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._deps_dock)
        self._deps_dock.show()
        self._deps_dock.raise_()

    def _on_install_requested(self):
        """Handle install button click from the deps dock."""
        if (
            self._deps_install_worker is not None
            and self._deps_install_worker.isRunning()
        ):
            return

        from .core.venv_manager import detect_nvidia_gpu
        from .workers.deps_install_worker import DepsInstallWorker

        has_gpu, _ = detect_nvidia_gpu()

        self._deps_install_worker = DepsInstallWorker(cuda_enabled=has_gpu)
        self._deps_install_worker.progress.connect(self._on_install_progress)
        self._deps_install_worker.finished.connect(self._on_install_finished)

        if self._deps_dock:
            self._deps_dock.show_progress_ui()

        self._deps_install_worker.start()

    def _on_install_progress(self, percent, message):
        """Handle progress updates from the install worker.

        Args:
            percent: Progress percentage (0-100).
            message: Status message.
        """
        if self._deps_dock:
            self._deps_dock.set_progress(percent, message)

    def _on_install_finished(self, success, message):
        """Handle installation completion.

        Args:
            success: Whether installation succeeded.
            message: Completion message.
        """
        if self._deps_dock:
            self._deps_dock.show_complete_ui(success, message)

        if success:
            self._deps_available = True

            # Ensure venv packages are on sys.path
            try:
                from .core.venv_manager import ensure_venv_packages_available

                ensure_venv_packages_available()
            except Exception:  # nosec B110
                pass

            # Close the deps dock and open the main plugin panel
            if self._deps_dock:
                self.iface.removeDockWidget(self._deps_dock)
                self._deps_dock.deleteLater()
                self._deps_dock = None

            # Now show the main plugin dock
            if self.dock_widget is None:
                self.dock_widget = self.create_dock_widget()
                self.iface.addDockWidget(
                    Qt.DockWidgetArea.RightDockWidgetArea, self.dock_widget
                )
            else:
                self.dock_widget.show()

    def _on_cancel_install(self):
        """Handle cancel button click during installation."""
        if (
            self._deps_install_worker is not None
            and self._deps_install_worker.isRunning()
        ):
            self._deps_install_worker.cancel()
        if self._deps_dock:
            self._deps_dock.show_install_ui()

    def run(self):
        """Run the plugin - check deps then show the dock widget."""
        if not self._ensure_dependencies():
            return  # Installer dock is now showing

        if self.dock_widget is None:
            self.dock_widget = self.create_dock_widget()
            self.iface.addDockWidget(
                Qt.DockWidgetArea.RightDockWidgetArea, self.dock_widget
            )
        else:
            self.dock_widget.show()

    def create_dock_widget(self):
        """Create the dock widget with all controls."""
        dock = QDockWidget("SamGeo Segmentation", self.iface.mainWindow())
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )

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

        # Model weights (follows the selected model family)
        weights_row = QHBoxLayout()
        weights_row.addWidget(QLabel("Weights:"))
        self.weights_combo = QComboBox()
        self.weights_combo.setToolTip(
            "Model checkpoint size. Larger weights are more accurate but\n"
            "slower and need more memory."
        )
        weights_row.addWidget(self.weights_combo)
        backend_layout.addLayout(weights_row)
        self.model_combo.currentTextChanged.connect(self._update_weight_options)
        self._update_weight_options(self.model_combo.currentText())

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

        # Tiled mode for large images
        self.tiled_check = QCheckBox("Tiled mode for large images")
        self.tiled_check.setChecked(False)
        self.tiled_check.setToolTip(
            "Split the image into overlapping tiles, segment each tile with\n"
            "the text prompt and merge the results (SamGeo3, GeoTIFF only).\n"
            "SAM works best on image patches of about 1000 pixels, so this\n"
            "gives better results on full scenes. Setting the image first is\n"
            "not required; the result is saved using the Output tab settings."
        )
        text_group_layout.addWidget(self.tiled_check)

        self.tiled_options = QWidget()
        tiled_layout = QVBoxLayout()
        tiled_layout.setContentsMargins(0, 0, 0, 0)
        self.tiled_options.setLayout(tiled_layout)

        tile_row = QHBoxLayout()
        tile_row.addWidget(QLabel("Tile size:"))
        self.tile_size_spin = QSpinBox()
        self.tile_size_spin.setRange(256, 8192)
        self.tile_size_spin.setSingleStep(128)
        self.tile_size_spin.setValue(TILE_SIZE_DEFAULT)
        self.tile_size_spin.setSuffix(" px")
        tile_row.addWidget(self.tile_size_spin)

        tile_row.addWidget(QLabel("Overlap:"))
        self.tile_overlap_spin = QSpinBox()
        self.tile_overlap_spin.setRange(0, 4096)
        self.tile_overlap_spin.setSingleStep(32)
        self.tile_overlap_spin.setValue(TILE_OVERLAP_DEFAULT)
        self.tile_overlap_spin.setSuffix(" px")
        self.tile_overlap_spin.setToolTip(
            "Overlap between neighboring tiles. More overlap merges objects\n"
            "on tile edges better but takes longer."
        )
        tile_row.addWidget(self.tile_overlap_spin)
        tiled_layout.addLayout(tile_row)

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Processing scale:"))
        self.tile_scale_spin = QSpinBox()
        self.tile_scale_spin.setRange(SCALE_MIN, SCALE_MAX)
        self.tile_scale_spin.setSingleStep(10)
        self.tile_scale_spin.setValue(100)
        self.tile_scale_spin.setSuffix(" %")
        self.tile_scale_spin.setToolTip(
            "Resample the image before tiling. Below 100% processes a coarser\n"
            "copy (faster, suits large objects); above 100% an upsampled one\n"
            "(slower, can help with small objects). The mask keeps the\n"
            "processing resolution."
        )
        scale_row.addWidget(self.tile_scale_spin)
        tiled_layout.addLayout(scale_row)

        self.tiled_options.setVisible(False)
        self.tiled_check.toggled.connect(self.tiled_options.setVisible)
        text_group_layout.addWidget(self.tiled_options)

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

        # Auto-show results after segmentation
        self.auto_show_check = QCheckBox("Auto-show results after segmentation")
        self.auto_show_check.setChecked(True)
        self.auto_show_check.setToolTip(
            "Automatically save and display results after running segmentation\n"
            "in the Text, Interactive, or Batch tabs"
        )
        output_group_layout.addWidget(self.auto_show_check)

        # Save next to the source image when no output path is given
        self.save_to_source_dir_check = QCheckBox(
            "Save next to source image when no path is set"
        )
        self.save_to_source_dir_check.setChecked(False)
        self.save_to_source_dir_check.setToolTip(
            "If the output path is empty, write the result to the source\n"
            "image's folder as <image>_<prompt>_masks.<ext> instead of a\n"
            "temporary file. Existing files are never overwritten."
        )
        output_group_layout.addWidget(self.save_to_source_dir_check)

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
                if layer.geometryType() == QgsWkbTypes.GeometryType.PointGeometry:
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
            self.iface.mainWindow(),
            "Save Batch Output",
            default_output_path(self.current_image_path, PROMPT_BATCH_POINT, ".tif"),
            "GeoTIFF (*.tif)",
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

        start_path = self.output_path_edit.text().strip()
        if not start_path:
            source = self.mask_source_path or self.current_image_path
            start_path = default_output_path(
                source,
                self.last_prompt_type,
                output_extension(format_text),
            ) or source_directory(source)

        file_path, _ = QFileDialog.getSaveFileName(
            self.iface.mainWindow(), "Save Output", start_path, filter_str
        )
        if file_path:
            self.output_path_edit.setText(file_path)

    def _update_weight_options(self, model_text):
        """Fill the weights combo with the choices of the selected family.

        Args:
            model_text: Text of the model combo, e.g. ``"SamGeo2 (SAM2)"``.
        """
        self.weights_combo.clear()
        for label, value in model_weight_options(model_family(model_text)):
            self.weights_combo.addItem(label, value)
        self.weights_combo.setCurrentIndex(0)

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

            family = model_family(model_version)
            weight_kwargs = model_constructor_kwargs(
                family, self.weights_combo.currentData()
            )
            weight_name = next(iter(weight_kwargs.values()))

            try:
                from ._samgeo_lib import get_samgeo
            except ImportError:
                from _samgeo_lib import get_samgeo

            samgeo = get_samgeo()

            # Import and initialize the appropriate model
            if family == "sam3":
                self.sam = samgeo.SamGeo3(
                    backend=backend,
                    device=device,
                    confidence_threshold=confidence,
                    enable_inst_interactivity=enable_interactive,
                    **weight_kwargs,
                )
                model_name = "SamGeo3"
            elif family == "sam2":
                self.sam = samgeo.SamGeo2(
                    device=device,
                    **weight_kwargs,
                )
                model_name = "SamGeo2"
            else:
                self.sam = samgeo.SamGeo(
                    device=device,
                    **weight_kwargs,
                )
                model_name = "SamGeo"
            model_name = f"{model_name} ({weight_name})"

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
            self.image_needs_reset = False
            self._clear_mask_result()

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
            self.image_needs_reset = False
            self._clear_mask_result()
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

        if self.tiled_check.isChecked():
            self.segment_by_text_tiled()
            return

        if not self._image_ready():
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
            self._record_mask_result(PROMPT_TEXT)

            num_masks = len(self.sam.masks) if self.sam.masks else 0
            self.results_text.setText(
                f"Text Segmentation Results:\n"
                f"Prompt: {prompt}\n"
                f"Objects found: {num_masks}\n"
            )

            # Update status label
            if num_masks > 0:
                if self.auto_show_check.isChecked():
                    self.text_status_label.setText(f"Found {num_masks} object(s).")
                else:
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

            # Auto-show results if enabled
            if num_masks > 0:
                self._auto_show_results()

        except Exception as e:
            self.text_status_label.setText("Segmentation failed!")
            self.text_status_label.setStyleSheet("color: red;")
            self.show_error(f"Segmentation failed: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def _image_ready(self):
        """Check that an image is set on the model, showing an error if not.

        Returns:
            True if prompts can run against the current image.
        """
        if self.current_image_path is None:
            self.show_error("Please set an image first.")
            return False
        if self.image_needs_reset:
            self.show_error(
                "Tiled mode replaced the model's image. Please set the image "
                "again before using this prompt."
            )
            return False
        return True

    def _tiled_source_path(self):
        """Return the raster path to use for tiled text segmentation.

        Tiled mode reads the file directly, so the image does not have to be
        set on the model first. The current image wins; otherwise the
        selected raster layer, then the file path on the Model tab.

        Returns:
            The source path, or None if no usable file is selected.
        """
        if self.current_image_path and os.path.exists(self.current_image_path):
            return self.current_image_path
        layer_id = self.layer_combo.currentData()
        if layer_id:
            layer = QgsProject.instance().mapLayer(layer_id)
            if layer is not None and os.path.exists(layer.source()):
                return layer.source()
        file_path = self.image_path_edit.text().strip()
        if file_path and os.path.exists(file_path):
            return file_path
        return None

    def _resample_for_tiling(self, source, scale):
        """Write a resampled GeoTIFF copy of the source at a processing scale.

        Args:
            source: Path of the source GeoTIFF.
            scale: Processing scale in percent of the native resolution.

        Returns:
            The path of the temporary resampled GeoTIFF.
        """
        import tempfile

        from osgeo import gdal

        temp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        output = temp_file.name
        temp_file.close()
        options = dict(
            format="GTiff",
            widthPct=scale,
            heightPct=scale,
            resampleAlg="average" if scale < 100 else "bilinear",
            creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"],
        )
        # Raise GDAL errors as exceptions locally without changing the
        # process-wide setting that QGIS and other plugins rely on.
        try:
            if hasattr(gdal, "ExceptionMgr"):
                with gdal.ExceptionMgr(useExceptions=True):
                    dataset = gdal.Translate(output, source, **options)
            else:
                dataset = gdal.Translate(output, source, **options)
            if dataset is None:
                raise RuntimeError(f"Could not resample {source} to {scale}%.")
            dataset = None  # Close and flush to disk.
        except Exception:
            if os.path.exists(output):
                os.remove(output)
            raise
        return output

    def segment_by_text_tiled(self):
        """Segment a large GeoTIFF with the text prompt, tile by tile.

        Uses ``SamGeo3.generate_masks_tiled``, which writes the merged mask
        raster directly, so the result is saved right away with the Output
        tab settings instead of being kept for **Save Masks**.
        """
        if not hasattr(self.sam, "generate_masks_tiled"):
            self.show_error(
                "Tiled mode requires the SamGeo3 (SAM3) model and "
                "segment-geospatial 1.2 or later."
            )
            return

        prompt = self.text_prompt_edit.text().strip()
        if not prompt:
            self.show_error("Please enter a text prompt.")
            return

        source = self._tiled_source_path()
        if source is None:
            self.show_error(
                "Please set an image, or select a raster layer or image file "
                "on the Model tab."
            )
            return
        if not is_geotiff(source):
            self.show_error("Tiled mode needs a GeoTIFF (.tif or .tiff) image.")
            return

        tile_size = self.tile_size_spin.value()
        overlap = self.tile_overlap_spin.value()
        scale = self.tile_scale_spin.value()
        error = validate_tile_settings(tile_size, overlap, scale)
        if error:
            self.show_error(error)
            return

        import tempfile

        min_size = self.min_size_spin.value()
        max_size = self.max_size_spin.value() or None
        format_text = self.output_format_combo.currentText()
        unique = self.unique_check.isChecked()
        is_raster = "Raster" in format_text
        image_was_set = self.current_image_path is not None
        tiles_started = False
        temp_files = []

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.text_status_label.setText("Processing tiles...")
            self.text_status_label.setStyleSheet("color: orange;")
            QCoreApplication.processEvents()

            work_source = source
            n_tiles = None
            source_layer = QgsRasterLayer(source, "samgeo_tiled_source")
            if source_layer.isValid():
                width, height = scaled_size(
                    source_layer.width(), source_layer.height(), scale
                )
                n_tiles = tile_count(width, height, tile_size, overlap)
            if scale != 100:
                self.text_status_label.setText(f"Resampling image to {scale}%...")
                QCoreApplication.processEvents()
                work_source = self._resample_for_tiling(source, scale)
                temp_files.append(work_source)

            if n_tiles is not None:
                self.text_status_label.setText(f"Processing {n_tiles} tile(s)...")
                QCoreApplication.processEvents()

            output_path, use_temp_file = self._resolve_output_path(
                format_text, source, PROMPT_TEXT
            )
            if is_raster:
                raster_path = output_path
            else:
                temp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
                raster_path = temp_file.name
                temp_file.close()
                temp_files.append(raster_path)

            tiles_started = True
            # generate_masks_tiled skips tiles that raise and only reports
            # them in its verbose output, so capture that output (it also
            # keeps tqdm off QGIS's stderr) and look for failed tiles.
            import contextlib
            import io

            log = io.StringIO()
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                self.sam.generate_masks_tiled(
                    source=work_source,
                    prompt=prompt,
                    output=raster_path,
                    tile_size=tile_size,
                    overlap=overlap,
                    min_size=min_size,
                    max_size=max_size,
                    unique=unique,
                    verbose=True,
                )
            failures = failed_tile_messages(log.getvalue())
            for message in failures:
                self.log_message(message, level=Qgis.MessageLevel.Warning)
            self.last_prompt_type = PROMPT_TEXT
            self.mask_source_path = source

            if not is_raster:
                try:
                    from ._samgeo_lib import get_samgeo
                except ImportError:
                    from _samgeo_lib import get_samgeo

                get_samgeo().common.raster_to_vector(raster_path, output_path)
                # Tiled masks have no per-mask scores.
                self._attach_mask_attributes(output_path, None, source)
                result_layer = QgsVectorLayer(
                    output_path,
                    "samgeo_masks" if use_temp_file else os.path.basename(output_path),
                    "ogr",
                )
                num_objects = (
                    result_layer.featureCount() if result_layer.isValid() else 0
                )
            else:
                result_layer = QgsRasterLayer(
                    output_path,
                    "samgeo_masks" if use_temp_file else os.path.basename(output_path),
                )
                objects = getattr(self.sam, "objects", None)
                num_objects = None
                if objects is not None and unique:
                    import numpy as np

                    num_objects = int(np.count_nonzero(np.unique(objects)))

            if self.add_to_map_check.isChecked() and result_layer.isValid():
                QgsProject.instance().addMapLayer(result_layer)

            found = "unknown" if num_objects is None else num_objects
            lines = [
                "Tiled Text Segmentation Results:",
                f"Prompt: {prompt}",
                f"Image: {os.path.basename(source)}",
                f"Tile size: {tile_size}, overlap: {overlap}, scale: {scale}%",
            ]
            if n_tiles is not None:
                lines.append(f"Tiles: {n_tiles}")
            if failures:
                lines.append(
                    f"Failed tiles: {len(failures)} (skipped; see the SamGeo log)"
                )
            if not is_raster:
                lines.append(f"Polygons: {found}")
            else:
                lines.append(f"Objects found: {found}")
            lines.append(f"Saved to: {output_path}")
            self.results_text.setText("\n".join(lines) + "\n")

            if failures:
                self.text_status_label.setText(
                    f"Incomplete: {len(failures)} tile(s) failed. Saved to "
                    f"{os.path.basename(output_path)}"
                )
                self.text_status_label.setStyleSheet("color: orange;")
            else:
                self.text_status_label.setText(
                    f"Done. Saved to {os.path.basename(output_path)}"
                )
                self.text_status_label.setStyleSheet("color: green;")
            self.log_message(f"Tiled text segmentation saved to: {output_path}")

        except Exception as e:
            self.text_status_label.setText("Segmentation failed!")
            self.text_status_label.setStyleSheet("color: red;")
            self.show_error(f"Tiled segmentation failed: {str(e)}")

        finally:
            for path in temp_files:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass
            # generate_masks_tiled replaces the model's image with the last
            # tile, so point/box prompts need the image to be set again.
            if image_was_set and tiles_started:
                self.image_needs_reset = True
                self.image_status.setText(
                    "Image: set it again (tiled mode replaced it)"
                )
                self.image_status.setStyleSheet("color: orange;")
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
            item.setForeground(
                Qt.GlobalColor.green if foreground else Qt.GlobalColor.red
            )
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
            item.setForeground(Qt.GlobalColor.green)
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

        if not self._image_ready():
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
            self._record_mask_result(PROMPT_POINT)

            num_masks = len(self.sam.masks) if self.sam.masks else 0
            self.results_text.setText(
                f"Point Segmentation Results:\n"
                f"Points used: {len(self.point_coords)}\n"
                f"Objects found: {num_masks}\n"
            )

            # Update status label
            if num_masks > 0:
                if self.auto_show_check.isChecked():
                    self.point_status_label.setText(f"Found {num_masks} object(s).")
                else:
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

            # Auto-show results if enabled
            if num_masks > 0:
                self._auto_show_results()

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

        if not self._image_ready():
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
            self._record_mask_result(PROMPT_BOX)

            num_masks = len(self.sam.masks) if self.sam.masks else 0
            self.results_text.setText(
                f"Box Segmentation Results:\n"
                f"Box: {self.box_coords}\n"
                f"Objects found: {num_masks}\n"
            )

            # Update status label
            if num_masks > 0:
                if self.auto_show_check.isChecked():
                    self.box_status_label.setText(f"Found {num_masks} object(s).")
                else:
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

            # Auto-show results if enabled
            if num_masks > 0:
                self._auto_show_results()

        except Exception as e:
            self.box_status_label.setText("Segmentation failed!")
            self.box_status_label.setStyleSheet("color: red;")
            self.show_error(f"Segmentation failed: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def _point_to_pixel(self, point):
        """Convert a map-coordinate point in the raster layer CRS to pixel coordinates."""
        if self.current_layer is None:
            raise ValueError("Please set an image first.")

        extent = self.current_layer.extent()
        width = self.current_layer.width()
        height = self.current_layer.height()

        px = (point.x() - extent.xMinimum()) / extent.width() * width
        py = (extent.yMaximum() - point.y()) / extent.height() * height
        return [px, py]

    def _vector_layer_to_pixel_points(self, layer, source_crs=None):
        """Extract point features from a vector layer as raster pixel coordinates.

        SamGeo's batch method can transform vector/map coordinates to pixel
        coordinates with rasterio/PROJ, but QGIS plugin environments sometimes
        lack a rasterio PROJ database context. Converting through QGIS first
        avoids that dependency and passes pixel coordinates directly to SamGeo.
        """
        if self.current_layer is None:
            raise ValueError("Please set an image first.")

        raster_crs = self.current_layer.crs()
        vector_crs = source_crs or layer.crs()

        if not vector_crs.isValid():
            raise ValueError(
                "Point layer CRS is unknown. Please enter a valid CRS, e.g., EPSG:4326."
            )

        transform = None
        if raster_crs.isValid() and vector_crs.isValid() and vector_crs != raster_crs:
            transform = QgsCoordinateTransform(
                vector_crs, raster_crs, QgsProject.instance()
            )

        pixel_points = []
        skipped = 0
        for feature in layer.getFeatures():
            geom = feature.geometry()
            if geom is None or geom.isEmpty():
                skipped += 1
                continue

            points = []
            if QgsWkbTypes.isMultiType(geom.wkbType()):
                points = geom.asMultiPoint()
            else:
                points = [geom.asPoint()]

            for point in points:
                try:
                    if transform is not None:
                        point = transform.transform(point)
                    px, py = self._point_to_pixel(point)
                    if (
                        0 <= px < self.current_layer.width()
                        and 0 <= py < self.current_layer.height()
                    ):
                        pixel_points.append([px, py])
                    else:
                        skipped += 1
                except Exception as exc:
                    skipped += 1
                    self.log_message(
                        f"Skipped point feature {feature.id()}: {exc}",
                        level=Qgis.MessageLevel.Warning,
                    )

        if not pixel_points:
            raise ValueError(
                "No valid point features fall within the current image extent. "
                "Check the point layer CRS and image extent."
            )

        if skipped:
            self.log_message(
                f"Skipped {skipped} point(s) outside the image extent or with invalid geometry.",
                level=Qgis.MessageLevel.Warning,
            )

        return pixel_points

    def segment_by_points_batch(self):
        """Segment using batch point prompts from interactive points or vector file/layer."""
        if self.sam is None:
            self.show_error("Please load the model first.")
            return

        if not self._image_ready():
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
            # Convert file/layer point coordinates to raster pixel coordinates using
            # QGIS CRS transforms, then pass pixel coordinates directly to SamGeo.
            # This avoids rasterio/PROJ failures in QGIS Python environments.
            crs_text = self.point_crs_edit.text().strip()
            source_crs = None
            if crs_text:
                source_crs = QgsCoordinateReferenceSystem(crs_text)
                if not source_crs.isValid():
                    self.show_error(f"Invalid point CRS: {crs_text}")
                    return

            layer = None
            layer_id = self.vector_layer_combo.currentData()
            if layer_id:
                layer = QgsProject.instance().mapLayer(layer_id)
                if layer and layer.isValid():
                    source_description = layer.name()
                else:
                    layer = None

            if layer is None:
                vector_file = self.vector_file_edit.text().strip()
                if vector_file and os.path.exists(vector_file):
                    layer = QgsVectorLayer(
                        vector_file, os.path.basename(vector_file), "ogr"
                    )
                    if not layer.isValid():
                        self.show_error(f"Could not load vector file: {vector_file}")
                        return
                    source_description = os.path.basename(vector_file)
                else:
                    self.show_error(
                        "Please add points interactively or select a vector layer/file."
                    )
                    return

            try:
                point_source = self._vector_layer_to_pixel_points(layer, source_crs)
                point_crs = None  # Converted to pixel coordinates already.
            except Exception as exc:
                self.show_error(f"Could not prepare point layer: {exc}")
                return

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
            self._record_mask_result(PROMPT_BATCH_POINT)

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
                if self.auto_show_check.isChecked():
                    self.batch_status_label.setText(f"Found {num_masks} object(s).")
                else:
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

            # Add output to map if saved and option is checked (for batch-specific output)
            if output_path and self.add_to_map_check.isChecked():
                layer = QgsRasterLayer(output_path, os.path.basename(output_path))
                if layer.isValid():
                    QgsProject.instance().addMapLayer(layer)
                    self.results_text.append("Added result layer to map.")

            # Auto-show results if enabled
            if num_masks > 0:
                self._auto_show_results()

        except Exception as e:
            self.batch_status_label.setText("Failed!")
            self.batch_status_label.setStyleSheet("color: red;")
            self.show_error(f"Batch segmentation failed: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def _auto_show_results(self):
        """Automatically save and show results based on Output tab settings.

        This method is called after successful segmentation to automatically
        display results without requiring the user to manually click on the
        Output tab and Save Masks button.
        """
        if not self.auto_show_check.isChecked():
            return

        if self.sam is None or self.sam.masks is None or len(self.sam.masks) == 0:
            return

        try:
            output_path = self._write_masks()
            self.results_text.append(f"\nAuto-saved to: {output_path}")
            self.log_message(f"Auto-saved masks to: {output_path}")

        except Exception as e:
            self.log_message(
                f"Auto-show failed: {str(e)}", level=Qgis.MessageLevel.Warning
            )
            self.show_error(f"Auto-show failed: {str(e)}")

    def _resolve_output_path(self, format_text, source_path=None, prompt_type=None):
        """Return the output path for the masks and whether it is temporary.

        Uses the Output tab path when set. Otherwise, if "Save next to source
        image" is checked and the source folder is writable, suggests a new
        file there; else falls back to a temporary file.

        Args:
            format_text: Text of the output format combo.
            source_path: Source raster path. Defaults to the current image.
            prompt_type: Prompt type for the suggested file name. Defaults
                to the prompt type of the current masks.

        Returns:
            A ``(output_path, use_temp_file)`` tuple.
        """
        import tempfile

        output_path = self.output_path_edit.text().strip()
        if output_path:
            return output_path, False

        extension = output_extension(format_text)
        if self.save_to_source_dir_check.isChecked():
            suggested = default_output_path(
                source_path or self.mask_source_path or self.current_image_path,
                prompt_type or self.last_prompt_type,
                extension,
            )
            if suggested and self._dir_is_writable(os.path.dirname(suggested)):
                return suggested, False
            self.log_message(
                "Source image folder is unknown or not writable; "
                "saving to a temporary file instead.",
                level=Qgis.MessageLevel.Warning,
            )

        if extension == ".shp":
            temp_dir = tempfile.mkdtemp()
            return os.path.join(temp_dir, "masks.shp"), True
        temp_file = tempfile.NamedTemporaryFile(suffix=extension, delete=False)
        output_path = temp_file.name
        temp_file.close()
        return output_path, True

    @staticmethod
    def _dir_is_writable(directory):
        """Return True if a file can actually be created in a directory.

        ``os.access`` is unreliable for this on Windows, where it ignores
        ACLs, so this creates and removes a temporary file instead.

        Args:
            directory: Directory to test.

        Returns:
            True if a file could be created there.
        """
        import tempfile

        try:
            with tempfile.TemporaryFile(dir=directory):
                pass
        except OSError:
            return False
        return True

    def _record_mask_result(self, prompt_type):
        """Remember how the current masks were made.

        Called right after a segmentation call returns, so the ``prompt`` and
        ``source`` fields describe the masks actually being saved.

        Args:
            prompt_type: Prompt type, e.g. ``"text"`` or ``"box"``.
        """
        self.last_prompt_type = prompt_type
        self.mask_source_path = self.current_image_path

    def _clear_mask_result(self):
        """Drop masks left over from the previous image.

        ``set_image`` does not clear the model's masks, but it does change
        the georeference that ``save_masks`` writes them with, so saving them
        after a new image is set would misplace and mislabel them.
        """
        if self.sam is not None and hasattr(self.sam, "masks"):
            self.sam.masks = None
        self.last_prompt_type = None
        self.mask_source_path = None

    def _write_masks(self):
        """Write the current masks using the Output tab settings.

        Rasters are written directly. Vectors are written as a temporary
        raster first and then polygonized, after which the ``source``,
        ``prompt`` and ``score`` attributes are added.

        Returns:
            The path of the written file.
        """
        import tempfile

        format_text = self.output_format_combo.currentText()
        output_path, use_temp_file = self._resolve_output_path(format_text)
        unique = self.unique_check.isChecked()
        layer_name = "samgeo_masks" if use_temp_file else os.path.basename(output_path)

        if "Raster" in format_text:
            self.sam.save_masks(output=output_path, unique=unique)

            if self.add_to_map_check.isChecked():
                layer = QgsRasterLayer(output_path, layer_name)
                if layer.isValid():
                    QgsProject.instance().addMapLayer(layer)
            return output_path

        # Save as vector - first save as raster, then convert
        temp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        temp_raster = temp_file.name
        temp_file.close()
        try:
            self.sam.save_masks(output=temp_raster, unique=unique)

            try:
                from ._samgeo_lib import get_samgeo
            except ImportError:
                from _samgeo_lib import get_samgeo

            common = get_samgeo().common

            common.raster_to_vector(temp_raster, output_path)
            self._attach_mask_attributes(
                output_path, getattr(self.sam, "mask_scores", None)
            )

            if self.add_to_map_check.isChecked():
                layer = QgsVectorLayer(output_path, layer_name, "ogr")
                if layer.isValid():
                    QgsProject.instance().addMapLayer(layer)
        finally:
            if os.path.exists(temp_raster):
                os.remove(temp_raster)
        return output_path

    def _attach_mask_attributes(self, output_path, scores=None, source_path=None):
        """Write per-mask attributes into the saved vector layer.

        Every feature gets ``source`` (the source raster's file name) and
        ``prompt`` (the prompt type, e.g. ``text``, ``point`` or ``box``).

        ``save_masks`` records ``mask_scores`` (raster value -> confidence) on
        the model (segment-geospatial >= 1.4.2), and ``raster_to_vector``
        keeps the raster value in a ``value`` field, so the two are joined
        here as a ``score`` attribute. Without scores (older
        segment-geospatial versions) no ``score`` field is added.

        Args:
            output_path: Path of the vector file written by raster_to_vector.
            scores: Optional mapping of raster value to confidence score.
            source_path: Source raster path. Defaults to the current image.
        """
        layer = QgsVectorLayer(output_path, "samgeo_attributes", "ogr")
        if not layer.isValid():
            return
        provider = layer.dataProvider()

        attrs = mask_attributes(
            source_path or self.mask_source_path or self.current_image_path,
            self.last_prompt_type,
        )
        new_fields = [
            QgsField(name, QVariant.String, "", 254)
            for name in (SOURCE_FIELD, PROMPT_FIELD)
            if layer.fields().indexOf(name) < 0
        ]
        value_idx = layer.fields().indexOf("value")
        if scores and value_idx >= 0 and layer.fields().indexOf("score") < 0:
            new_fields.append(QgsField("score", QVariant.Double))
        if new_fields:
            if not provider.addAttributes(new_fields):
                self.log_message(
                    f"Could not add attribute fields to {output_path}",
                    level=Qgis.MessageLevel.Warning,
                )
                return
            layer.updateFields()

        fields = layer.fields()
        fixed = {fields.indexOf(name): value for name, value in attrs.items()}
        score_idx = fields.indexOf("score") if scores and value_idx >= 0 else -1

        changes = {}
        for feature in layer.getFeatures():
            values = dict(fixed)
            if score_idx >= 0:
                try:
                    key = int(feature[value_idx])
                except (TypeError, ValueError):
                    key = None
                if key in scores:
                    values[score_idx] = float(scores[key])
            changes[feature.id()] = values
        if changes:
            provider.changeAttributeValues(changes)

    def save_masks(self):
        """Save the segmentation masks."""
        if self.sam is None or self.sam.masks is None or len(self.sam.masks) == 0:
            self.show_error("No masks to save. Please run segmentation first.")
            return

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            QCoreApplication.processEvents()

            output_path = self._write_masks()
            self.results_text.append(f"\nSaved to: {output_path}")
            self.log_message(f"Masks saved to: {output_path}")

        except Exception as e:
            self.show_error(f"Failed to save masks: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def show_error(self, message):
        """Show an error message."""
        QMessageBox.critical(self.iface.mainWindow(), "SamGeo Error", message)
        self.log_message(message, level=Qgis.MessageLevel.Critical)

    def log_message(self, message, level=Qgis.MessageLevel.Info):
        """Log a message to QGIS."""
        QgsMessageLog.logMessage(message, "SamGeo", level)

    def clear_gpu_memory(self):
        """Clear GPU memory and release CUDA resources."""
        import gc

        cleared_items = []

        # Import torch early to use for cleanup
        torch = None
        try:
            import torch as _torch

            torch = _torch
        except ImportError:
            # PyTorch is optional; continue without GPU memory clearing if not installed.
            pass

        # Clear SamGeo model if loaded
        if self.sam is not None:
            try:
                sam_obj = self.sam
                # Clear the model
                if hasattr(sam_obj, "model") and sam_obj.model is not None:
                    try:
                        sam_obj.model.cpu()
                    except Exception:  # nosec B110
                        pass
                    try:
                        for param in sam_obj.model.parameters():
                            param.data = None
                            if param.grad is not None:
                                param.grad = None
                    except Exception:  # nosec B110
                        pass
                    del sam_obj.model
                    sam_obj.model = None

                # Clear any other attributes
                for attr in list(vars(sam_obj).keys()):
                    try:
                        setattr(sam_obj, attr, None)
                    except Exception:  # nosec B110
                        # Ignore errors when clearing attributes; some may be read-only or protected.
                        pass

                # Delete the sam object
                self.sam = None
                del sam_obj
                cleared_items.append("SamGeo model")

                # Update UI status
                if hasattr(self, "model_status") and self.model_status is not None:
                    self.model_status.setText("Model: Not loaded")
                    self.model_status.setStyleSheet("color: gray;")
                if hasattr(self, "image_status") and self.image_status is not None:
                    self.image_status.setText("Image: Not set")
                    self.image_status.setStyleSheet("color: gray;")

                # Also clear internal state for image and layer
                self.current_image_path = None
                self.current_layer = None

            except Exception:  # nosec B110
                pass

        # Run garbage collection multiple times
        for _ in range(5):
            gc.collect()

        # Clear PyTorch CUDA cache
        if torch is not None and torch.cuda.is_available():
            try:
                # Synchronize first
                torch.cuda.synchronize()
                # Empty cache
                torch.cuda.empty_cache()
                # IPC collect if available
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
                # Run gc and empty cache again
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                cleared_items.append("CUDA cache")

                # Get memory info for display
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                memory_info = f"\n\nGPU Memory:\n  Allocated: {allocated:.1f} MB\n  Reserved: {reserved:.1f} MB"

                if allocated > 100:  # More than 100MB still allocated
                    memory_info += "\n\nNote: Some GPU memory may still be held by PyTorch's memory allocator. Restart QGIS to fully release all GPU memory."
            except Exception as e:
                memory_info = f"\n\nError clearing CUDA: {str(e)}"
        elif torch is None:
            memory_info = "\n\nPyTorch not installed."
        else:
            memory_info = "\n\nNo CUDA GPU available."

        if cleared_items:
            message = f"Cleared: {', '.join(cleared_items)}{memory_info}"
        else:
            message = f"No models loaded to clear.{memory_info}"

        self.iface.statusBarIface().showMessage("GPU memory cleared", 3000)
        QMessageBox.information(
            self.iface.mainWindow(),
            "Clear GPU Memory",
            message,
        )

    def show_update_checker(self):
        """Display the update checker dialog."""
        try:
            from .update_checker import UpdateCheckerDialog
        except ImportError:
            try:
                from update_checker import UpdateCheckerDialog
            except ImportError as e:
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Error",
                    f"Failed to import update checker dialog:\n{str(e)}",
                )
                return

        try:
            dialog = UpdateCheckerDialog(self.plugin_dir, self.iface.mainWindow())
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(
                self.iface.mainWindow(),
                "Error",
                f"Failed to open update checker:\n{str(e)}",
            )

    def show_about(self):
        """Display the about dialog."""
        import re

        # Read version from metadata.txt
        version = "Unknown"
        try:
            metadata_path = os.path.join(self.plugin_dir, "metadata.txt")
            with open(metadata_path, "r", encoding="utf-8") as f:
                content = f.read()
                version_match = re.search(r"^version=(.+)$", content, re.MULTILINE)
                if version_match:
                    version = version_match.group(1).strip()
        except Exception as e:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "SamGeo Plugin",
                f"Could not read version from metadata.txt:\n{str(e)}",
            )

        about_text = f"""
<h2>SamGeo Plugin for QGIS</h2>
<p>Version: {version}</p>
<p>Author: Qiusheng Wu</p>

<h3>Features:</h3>
<ul>
<li><b>Text-based Segmentation:</b> Describe objects to segment using natural language</li>
<li><b>Interactive Point Prompts:</b> Click on map to segment objects with foreground/background points</li>
<li><b>Box Prompts:</b> Draw rectangles to segment regions of interest</li>
<li><b>Batch Point Mode:</b> Process multiple points for batch segmentation</li>
<li><b>Multiple SAM Models:</b> Support for SAM1, SAM2, and SAM3 models</li>
</ul>

<h3>Links:</h3>
<ul>
<li><a href="https://samgeo.gishub.org">Documentation</a></li>
<li><a href="https://github.com/opengeos/qgis-samgeo-plugin">GitHub Repository</a></li>
<li><a href="https://github.com/opengeos/qgis-samgeo-plugin/issues">Report Issues</a></li>
</ul>

<p>Licensed under MIT License</p>
"""
        QMessageBox.about(
            self.iface.mainWindow(),
            "About SamGeo Plugin",
            about_text,
        )
