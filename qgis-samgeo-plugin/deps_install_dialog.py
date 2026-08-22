"""Dependency installation dock widget for SamGeo QGIS Plugin.

Provides a user-facing panel for installing AI dependencies on first use.
Shows GPU detection info, progress bars, and status messages during
the installation process.
"""

import os
import platform
import sys

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QDockWidget,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class DepsInstallDockWidget(QDockWidget):
    """Dock widget for managing dependency installation.

    Signals:
        install_requested: Emitted when user clicks Install Dependencies.
        cancel_requested: Emitted when user clicks Cancel.
    """

    install_requested = pyqtSignal()
    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize the dock widget.

        Args:
            parent: Optional parent widget.
        """
        super().__init__("SamGeo - Setup", parent)
        self.setObjectName("SamGeoDepsInstallDock")
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Welcome section
        welcome_group = QGroupBox("Welcome")
        welcome_layout = QVBoxLayout(welcome_group)

        welcome_text = QLabel(
            "SamGeo needs to install AI dependencies before you can use it.\n\n"
            "This is a one-time setup that will:\n"
            "  \u2022 Download a Python runtime (~50 MB)\n"
            "  \u2022 Download a fast package installer (~15 MB)\n"
            "  \u2022 Install AI packages (~1\u20133 GB)"
        )
        welcome_text.setWordWrap(True)
        welcome_layout.addWidget(welcome_text)

        from .core.venv_manager import CACHE_DIR

        _home = os.path.expanduser("~")
        _display = (
            ("~" + CACHE_DIR[len(_home) :])
            if CACHE_DIR.startswith(_home)
            else CACHE_DIR
        )
        location_label = QLabel(
            f"<small>Installation location: <code>{_display}</code></small>"
        )
        location_label.setWordWrap(True)
        welcome_layout.addWidget(location_label)

        layout.addWidget(welcome_group)

        # GPU info section
        self.gpu_group = QGroupBox("GPU Detection")
        gpu_layout = QVBoxLayout(self.gpu_group)

        self.gpu_label = QLabel("Detecting GPU...")
        self.gpu_label.setWordWrap(True)
        gpu_layout.addWidget(self.gpu_label)

        layout.addWidget(self.gpu_group)

        # Action buttons
        button_layout = QHBoxLayout()

        self.install_button = QPushButton("Install Dependencies")
        self.install_button.setMinimumHeight(36)
        self.install_button.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 13px; }"
        )
        self.install_button.clicked.connect(self.install_requested.emit)
        button_layout.addWidget(self.install_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setMinimumHeight(36)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        self.cancel_button.hide()
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        # Progress section
        self.progress_frame = QFrame()
        progress_layout = QVBoxLayout(self.progress_frame)
        progress_layout.setContentsMargins(0, 0, 0, 0)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        progress_layout.addWidget(self.progress_label)

        self.progress_frame.hide()
        layout.addWidget(self.progress_frame)

        # Status section
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.hide()
        layout.addWidget(self.status_label)

        # Reinstall button (shown after errors)
        self.reinstall_button = QPushButton("Reinstall Dependencies")
        self.reinstall_button.setMinimumHeight(36)
        self.reinstall_button.clicked.connect(self._on_reinstall_clicked)
        self.reinstall_button.hide()
        layout.addWidget(self.reinstall_button)

        # Help links
        help_label = QLabel(
            "<small>"
            "If you encounter issues, please check the "
            '<a href="https://samgeo.gishub.org">documentation</a> '
            "or report a bug on "
            '<a href="https://github.com/opengeos/qgis-samgeo-plugin/issues">'
            "GitHub</a>."
            "</small>"
        )
        help_label.setWordWrap(True)
        help_label.setOpenExternalLinks(True)
        layout.addWidget(help_label)

        layout.addStretch()
        self.setWidget(container)

        # Detect GPU on creation
        self._detect_gpu()

    def _detect_gpu(self):
        """Detect GPU and update the GPU info label."""
        try:
            from .core.venv_manager import detect_nvidia_gpu

            has_gpu, gpu_info = detect_nvidia_gpu()
            if has_gpu:
                name = gpu_info.get("name", "Unknown GPU")
                memory_mb = gpu_info.get("memory_mb")
                memory_str = f" ({memory_mb} MB)" if memory_mb else ""
                self.gpu_label.setText(
                    f"GPU Detected: {name}{memory_str}\n"
                    "CUDA acceleration will be enabled."
                )
                self.gpu_group.setStyleSheet(
                    "QGroupBox { border: 1px solid #4CAF50; "
                    "border-radius: 4px; padding: 8px; margin-top: 8px; }"
                )
            else:
                if sys.platform == "darwin":
                    machine = platform.machine().lower()
                    if machine in ("arm64", "aarch64"):
                        msg = (
                            "No NVIDIA GPU detected (CUDA not applicable on "
                            "macOS).\n"
                            "Apple Silicon MPS may still be used at runtime if "
                            "supported by PyTorch/QGIS. CPU fallback is "
                            "available."
                        )
                    else:
                        msg = (
                            "No NVIDIA GPU detected (CUDA unavailable on this "
                            "system).\n"
                            "CPU mode will be used. MPS is only available on "
                            "Apple Silicon."
                        )
                else:
                    msg = (
                        "No NVIDIA GPU detected.\n"
                        "CPU mode will be used (slower but functional)."
                    )
                self.gpu_label.setText(msg)
                self.gpu_group.setStyleSheet(
                    "QGroupBox { border: 1px solid #888; "
                    "border-radius: 4px; padding: 8px; margin-top: 8px; }"
                )
        except Exception:
            if sys.platform == "darwin":
                self.gpu_label.setText(
                    "Could not detect NVIDIA GPU.\n"
                    "CUDA will not be enabled. Apple Silicon MPS may still be "
                    "used at runtime if available."
                )
            else:
                self.gpu_label.setText("Could not detect GPU.\nCPU mode will be used.")

    def _on_reinstall_clicked(self):
        """Handle reinstall button click by removing existing venv first."""
        try:
            from .core.venv_manager import remove_venv

            remove_venv()
        except Exception:  # nosec B110
            pass
        self.reinstall_button.hide()
        self.install_requested.emit()

    def show_install_ui(self):
        """Show the install button, hide progress and status."""
        self.install_button.show()
        self.cancel_button.hide()
        self.progress_frame.hide()
        self.status_label.hide()
        self.reinstall_button.hide()

    def show_progress_ui(self):
        """Show progress bar and cancel button, hide install button."""
        self.install_button.hide()
        self.cancel_button.show()
        self.progress_frame.show()
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting installation...")
        self.status_label.hide()
        self.reinstall_button.hide()

    def set_progress(self, percent: int, message: str):
        """Update the progress bar and label.

        Args:
            percent: Progress percentage (0-100).
            message: Status message to display.
        """
        self.progress_bar.setValue(percent)
        self.progress_label.setText(message)

    def set_status(self, ok: bool, message: str):
        """Set the status display.

        Args:
            ok: Whether the status is a success.
            message: The status message to display.
        """
        self.status_label.setText(message)
        self.status_label.show()
        if ok:
            self.status_label.setStyleSheet(
                "QLabel { color: #4CAF50; font-weight: bold; }"
            )
        else:
            self.status_label.setStyleSheet(
                "QLabel { color: #F44336; font-weight: bold; }"
            )

    def show_complete_ui(self, success: bool, message: str):
        """Show the completion status.

        Args:
            success: Whether installation was successful.
            message: Completion message.
        """
        self.cancel_button.hide()

        if success:
            self.progress_bar.setValue(100)
            self.progress_label.setText("Installation complete!")
            self.set_status(True, message)
            self.install_button.hide()
            self.reinstall_button.hide()
        else:
            self.progress_frame.hide()
            self.set_status(False, message)
            self.install_button.hide()
            self.reinstall_button.show()
