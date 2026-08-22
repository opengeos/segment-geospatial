"""Background worker thread for installing SamGeo dependencies.

Runs the full installation pipeline (download Python, create venv,
install packages) in a separate thread to keep the QGIS UI responsive.
"""

import traceback

from qgis.PyQt.QtCore import QThread, pyqtSignal


class DepsInstallWorker(QThread):
    """Worker thread that installs dependencies in the background.

    Signals:
        progress(int, str): Emitted with (percent, message) during install.
        finished(bool, str): Emitted with (success, message) when done.
    """

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, cuda_enabled: bool = False, parent=None):
        """Initialize the worker.

        Args:
            cuda_enabled: Whether to install CUDA-enabled PyTorch.
            parent: Optional parent QObject.
        """
        super().__init__(parent)
        self._cancelled = False
        self._cuda_enabled = cuda_enabled

    def cancel(self):
        """Request cancellation of the installation."""
        self._cancelled = True

    def run(self):
        """Run the installation pipeline."""
        try:
            from ..core.venv_manager import create_venv_and_install

            success, message = create_venv_and_install(
                progress_callback=lambda p, m: self.progress.emit(p, m),
                cancel_check=lambda: self._cancelled,
                cuda_enabled=self._cuda_enabled,
            )
            self.finished.emit(success, message)
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.finished.emit(False, error_msg)
