"""Python Standalone Manager for SamGeo QGIS Plugin.

Downloads and manages a standalone Python interpreter that matches
the QGIS Python version, ensuring compatibility.

Adapted from the GeoAI QGIS plugin's python_manager.py.
"""

import os
import platform
import shutil
import subprocess  # nosec B404
import sys
import tarfile
import tempfile
import zipfile
from typing import Callable, Optional, Tuple

from qgis.core import Qgis, QgsBlockingNetworkRequest, QgsMessageLog
from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtNetwork import QNetworkRequest

CACHE_DIR = (
    os.environ.get("SAMGEO_CACHE_DIR")
    or os.environ.get("SAMGEO_VENV_DIR")
    or os.path.expanduser("~/.qgis_samgeo")
)
STANDALONE_DIR = os.path.join(CACHE_DIR, "python_standalone")

# Release tag from python-build-standalone
RELEASE_TAG = "20241219"

# Mapping of Python minor versions to their latest patch versions in the release
PYTHON_VERSIONS = {
    (3, 9): "3.9.21",
    (3, 10): "3.10.16",
    (3, 11): "3.11.11",
    (3, 12): "3.12.8",
    (3, 13): "3.13.1",
}


def _log(message: str, level=Qgis.MessageLevel.Info):
    """Log a message to the QGIS message log.

    Args:
        message: The message to log.
        level: The log level (default: Qgis.MessageLevel.Info).
    """
    QgsMessageLog.logMessage(message, "SamGeo", level=level)


def _safe_extract_tar(tar: tarfile.TarFile, dest_dir: str) -> None:
    """Safely extract tar archive with path traversal protection.

    Args:
        tar: The tarfile object to extract.
        dest_dir: The destination directory for extraction.

    Raises:
        ValueError: If path traversal is detected.
    """
    dest_dir = os.path.realpath(dest_dir)
    use_filter = sys.version_info >= (3, 12)
    for member in tar.getmembers():
        member_path = os.path.realpath(os.path.join(dest_dir, member.name))
        if not member_path.startswith(dest_dir + os.sep) and member_path != dest_dir:
            raise ValueError(f"Attempted path traversal in tar archive: {member.name}")
        if not use_filter and (member.issym() or member.islnk()):
            raise ValueError(f"Refusing symlink/hardlink in tar archive: {member.name}")
        if use_filter:
            tar.extract(member, dest_dir, filter="data")
        else:
            tar.extract(member, dest_dir)


def _safe_extract_zip(zip_file: zipfile.ZipFile, dest_dir: str) -> None:
    """Safely extract zip archive with path traversal protection.

    Args:
        zip_file: The zipfile object to extract.
        dest_dir: The destination directory for extraction.

    Raises:
        ValueError: If path traversal is detected.
    """
    dest_dir = os.path.realpath(dest_dir)
    for member in zip_file.namelist():
        member_path = os.path.realpath(os.path.join(dest_dir, member))
        if not member_path.startswith(dest_dir + os.sep) and member_path != dest_dir:
            raise ValueError(f"Attempted path traversal in zip archive: {member}")
        zip_file.extract(member, dest_dir)


def get_qgis_python_version() -> Tuple[int, int]:
    """Get the Python version used by QGIS.

    Returns:
        Tuple of (major, minor) version numbers.
    """
    return (sys.version_info.major, sys.version_info.minor)


def get_python_full_version() -> str:
    """Get the full Python version string for download.

    Returns:
        Full Python version string (e.g., '3.12.8').
    """
    version_tuple = get_qgis_python_version()
    if version_tuple in PYTHON_VERSIONS:
        return PYTHON_VERSIONS[version_tuple]
    return f"{version_tuple[0]}.{version_tuple[1]}.0"


def get_standalone_dir() -> str:
    """Get the directory where Python standalone is installed.

    Returns:
        Path to the standalone Python directory.
    """
    return STANDALONE_DIR


def get_standalone_python_path() -> str:
    """Get the path to the standalone Python executable.

    Returns:
        Path to the Python executable.
    """
    python_dir = os.path.join(STANDALONE_DIR, "python")
    if sys.platform == "win32":
        return os.path.join(python_dir, "python.exe")
    else:
        return os.path.join(python_dir, "bin", "python3")


def standalone_python_exists() -> bool:
    """Check if standalone Python is already installed.

    Returns:
        True if the standalone Python executable exists.
    """
    python_path = get_standalone_python_path()
    return os.path.exists(python_path)


def _get_platform_info() -> Tuple[str, str]:
    """Get platform and architecture info for download URL.

    Returns:
        Tuple of (platform_string, file_extension).
    """
    system = sys.platform
    machine = platform.machine().lower()

    if system == "darwin":
        if machine in ("arm64", "aarch64"):
            return ("aarch64-apple-darwin", ".tar.gz")
        else:
            return ("x86_64-apple-darwin", ".tar.gz")
    elif system == "win32":
        return ("x86_64-pc-windows-msvc", ".tar.gz")
    else:  # Linux
        if machine in ("arm64", "aarch64"):
            return ("aarch64-unknown-linux-gnu", ".tar.gz")
        else:
            return ("x86_64-unknown-linux-gnu", ".tar.gz")


def get_download_url() -> str:
    """Construct the download URL for the standalone Python.

    Returns:
        Full URL for downloading the Python standalone archive.
    """
    python_version = get_python_full_version()
    platform_str, ext = _get_platform_info()

    filename = (
        f"cpython-{python_version}+{RELEASE_TAG}-{platform_str}-install_only{ext}"
    )
    url = (
        f"https://github.com/astral-sh/python-build-standalone/releases/"
        f"download/{RELEASE_TAG}/{filename}"
    )
    return url


def _get_subprocess_kwargs() -> dict:
    """Get platform-specific subprocess kwargs for hiding console windows.

    Returns:
        Dict with startupinfo for Windows, empty dict otherwise.
    """
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        return {"startupinfo": startupinfo}
    return {}


def _get_clean_env() -> dict:
    """Get a clean environment dict without QGIS-specific variables.

    Returns:
        A copy of os.environ with PYTHONPATH, PYTHONHOME, etc. removed.
    """
    env = os.environ.copy()
    for var in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "QGIS_PREFIX_PATH",
        "QGIS_PLUGINPATH",
    ):
        env.pop(var, None)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def download_python_standalone(
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[bool, str]:
    """Download and install Python standalone using QGIS network manager.

    Uses QgsBlockingNetworkRequest to respect QGIS proxy settings.

    Args:
        progress_callback: Function called with (percent, message) for progress.
        cancel_check: Function that returns True if operation should be cancelled.

    Returns:
        Tuple of (success, message).
    """
    if standalone_python_exists():
        _log("Python standalone already exists", Qgis.MessageLevel.Info)
        return True, "Python standalone already installed"

    url = get_download_url()
    python_version = get_python_full_version()

    _log(f"Downloading Python {python_version} from: {url}", Qgis.MessageLevel.Info)

    if progress_callback:
        progress_callback(0, f"Downloading Python {python_version}...")

    fd, temp_path = tempfile.mkstemp(suffix=".tar.gz")
    os.close(fd)

    try:
        if cancel_check and cancel_check():
            return False, "Download cancelled"

        request = QgsBlockingNetworkRequest()
        qurl = QUrl(url)

        if progress_callback:
            progress_callback(2, "Connecting to download server...")

        err = request.get(QNetworkRequest(qurl))

        if err != QgsBlockingNetworkRequest.NoError:
            error_msg = request.errorMessage()
            if "404" in error_msg or "Not Found" in error_msg:
                error_msg = (
                    f"Python {python_version} not available for this platform. "
                    f"URL: {url}"
                )
            else:
                error_msg = f"Download failed: {error_msg}"
            _log(error_msg, Qgis.MessageLevel.Critical)
            return False, error_msg

        if cancel_check and cancel_check():
            return False, "Download cancelled"

        reply = request.reply()
        content = reply.content()

        if progress_callback:
            total_mb = len(content) / (1024 * 1024)
            progress_callback(5, f"Downloaded {total_mb:.1f} MB, extracting...")

        with open(temp_path, "wb") as f:
            f.write(content.data())

        _log(
            f"Download complete ({len(content)} bytes), extracting...",
            Qgis.MessageLevel.Info,
        )

        if progress_callback:
            progress_callback(6, "Extracting Python...")

        if os.path.exists(STANDALONE_DIR):
            shutil.rmtree(STANDALONE_DIR)

        os.makedirs(STANDALONE_DIR, exist_ok=True)

        if temp_path.endswith(".tar.gz") or temp_path.endswith(".tgz"):
            with tarfile.open(temp_path, "r:gz") as tar:
                _safe_extract_tar(tar, STANDALONE_DIR)
        else:
            with zipfile.ZipFile(temp_path, "r") as z:
                _safe_extract_zip(z, STANDALONE_DIR)

        if progress_callback:
            progress_callback(9, "Verifying Python installation...")

        success, verify_msg = verify_standalone_python()

        if success:
            if progress_callback:
                progress_callback(10, f"Python {python_version} installed")
            _log("Python standalone installed successfully", Qgis.MessageLevel.Success)
            return True, f"Python {python_version} installed successfully"
        else:
            return False, f"Verification failed: {verify_msg}"

    except InterruptedError:
        return False, "Download cancelled"
    except Exception as e:
        error_msg = f"Installation failed: {str(e)}"
        _log(error_msg, Qgis.MessageLevel.Critical)

        if sys.platform == "win32":
            error_lower = str(e).lower()
            if any(w in error_lower for w in ("denied", "access", "permission")):
                antivirus_help = (
                    "This may be caused by antivirus software blocking "
                    "the extraction.\nPlease try:\n"
                    "  1. Temporarily disable your antivirus\n"
                    "  2. Add an exclusion for: {}\n"
                    "  3. Try the installation again".format(STANDALONE_DIR)
                )
                _log(antivirus_help, Qgis.MessageLevel.Warning)
                error_msg = f"{error_msg}\n\n{antivirus_help}"

        return False, error_msg
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def verify_standalone_python() -> Tuple[bool, str]:
    """Verify that the standalone Python installation works.

    Returns:
        Tuple of (success, message).
    """
    python_path = get_standalone_python_path()

    if not os.path.exists(python_path):
        return False, f"Python executable not found at {python_path}"

    if sys.platform != "win32":
        try:
            import stat

            os.chmod(
                python_path,
                stat.S_IRWXU
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH,
            )
        except OSError:
            pass

    try:
        env = _get_clean_env()
        subprocess_kwargs = _get_subprocess_kwargs()

        result = subprocess.run(  # nosec B603
            [python_path, "-c", "import sys; print(sys.version)"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            **subprocess_kwargs,
        )

        if result.returncode == 0:
            version_output = result.stdout.strip().split()[0]

            if not version_output.startswith(
                f"{sys.version_info.major}.{sys.version_info.minor}"
            ):
                _log(
                    f"Python version mismatch: got {version_output}, "
                    f"expected {get_python_full_version()}",
                    Qgis.MessageLevel.Warning,
                )
                return (
                    False,
                    f"Version mismatch: downloaded {version_output}, "
                    f"expected {get_python_full_version()}",
                )

            _log(
                f"Verified Python standalone: {version_output}",
                Qgis.MessageLevel.Success,
            )
            return True, f"Python {version_output} verified"
        else:
            error = result.stderr or "Unknown error"
            _log(f"Python verification failed: {error}", Qgis.MessageLevel.Warning)
            return False, f"Verification failed: {error[:100]}"

    except subprocess.TimeoutExpired:
        return False, "Python verification timed out"
    except Exception as e:
        return False, f"Verification error: {str(e)[:100]}"


def remove_standalone_python() -> Tuple[bool, str]:
    """Remove the standalone Python installation.

    Returns:
        Tuple of (success, message).
    """
    if not os.path.exists(STANDALONE_DIR):
        return True, "Standalone Python not installed"

    try:
        shutil.rmtree(STANDALONE_DIR)
        _log("Removed standalone Python installation", Qgis.MessageLevel.Success)
        return True, "Standalone Python removed"
    except Exception as e:
        error_msg = f"Failed to remove: {str(e)}"
        _log(error_msg, Qgis.MessageLevel.Warning)
        return False, error_msg
