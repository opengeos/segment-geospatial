"""Virtual environment manager for SamGeo QGIS Plugin.

Manages the creation, package installation, and verification of a
virtual environment used to isolate SamGeo's Python dependencies
from the QGIS environment.

Adapted from the GeoAI QGIS plugin's venv_manager.py.
"""

import hashlib
import os
import platform
import re
import shutil
import subprocess  # nosec B404
import sys
import tempfile
import time
from typing import Callable, List, Optional, Tuple

from qgis.core import Qgis, QgsMessageLog

PYTHON_VERSION = f"py{sys.version_info.major}.{sys.version_info.minor}"
CACHE_DIR = (
    os.environ.get("SAMGEO_CACHE_DIR")
    or os.environ.get("SAMGEO_VENV_DIR")
    or os.path.expanduser("~/.qgis_samgeo")
)
VENV_DIR = os.path.join(CACHE_DIR, f"venv_{PYTHON_VERSION}")

REQUIRED_PACKAGES = [
    ("torch", ">=2.0.0"),
    ("torchvision", ">=0.15.0"),
    ("segment-geospatial", ""),
    ("sam3", ""),
    ("psutil", ""),
    ("scikit-image", ""),
    ("scikit-learn", ""),
    ("transformers", ""),
]

DEPS_HASH_FILE = os.path.join(VENV_DIR, "deps_hash.txt")
CUDA_FLAG_FILE = os.path.join(VENV_DIR, "cuda_installed.txt")

# Bump when install logic changes significantly to force re-install.
_INSTALL_LOGIC_VERSION = "1"

# Bump independently for CUDA-specific install logic changes.
_CUDA_LOGIC_VERSION = "1"

# Minimum NVIDIA driver versions for each CUDA toolkit version.
_CUDA_DRIVER_REQUIREMENTS = {
    "cu128": 570,
    "cu126": 560,
    "cu124": 550,
    "cu121": 530,
}

# Blackwell (sm_120+) requires cu128.
_MIN_COMPUTE_CAP_FOR_CU128 = 12.0

# Cache for detect_nvidia_gpu() -- avoids re-running nvidia-smi.
_gpu_detect_cache = None  # type: Optional[Tuple[bool, dict]]


def _log(message: str, level=Qgis.MessageLevel.Info):
    """Log a message to the QGIS message log.

    Args:
        message: The message to log.
        level: The log level (default: Qgis.MessageLevel.Info).
    """
    QgsMessageLog.logMessage(message, "SamGeo", level=level)


def _log_system_info():
    """Log system information for debugging installation issues."""
    try:
        qgis_version = Qgis.QGIS_VERSION
    except Exception:
        qgis_version = "Unknown"

    info_lines = [
        "=" * 50,
        "Installation Environment:",
        f"  OS: {sys.platform} ({platform.system()} {platform.release()})",
        f"  Architecture: {platform.machine()}",
        f"  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        f"  QGIS: {qgis_version}",
    ]
    if os.environ.get("SAMGEO_CACHE_DIR"):
        info_lines.append(f"  SAMGEO_CACHE_DIR: {os.environ['SAMGEO_CACHE_DIR']}")
    elif os.environ.get("SAMGEO_VENV_DIR"):
        info_lines.append(f"  SAMGEO_VENV_DIR: {os.environ['SAMGEO_VENV_DIR']}")
    info_lines.append(f"  Cache directory: {CACHE_DIR}")
    info_lines.append("=" * 50)
    for line in info_lines:
        _log(line, Qgis.MessageLevel.Info)


# ---------------------------------------------------------------------------
# CUDA flag persistence
# ---------------------------------------------------------------------------


def _write_cuda_flag(value: str):
    """Persist CUDA install state.

    Args:
        value: One of 'cuda', 'cpu', or 'cuda_fallback'.
    """
    if value == "cuda_fallback":
        content = f"cuda_fallback:{_CUDA_LOGIC_VERSION}"
    else:
        content = value
    try:
        os.makedirs(os.path.dirname(CUDA_FLAG_FILE), exist_ok=True)
        with open(CUDA_FLAG_FILE, "w", encoding="utf-8") as f:
            f.write(content)
    except (OSError, IOError) as e:
        _log(f"Failed to write CUDA flag: {e}", Qgis.MessageLevel.Warning)


def _read_cuda_flag() -> Optional[str]:
    """Read CUDA install state.

    Returns:
        One of 'cuda', 'cpu', 'cuda_fallback', or None.
    """
    try:
        with open(CUDA_FLAG_FILE, "r", encoding="utf-8") as f:
            value = f.read().strip()
        base = value.split(":")[0]
        if base in ("cuda", "cpu", "cuda_fallback"):
            return base
    except (OSError, IOError):
        pass
    return None


# ---------------------------------------------------------------------------
# Dependency hash tracking
# ---------------------------------------------------------------------------


def _compute_deps_hash() -> str:
    """Compute MD5 hash of REQUIRED_PACKAGES + install logic version.

    Returns:
        Hex digest string.
    """
    data = repr(_get_required_packages()).encode("utf-8")
    data += _INSTALL_LOGIC_VERSION.encode("utf-8")
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def _get_required_packages() -> List[Tuple[str, str]]:
    """Return platform-aware dependency list.

    On Windows, adds ``triton-windows`` for SAM3 support.
    """
    packages = list(REQUIRED_PACKAGES)
    if sys.platform == "win32":
        sam3_idx = next(
            (i for i, (name, _) in enumerate(packages) if name == "sam3"),
            len(packages),
        )
        packages.insert(sam3_idx, ("triton-windows", ""))
    return packages


def _read_deps_hash() -> Optional[str]:
    """Read stored deps hash from the venv directory.

    Returns:
        The stored hash string, or None if not found.
    """
    try:
        with open(DEPS_HASH_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except (OSError, IOError):
        return None


def _write_deps_hash():
    """Write the current deps hash to the venv directory."""
    try:
        os.makedirs(os.path.dirname(DEPS_HASH_FILE), exist_ok=True)
        with open(DEPS_HASH_FILE, "w", encoding="utf-8") as f:
            f.write(_compute_deps_hash())
    except (OSError, IOError) as e:
        _log(f"Failed to write deps hash: {e}", Qgis.MessageLevel.Warning)


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------


def detect_nvidia_gpu() -> Tuple[bool, dict]:
    """Detect if an NVIDIA GPU is present by querying nvidia-smi.

    Results are cached for the lifetime of the QGIS session.

    Returns:
        Tuple of (has_gpu, info_dict). info_dict keys: name, compute_cap,
        driver_version, memory_mb (any key may be missing).
    """
    global _gpu_detect_cache
    if _gpu_detect_cache is not None:
        return _gpu_detect_cache

    try:
        subprocess_kwargs = _get_subprocess_kwargs()
        result = subprocess.run(  # nosec B603 B607
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            **subprocess_kwargs,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            best_gpu = {}
            best_compute_cap = -1.0

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]

                gpu_info = {}
                if len(parts) >= 1 and parts[0]:
                    gpu_info["name"] = parts[0]
                if len(parts) >= 2 and parts[1]:
                    try:
                        gpu_info["compute_cap"] = float(parts[1])
                    except ValueError:
                        pass
                if len(parts) >= 3 and parts[2]:
                    gpu_info["driver_version"] = parts[2]
                if len(parts) >= 4 and parts[3]:
                    try:
                        gpu_info["memory_mb"] = int(float(parts[3]))
                    except ValueError:
                        pass

                cc = gpu_info.get("compute_cap", 0.0)
                if cc > best_compute_cap:
                    best_compute_cap = cc
                    best_gpu = gpu_info

            if not best_gpu:
                _gpu_detect_cache = (False, {})
                return _gpu_detect_cache

            _log(
                "NVIDIA GPU detected (best of {}): {}".format(len(lines), best_gpu),
                Qgis.MessageLevel.Info,
            )
            _gpu_detect_cache = (True, best_gpu)
            return _gpu_detect_cache
    except FileNotFoundError:
        pass
    except subprocess.TimeoutExpired:
        _log("nvidia-smi timed out", Qgis.MessageLevel.Warning)
    except Exception as e:
        _log(f"nvidia-smi check failed: {e}", Qgis.MessageLevel.Warning)

    _gpu_detect_cache = (False, {})
    return _gpu_detect_cache


def _select_cuda_index(gpu_info: dict) -> Optional[str]:
    """Choose the correct PyTorch CUDA wheel index based on GPU info.

    Args:
        gpu_info: Dict with GPU information from detect_nvidia_gpu().

    Returns:
        'cu128', 'cu126', 'cu124', or None if driver is too old.
    """
    compute_cap = gpu_info.get("compute_cap")
    gpu_name = gpu_info.get("name", "")

    if compute_cap is not None:
        needs_cu128 = compute_cap >= _MIN_COMPUTE_CAP_FOR_CU128
    else:
        needs_cu128 = "RTX 50" in gpu_name.upper()

    driver_str = gpu_info.get("driver_version", "")
    driver_major = None
    if driver_str:
        try:
            driver_major = int(driver_str.split(".")[0])
        except (ValueError, IndexError):
            _log(
                f"Could not parse driver version: {driver_str}",
                Qgis.MessageLevel.Warning,
            )

    if needs_cu128:
        cuda_index = "cu128"
    else:
        if (
            sys.platform == "win32"
            and driver_major is not None
            and driver_major >= _CUDA_DRIVER_REQUIREMENTS.get("cu126", 0)
        ):
            cuda_index = "cu126"
        else:
            cuda_index = "cu124"

    if driver_major is not None:
        required = _CUDA_DRIVER_REQUIREMENTS.get(cuda_index, 0)
        if driver_major < required:
            _log(
                "NVIDIA driver {} too old for {} (needs >= {}), "
                "will use CPU instead".format(driver_str, cuda_index, required),
                Qgis.MessageLevel.Warning,
            )
            return None

    return cuda_index


# ---------------------------------------------------------------------------
# Error detection helpers
# ---------------------------------------------------------------------------

_SSL_ERROR_PATTERNS = [
    "ssl",
    "certificate verify failed",
    "CERTIFICATE_VERIFY_FAILED",
    "SSLError",
    "SSLCertVerificationError",
    "tlsv1 alert",
    "unable to get local issuer certificate",
    "self signed certificate in certificate chain",
]

_NETWORK_ERROR_PATTERNS = [
    "connectionreseterror",
    "connection aborted",
    "connection was forcibly closed",
    "remotedisconnected",
    "connectionerror",
    "newconnectionerror",
    "maxretryerror",
    "protocolerror",
    "readtimeouterror",
    "connecttimeouterror",
    "network is unreachable",
    "temporary failure in name resolution",
    "name or service not known",
]

# Windows NTSTATUS crash codes
_WINDOWS_CRASH_CODES = {
    3221225477,  # 0xC0000005 unsigned - ACCESS_VIOLATION
    -1073741819,  # 0xC0000005 signed
    3221225725,  # 0xC00000FD unsigned - STACK_OVERFLOW
    -1073741571,  # 0xC00000FD signed
    3221225781,  # 0xC0000135 unsigned - DLL_NOT_FOUND
    -1073741515,  # 0xC0000135 signed
}


def _is_ssl_error(stderr: str) -> bool:
    """Detect SSL/certificate errors in pip output.

    Args:
        stderr: The error output from pip.

    Returns:
        True if SSL errors detected.
    """
    stderr_lower = stderr.lower()
    return any(pattern.lower() in stderr_lower for pattern in _SSL_ERROR_PATTERNS)


def _is_hash_mismatch(output: str) -> bool:
    """Detect pip hash mismatch errors.

    Args:
        output: The pip output to check.

    Returns:
        True if hash mismatch detected.
    """
    output_lower = output.lower()
    return "do not match the hashes" in output_lower or "hash mismatch" in output_lower


def _get_pip_ssl_flags() -> List[str]:
    """Get pip flags to bypass SSL verification for corporate proxies.

    Returns:
        List of pip command-line flags.
    """
    return [
        "--trusted-host",
        "pypi.org",
        "--trusted-host",
        "pypi.python.org",
        "--trusted-host",
        "files.pythonhosted.org",
    ]


def _get_uv_ssl_flags() -> List[str]:
    """Get uv flags to bypass SSL verification for corporate proxies.

    Returns:
        List of uv command-line flags.
    """
    return [
        "--allow-insecure-host",
        "pypi.org",
        "--allow-insecure-host",
        "files.pythonhosted.org",
    ]


def _is_network_error(output: str) -> bool:
    """Detect transient network/connection errors in pip output.

    Args:
        output: The pip output to check.

    Returns:
        True if network errors detected (excluding SSL).
    """
    output_lower = output.lower()
    if _is_ssl_error(output):
        return False
    return any(p in output_lower for p in _NETWORK_ERROR_PATTERNS)


def _is_antivirus_error(stderr: str) -> bool:
    """Detect antivirus/permission blocking in pip output.

    Args:
        stderr: The error output from pip.

    Returns:
        True if antivirus blocking detected.
    """
    stderr_lower = stderr.lower()
    patterns = [
        "access is denied",
        "winerror 5",
        "winerror 225",
        "permission denied",
        "operation did not complete successfully because the file contains a virus",
        "blocked by your administrator",
        "blocked by group policy",
        "applocker",
        "blocked by your organization",
    ]
    return any(p in stderr_lower for p in patterns)


def _is_proxy_auth_error(output: str) -> bool:
    """Detect proxy authentication errors (HTTP 407).

    Args:
        output: The pip output to check.

    Returns:
        True if proxy auth error detected.
    """
    output_lower = output.lower()
    patterns = [
        "407 proxy authentication",
        "proxy authentication required",
        "proxyerror",
    ]
    return any(p in output_lower for p in patterns)


def _is_windows_process_crash(returncode: int) -> bool:
    """Detect Windows process crashes.

    Args:
        returncode: The subprocess return code.

    Returns:
        True if the return code indicates a Windows crash.
    """
    if sys.platform != "win32":
        return False
    return returncode in _WINDOWS_CRASH_CODES


def _classify_batch_error(error_output: str, package_specs: List[str]) -> Optional[str]:
    """Identify which package in a batch install caused the failure.

    Args:
        error_output: Combined stdout+stderr from the failed batch install.
        package_specs: List of package specifiers that were in the batch.

    Returns:
        The package name that likely caused the failure, or None.
    """
    error_lower = error_output.lower()
    for spec in package_specs:
        name = re.split(r"[><=!]", spec)[0].strip()
        if name.lower().replace("-", "_") in error_lower.replace("-", "_"):
            return name
    return None


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _get_clean_env_for_venv() -> dict:
    """Get a clean environment dict for subprocess calls.

    Strips QGIS-specific variables to prevent interference.

    Returns:
        A clean copy of os.environ.
    """
    env = os.environ.copy()
    for var in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "QGIS_PREFIX_PATH",
        "QGIS_PLUGINPATH",
        "PROJ_DATA",
        "PROJ_LIB",
        "GDAL_DATA",
        "GDAL_DRIVER_PATH",
    ):
        env.pop(var, None)
    env["PYTHONIOENCODING"] = "utf-8"

    # Ensure CUDA libraries are discoverable for GPU-accelerated torch.
    _cuda_lib_dirs = []
    cuda_path = env.get("CUDA_PATH", "")
    if cuda_path:
        _cuda_lib_dirs.append(os.path.join(cuda_path, "lib64"))
    for candidate in ("/opt/cuda/lib64", "/usr/local/cuda/lib64"):
        if os.path.isdir(candidate) and candidate not in _cuda_lib_dirs:
            _cuda_lib_dirs.append(candidate)
    if _cuda_lib_dirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        parts = [p for p in existing.split(":") if p]
        for d in _cuda_lib_dirs:
            if d not in parts:
                parts.append(d)
        env["LD_LIBRARY_PATH"] = ":".join(parts)

    proxy_url = _get_qgis_proxy_settings()
    if proxy_url:
        env.setdefault("HTTP_PROXY", proxy_url)
        env.setdefault("HTTPS_PROXY", proxy_url)
    return env


def _get_subprocess_kwargs() -> dict:
    """Get platform-specific subprocess kwargs.

    Includes a safe ``cwd`` so that the venv Python never finds the QGIS
    plugin package via the current working directory.

    Returns:
        Dict with cwd and startupinfo (Windows).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    kwargs = {"cwd": CACHE_DIR}
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kwargs


def _get_qgis_proxy_settings() -> Optional[str]:
    """Read proxy configuration from QGIS settings.

    Returns:
        A proxy URL string, or None if not configured.
    """
    try:
        from qgis.core import QgsSettings
        from urllib.parse import quote as url_quote

        settings = QgsSettings()
        enabled = settings.value("proxy/proxyEnabled", False, type=bool)
        if not enabled:
            return None

        host = settings.value("proxy/proxyHost", "", type=str)
        if not host:
            return None

        port = settings.value("proxy/proxyPort", "", type=str)
        user = settings.value("proxy/proxyUser", "", type=str)
        password = settings.value("proxy/proxyPassword", "", type=str)

        proxy_url = "http://"
        if user:
            proxy_url += url_quote(user, safe="")
            if password:
                proxy_url += ":" + url_quote(password, safe="")
            proxy_url += "@"
        proxy_url += host
        if port:
            proxy_url += f":{port}"
        return proxy_url
    except Exception as e:
        _log(f"Could not read QGIS proxy settings: {e}", Qgis.MessageLevel.Warning)
        return None


def _get_pip_proxy_args() -> List[str]:
    """Get pip --proxy argument if QGIS proxy is configured.

    Returns:
        List with --proxy args, or empty list.
    """
    proxy_url = _get_qgis_proxy_settings()
    if proxy_url:
        safe_url = proxy_url.split("@")[-1] if "@" in proxy_url else proxy_url
        _log(f"Using QGIS proxy for pip: {safe_url}", Qgis.MessageLevel.Info)
        return ["--proxy", proxy_url]
    return []


# ---------------------------------------------------------------------------
# Venv path helpers
# ---------------------------------------------------------------------------


def get_venv_dir() -> str:
    """Get the venv directory path.

    Returns:
        Path to the virtual environment directory.
    """
    return VENV_DIR


def get_venv_site_packages(venv_dir: str = None) -> str:
    """Get the site-packages directory within the venv.

    Args:
        venv_dir: Optional venv directory path. Uses VENV_DIR if None.

    Returns:
        Path to the site-packages directory.
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    if sys.platform == "win32":
        return os.path.join(venv_dir, "Lib", "site-packages")
    else:
        lib_dir = os.path.join(venv_dir, "lib")
        if os.path.exists(lib_dir):
            for entry in os.listdir(lib_dir):
                if entry.startswith("python") and os.path.isdir(
                    os.path.join(lib_dir, entry)
                ):
                    site_packages = os.path.join(lib_dir, entry, "site-packages")
                    if os.path.exists(site_packages):
                        return site_packages

        py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        return os.path.join(venv_dir, "lib", py_version, "site-packages")


def get_venv_python_path(venv_dir: str = None) -> str:
    """Get the Python executable path within the venv.

    Args:
        venv_dir: Optional venv directory path. Uses VENV_DIR if None.

    Returns:
        Path to the venv Python executable.
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        return os.path.join(venv_dir, "bin", "python3")


def get_venv_pip_path(venv_dir: str = None) -> str:
    """Get the pip executable path within the venv.

    Args:
        venv_dir: Optional venv directory path. Uses VENV_DIR if None.

    Returns:
        Path to the venv pip executable.
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "pip.exe")
    else:
        return os.path.join(venv_dir, "bin", "pip")


def venv_exists(venv_dir: str = None) -> bool:
    """Check if the venv exists and has a Python executable.

    Args:
        venv_dir: Optional venv directory path. Uses VENV_DIR if None.

    Returns:
        True if the venv Python executable exists.
    """
    if venv_dir is None:
        venv_dir = VENV_DIR
    python_path = get_venv_python_path(venv_dir)
    return os.path.exists(python_path)


def ensure_venv_packages_available() -> bool:
    """Add the venv site-packages to sys.path so packages can be imported.

    Returns:
        True if packages were made available, False otherwise.
    """
    if not venv_exists():
        _log("Venv does not exist, cannot load packages", Qgis.MessageLevel.Warning)
        return False

    site_packages = get_venv_site_packages()
    if not os.path.exists(site_packages):
        _log(
            f"Venv site-packages not found: {site_packages}", Qgis.MessageLevel.Warning
        )
        return False

    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)
        _log(
            f"Added venv site-packages to sys.path: {site_packages}",
            Qgis.MessageLevel.Info,
        )

    # Fix PROJ database for the venv's pyproj / rasterio / pyogrio.
    _fix_proj_data(site_packages)

    # On Windows, register DLL directories for native packages (torch, etc.)
    if sys.platform == "win32":
        _add_windows_dll_directories(site_packages)

    # Fix stale typing_extensions (QGIS may load old version missing TypeIs)
    if "typing_extensions" in sys.modules:
        try:
            typing_ext = sys.modules["typing_extensions"]
            if not hasattr(typing_ext, "TypeIs"):
                old_ver = getattr(typing_ext, "__version__", "unknown")
                del sys.modules["typing_extensions"]
                import typing_extensions as new_te

                _log(
                    "Reloaded typing_extensions {} -> {} from venv".format(
                        old_ver, new_te.__version__
                    ),
                    Qgis.MessageLevel.Info,
                )
        except Exception:
            _log(
                "Failed to reload typing_extensions, torch may fail",
                Qgis.MessageLevel.Warning,
            )

    return True


def _add_windows_dll_directories(site_packages: str) -> None:
    """Register DLL search directories for native packages on Windows.

    Args:
        site_packages: Path to the venv site-packages directory.
    """
    dll_dirs = [
        os.path.join(site_packages, "torch", "lib"),
        os.path.join(site_packages, "torch", "bin"),
        os.path.join(site_packages, "torchvision"),
    ]

    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    for dll_dir in dll_dirs:
        if os.path.isdir(dll_dir):
            try:
                os.add_dll_directory(dll_dir)
                _log(f"Added DLL directory: {dll_dir}", Qgis.MessageLevel.Info)
            except OSError as exc:
                _log(
                    f"add_dll_directory({dll_dir}) failed: {exc}",
                    Qgis.MessageLevel.Warning,
                )
            if dll_dir not in path_parts:
                path_parts.insert(0, dll_dir)

    os.environ["PATH"] = os.pathsep.join(path_parts)


def _fix_proj_data(site_packages: str) -> None:
    """Set PROJ_DATA/PROJ_LIB and GDAL_DATA for the venv's geospatial libraries.

    Args:
        site_packages: Path to the venv's site-packages directory.
    """
    # --- PROJ database ---
    proj_candidates = [
        os.path.join(site_packages, "pyproj", "proj_dir", "share", "proj"),
        os.path.join(site_packages, "rasterio", "proj_data"),
        os.path.join(site_packages, "pyogrio", "proj_data"),
    ]

    for candidate in proj_candidates:
        proj_db = os.path.join(candidate, "proj.db")
        if os.path.exists(proj_db):
            os.environ["PROJ_DATA"] = candidate
            os.environ["PROJ_LIB"] = candidate
            _log(f"Set PROJ_DATA={candidate}", Qgis.MessageLevel.Info)
            break
    else:
        _log("No venv proj.db found; PROJ_DATA unchanged", Qgis.MessageLevel.Warning)

    # --- GDAL data ---
    gdal_candidates = [
        os.path.join(site_packages, "rasterio", "gdal_data"),
        os.path.join(site_packages, "pyogrio", "gdal_data"),
    ]

    for candidate in gdal_candidates:
        if os.path.isdir(candidate):
            os.environ["GDAL_DATA"] = candidate
            _log(f"Set GDAL_DATA={candidate}", Qgis.MessageLevel.Info)
            break


# ---------------------------------------------------------------------------
# System Python resolution
# ---------------------------------------------------------------------------


def _is_python_executable_name(path: str) -> bool:
    """Return True when a path name looks like a Python interpreter."""
    name = os.path.basename(path).lower()
    if name.endswith(".exe"):
        name = name[:-4]
    if name in ("python", "python3"):
        return True
    if not name.startswith("python"):
        return False
    suffix = name[6:]
    if "-" in suffix:
        return False
    return suffix.isdigit() or (
        suffix.count(".") == 1 and all(part.isdigit() for part in suffix.split("."))
    )


def _is_macos_qgis_app_bundle_python(path: str) -> bool:
    """Return True for unsafe Python launchers in QGIS.app/Contents/MacOS."""
    if not (platform.system() == "Darwin" or sys.platform == "darwin"):
        return False
    parts = os.path.abspath(path).split(os.sep)
    for idx, part in enumerate(parts):
        lower = part.lower()
        if not (lower.startswith("qgis") and lower.endswith(".app")):
            continue
        if idx + 2 >= len(parts):
            return False
        if parts[idx + 1].lower() != "contents" or parts[idx + 2].lower() != "macos":
            return False
        name = os.path.basename(path).lower()
        return name.startswith("qgis") or _is_python_executable_name(path)
    return False


def _get_qgis_python() -> Optional[str]:
    """Get the path to QGIS's bundled Python on Windows.

    Returns:
        Path to the Python executable, or None if not found.
    """
    if sys.platform != "win32":
        return None

    python_path = os.path.join(sys.prefix, "python.exe")
    if not os.path.exists(python_path):
        python_path = os.path.join(sys.prefix, "python3.exe")

    if not os.path.exists(python_path):
        _log("QGIS bundled Python not found at sys.prefix", Qgis.MessageLevel.Warning)
        return None

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        subprocess_kwargs = _get_subprocess_kwargs()

        result = subprocess.run(  # nosec B603
            [python_path, "-c", "import sys; print(sys.version)"],
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
            **subprocess_kwargs,
        )
        if result.returncode == 0:
            _log(
                f"QGIS Python verified: {result.stdout.strip()}", Qgis.MessageLevel.Info
            )
            return python_path
        else:
            _log(
                f"QGIS Python failed verification: {result.stderr}",
                Qgis.MessageLevel.Warning,
            )
            return None
    except Exception as e:
        _log(f"QGIS Python verification error: {e}", Qgis.MessageLevel.Warning)
        return None


def _get_system_python() -> str:
    """Get the path to the Python executable for creating venvs.

    Uses standalone Python downloaded by python_manager, with fallback
    to QGIS's bundled Python on Windows.

    Returns:
        Path to the Python executable.

    Raises:
        RuntimeError: If no suitable Python is found.
    """
    from .python_manager import get_standalone_python_path, standalone_python_exists

    if standalone_python_exists():
        python_path = get_standalone_python_path()
        _log(f"Using standalone Python: {python_path}", Qgis.MessageLevel.Info)
        return python_path

    if sys.platform == "win32":
        qgis_python = _get_qgis_python()
        if qgis_python:
            _log(
                "Standalone Python unavailable, using QGIS Python as fallback",
                Qgis.MessageLevel.Warning,
            )
            return qgis_python
    elif _is_macos_qgis_app_bundle_python(sys.executable):
        raise RuntimeError(
            "QGIS app-bundle Python is not safe for creating virtual "
            "environments; use uv-managed Python instead."
        )

    raise RuntimeError(
        "Python standalone not installed. "
        "Please click 'Install Dependencies' to download Python automatically."
    )


# ---------------------------------------------------------------------------
# Venv creation
# ---------------------------------------------------------------------------


def _cleanup_partial_venv(venv_dir: str):
    """Remove a partially-created venv directory.

    Args:
        venv_dir: Path to the venv directory to clean up.
    """
    if os.path.exists(venv_dir):
        try:
            shutil.rmtree(venv_dir, ignore_errors=True)
            _log(f"Cleaned up partial venv: {venv_dir}", Qgis.MessageLevel.Info)
        except Exception:
            _log(
                f"Could not clean up partial venv: {venv_dir}",
                Qgis.MessageLevel.Warning,
            )


def create_venv(
    venv_dir: str = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Tuple[bool, str]:
    """Create a virtual environment.

    Args:
        venv_dir: Optional directory for the venv. Uses VENV_DIR if None.
        progress_callback: Optional function called with (percent, message).

    Returns:
        Tuple of (success, message).
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    _log(f"Creating virtual environment at: {venv_dir}", Qgis.MessageLevel.Info)

    if progress_callback:
        progress_callback(10, "Creating virtual environment...")

    system_python = None
    python_lookup_error = ""
    try:
        system_python = _get_system_python()
    except RuntimeError as exc:
        python_lookup_error = str(exc)
    if python_lookup_error and system_python is None:
        _log(
            "Python lookup failed; falling back to uv-managed Python if "
            f"available: {python_lookup_error}",
            Qgis.MessageLevel.Warning,
        )
    if system_python:
        _log(f"Using Python: {system_python}", Qgis.MessageLevel.Info)

    from .uv_manager import get_uv_path, uv_exists

    use_uv = uv_exists()

    if use_uv:
        uv_path = get_uv_path()
        uv_python = (
            system_python or f"{sys.version_info.major}.{sys.version_info.minor}"
        )
        cmd = [uv_path, "venv"]
        if system_python is None:
            cmd.append("--managed-python")
        cmd += ["--python", uv_python, venv_dir]
        _log(f"Creating venv with uv: {uv_path}", Qgis.MessageLevel.Info)
    else:
        if system_python is None:
            return False, python_lookup_error
        cmd = [system_python, "-m", "venv", venv_dir]
        _log("Creating venv with python -m venv", Qgis.MessageLevel.Info)

    try:
        env = _get_clean_env_for_venv()
        subprocess_kwargs = _get_subprocess_kwargs()

        result = subprocess.run(  # nosec B603
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            **subprocess_kwargs,
        )

        if result.returncode != 0 and use_uv and system_python:
            _log(
                "uv venv failed ({}), falling back to python -m venv".format(
                    result.stderr.strip() if result.stderr else result.returncode
                ),
                Qgis.MessageLevel.Warning,
            )
            from .uv_manager import remove_uv

            remove_uv()
            use_uv = False
            _cleanup_partial_venv(venv_dir)
            cmd = [system_python, "-m", "venv", venv_dir]
            result = subprocess.run(  # nosec B603
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
                **subprocess_kwargs,
            )

        if result.returncode == 0:
            _log("Virtual environment created successfully", Qgis.MessageLevel.Success)

            if not use_uv:
                pip_path = get_venv_pip_path(venv_dir)
                if not os.path.exists(pip_path):
                    _log(
                        "pip not found in venv, bootstrapping with ensurepip...",
                        Qgis.MessageLevel.Info,
                    )
                    python_in_venv = get_venv_python_path(venv_dir)
                    ensurepip_cmd = [python_in_venv, "-m", "ensurepip", "--upgrade"]
                    try:
                        ensurepip_result = subprocess.run(  # nosec B603
                            ensurepip_cmd,
                            capture_output=True,
                            text=True,
                            timeout=120,
                            env=env,
                            **subprocess_kwargs,
                        )
                        if ensurepip_result.returncode == 0:
                            _log(
                                "pip bootstrapped via ensurepip",
                                Qgis.MessageLevel.Success,
                            )
                        else:
                            err = ensurepip_result.stderr or ensurepip_result.stdout
                            _log(
                                f"ensurepip failed: {err[:200]}",
                                Qgis.MessageLevel.Warning,
                            )
                            _cleanup_partial_venv(venv_dir)
                            return (
                                False,
                                f"Failed to bootstrap pip: {err[:200]}",
                            )
                    except Exception as e:
                        _log(f"ensurepip exception: {e}", Qgis.MessageLevel.Warning)
                        _cleanup_partial_venv(venv_dir)
                        return (
                            False,
                            f"Failed to bootstrap pip: {str(e)[:200]}",
                        )

            if progress_callback:
                progress_callback(15, "Virtual environment created")
            return True, "Virtual environment created"
        else:
            error_msg = (
                result.stderr or result.stdout or f"Return code {result.returncode}"
            )
            _log(f"Failed to create venv: {error_msg}", Qgis.MessageLevel.Critical)
            _cleanup_partial_venv(venv_dir)
            return False, f"Failed to create venv: {error_msg[:200]}"

    except subprocess.TimeoutExpired:
        _log("Virtual environment creation timed out", Qgis.MessageLevel.Critical)
        _cleanup_partial_venv(venv_dir)
        return False, "Virtual environment creation timed out"
    except FileNotFoundError:
        missing_executable = cmd[0] if cmd else system_python
        _log(
            f"Venv creation executable not found: {missing_executable}",
            Qgis.MessageLevel.Critical,
        )
        return False, f"Executable not found: {missing_executable}"
    except Exception as e:
        _log(f"Exception during venv creation: {str(e)}", Qgis.MessageLevel.Critical)
        _cleanup_partial_venv(venv_dir)
        return False, f"Error: {str(e)[:200]}"


# ---------------------------------------------------------------------------
# Pip install with progress
# ---------------------------------------------------------------------------


class _PipResult:
    """Lightweight result object compatible with subprocess.CompletedProcess."""

    def __init__(self, returncode: int, stdout: str, stderr: str):
        """Initialize the result.

        Args:
            returncode: The process return code.
            stdout: Standard output text.
            stderr: Standard error text.
        """
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _parse_pip_download_line(line: str) -> Optional[str]:
    """Extract a human-readable status from a pip stdout/stderr line.

    Args:
        line: A line from pip output.

    Returns:
        Human-readable download status string, or None.
    """
    m = re.search(r"Downloading\s+(\S+)\s+\(([^)]+)\)", line)
    if not m:
        return None

    raw_name = m.group(1)
    size = m.group(2)

    if "/" in raw_name:
        raw_name = raw_name.rsplit("/", 1)[-1]

    name_match = re.match(r"([A-Za-z][A-Za-z0-9_]*)", raw_name)
    pkg_name = name_match.group(1) if name_match else raw_name

    size_match = re.match(r"([\d.]+)\s*(kB|MB|GB)", size)
    if size_match:
        num = float(size_match.group(1))
        unit = size_match.group(2)
        if unit == "MB" and num >= 1000:
            size = "{:.1f} GB".format(num / 1000)

    return "Downloading {} ({})".format(pkg_name, size)


def _run_pip_install(
    cmd: List[str],
    timeout: int,
    env: dict,
    subprocess_kwargs: dict,
    label: str,
    progress_start: int,
    progress_end: int,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> _PipResult:
    """Run a pip/uv install command with real-time progress updates.

    Args:
        cmd: The install command to run.
        timeout: Maximum time in seconds.
        env: Environment variables dict.
        subprocess_kwargs: Platform-specific kwargs for subprocess.
        label: Human-readable label for progress messages.
        progress_start: Start percentage for this install's progress range.
        progress_end: End percentage for this install's progress range.
        progress_callback: Optional progress callback.
        cancel_check: Optional cancellation check callback.

    Returns:
        _PipResult with returncode, stdout, and stderr.
    """
    poll_interval = 2

    stdout_fd, stdout_path = tempfile.mkstemp(suffix="_stdout.txt", prefix="pip_")
    stderr_fd, stderr_path = tempfile.mkstemp(suffix="_stderr.txt", prefix="pip_")

    try:
        stdout_file = os.fdopen(stdout_fd, "w", encoding="utf-8")
        stderr_file = os.fdopen(stderr_fd, "w", encoding="utf-8")
    except Exception:
        try:
            os.close(stdout_fd)
        except Exception:  # nosec B110
            pass
        try:
            os.close(stderr_fd)
        except Exception:  # nosec B110
            pass
        raise

    process = None
    try:
        process = subprocess.Popen(  # nosec B603
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            env=env,
            **subprocess_kwargs,
        )

        start_time = time.monotonic()
        last_download_status = ""

        while True:
            try:
                process.wait(timeout=poll_interval)
                break
            except subprocess.TimeoutExpired:
                pass

            elapsed = int(time.monotonic() - start_time)

            if cancel_check and cancel_check():
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                return _PipResult(-1, "", "Installation cancelled")

            if elapsed >= timeout:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                raise subprocess.TimeoutExpired(cmd, timeout)

            # Read last lines to find download progress
            try:
                with open(stdout_path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(0, 2)
                    file_size = f.tell()
                    read_from = max(0, file_size - 4096)
                    f.seek(read_from)
                    tail = f.read()
                    lines = tail.strip().split("\n")
                    for line in reversed(lines):
                        parsed = _parse_pip_download_line(line)
                        if parsed:
                            last_download_status = parsed
                            break
            except Exception:  # nosec B110
                pass

            if elapsed >= 60:
                elapsed_str = "{}m {}s".format(elapsed // 60, elapsed % 60)
            else:
                elapsed_str = "{}s".format(elapsed)

            if last_download_status:
                msg = "{}... {}".format(last_download_status, elapsed_str)
            else:
                msg = "Installing {}... {}".format(label, elapsed_str)

            progress_range = progress_end - progress_start
            if timeout > 0:
                fraction = min(elapsed / timeout, 0.9)
            else:
                fraction = 0
            interpolated = progress_start + int(progress_range * fraction)
            interpolated = min(interpolated, progress_end - 1)

            if progress_callback:
                progress_callback(interpolated, msg)

        stdout_file.close()
        stderr_file.close()
        stdout_file = None
        stderr_file = None

        try:
            with open(stdout_path, "r", encoding="utf-8", errors="replace") as f:
                full_stdout = f.read()
        except Exception:
            full_stdout = ""

        try:
            with open(stderr_path, "r", encoding="utf-8", errors="replace") as f:
                full_stderr = f.read()
        except Exception:
            full_stderr = ""

        return _PipResult(process.returncode, full_stdout, full_stderr)

    except subprocess.TimeoutExpired:
        raise
    except Exception:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except Exception:
                process.kill()
        raise
    finally:
        if stdout_file is not None:
            try:
                stdout_file.close()
            except Exception:  # nosec B110
                pass
        if stderr_file is not None:
            try:
                stderr_file.close()
            except Exception:  # nosec B110
                pass
        try:
            os.unlink(stdout_path)
        except Exception:  # nosec B110
            pass
        try:
            os.unlink(stderr_path)
        except Exception:  # nosec B110
            pass


# ---------------------------------------------------------------------------
# Dependency installation
# ---------------------------------------------------------------------------


def _is_cpu_torch_installed(
    python_path: str, env: dict, subprocess_kwargs: dict
) -> bool:
    """Check if installed torch has no CUDA support (CPU-only build).

    Args:
        python_path: Path to the venv Python.
        env: Environment dict for subprocess.
        subprocess_kwargs: Platform-specific subprocess kwargs.

    Returns:
        True if CPU-only torch is installed.
    """
    try:
        result = subprocess.run(  # nosec B603
            [python_path, "-c", "import torch; print(torch.version.cuda)"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            **subprocess_kwargs,
        )
        if result.returncode == 0:
            return result.stdout.strip() == "None"
    except Exception:  # nosec B110
        pass
    return False


def _reinstall_cpu_torch(
    venv_dir: str,
    progress_callback: Optional[Callable[[int, str], None]] = None,
):
    """Reinstall CPU-only torch/torchvision after CUDA failure.

    Args:
        venv_dir: Path to the virtual environment.
        progress_callback: Optional progress callback.
    """
    from .uv_manager import get_uv_path, uv_exists

    python_path = get_venv_python_path(venv_dir)
    env = _get_clean_env_for_venv()
    subprocess_kwargs = _get_subprocess_kwargs()
    _use_uv = uv_exists()
    _uv_path = get_uv_path() if _use_uv else None

    _log("Reinstalling CPU-only torch/torchvision...", Qgis.MessageLevel.Warning)
    if progress_callback:
        progress_callback(96, "CUDA failed, reinstalling CPU torch...")

    try:
        if _use_uv:
            uninstall_cmd = [
                _uv_path,
                "pip",
                "uninstall",
                "--python",
                python_path,
                "torch",
                "torchvision",
            ]
        else:
            uninstall_cmd = [
                python_path,
                "-m",
                "pip",
                "uninstall",
                "-y",
                "torch",
                "torchvision",
            ]
        subprocess.run(  # nosec B603
            uninstall_cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            **subprocess_kwargs,
        )
    except Exception as e:
        _log(f"torch uninstall error (continuing): {e}", Qgis.MessageLevel.Warning)

    for pkg in ("torch>=2.0.0", "torchvision>=0.15.0"):
        try:
            if _use_uv:
                cmd = (
                    [
                        _uv_path,
                        "pip",
                        "install",
                        "--python",
                        python_path,
                        "--upgrade",
                    ]
                    + _get_uv_ssl_flags()
                    + [pkg]
                )
            else:
                cmd = (
                    [
                        python_path,
                        "-m",
                        "pip",
                        "install",
                        "--no-warn-script-location",
                        "--disable-pip-version-check",
                        "--prefer-binary",
                    ]
                    + _get_pip_ssl_flags()
                    + [pkg]
                )
            result = subprocess.run(  # nosec B603
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
                **subprocess_kwargs,
            )
            if result.returncode == 0:
                _log(f"Installed {pkg} (CPU)", Qgis.MessageLevel.Success)
            else:
                err = result.stderr or result.stdout or ""
                _log(
                    f"Failed to install {pkg} (CPU): {err[:200]}",
                    Qgis.MessageLevel.Warning,
                )
        except Exception as e:
            _log(f"Exception installing {pkg} (CPU): {e}", Qgis.MessageLevel.Warning)

    if progress_callback:
        progress_callback(98, "CPU torch installed, re-verifying...")


def _verify_cuda_in_venv(venv_dir: str) -> bool:
    """Run a CUDA smoke test inside the venv.

    Args:
        venv_dir: Path to the virtual environment.

    Returns:
        True if CUDA is functional.
    """
    python_path = get_venv_python_path(venv_dir)
    env = _get_clean_env_for_venv()
    subprocess_kwargs = _get_subprocess_kwargs()

    cuda_test_code = (
        "import torch; "
        "print('torch=' + torch.__version__); "
        "print('cuda_built=' + str(torch.version.cuda)); "
        "assert torch.cuda.is_available(), 'CUDA not available'; "
        "print('device=' + torch.cuda.get_device_name(0)); "
        "t = torch.zeros(1, device='cuda'); "
        "torch.cuda.synchronize(); "
        "print('CUDA OK')"
    )

    try:
        for attempt in (1, 2):
            result = subprocess.run(  # nosec B603
                [python_path, "-c", cuda_test_code],
                capture_output=True,
                text=True,
                timeout=180 if attempt == 2 else 120,
                env=env,
                **subprocess_kwargs,
            )
            if result.returncode == 0 and "CUDA OK" in result.stdout:
                _log(
                    "CUDA verification passed: {}".format(result.stdout.strip()[:400]),
                    Qgis.MessageLevel.Success,
                )
                return True

            out = result.stdout or ""
            err = result.stderr or ""
            _log(
                "CUDA verification attempt {} failed (rc={}).\n"
                "stdout: {}\nstderr: {}".format(
                    attempt, result.returncode, out[:400], err[:400]
                ),
                Qgis.MessageLevel.Warning,
            )
            if attempt == 1:
                time.sleep(2)
        return False
    except Exception as e:
        _log(f"CUDA verification exception: {e}", Qgis.MessageLevel.Warning)
        return False


def _is_torch_related_verify_failure(message: str) -> bool:
    """Return True if a venv verification failure is likely caused by torch/CUDA.

    This is used to decide whether it is appropriate to auto-fallback from CUDA
    torch to CPU torch. Non-torch package verification failures (e.g. ``sam3``)
    should NOT trigger a torch reinstall.

    Args:
        message: The verification failure message.

    Returns:
        True if the failure is torch/CUDA related.
    """
    msg = (message or "").lower()
    if not msg:
        return False

    # Do not treat optional package import failures as torch/CUDA verification
    # failures, even if their traceback mentions torch imports internally.
    if "package sam3 is broken" in msg:
        return False

    torch_markers = (
        "package torch is broken",
        "package torchvision is broken",
        "verification error: torch",
        "verification error: torchvision",
        "torch not compiled with cuda",
        "cuda not available",
        "shm.dll",
        "torch dll",
    )
    return any(marker in msg for marker in torch_markers)


def _is_optional_install_package(package_name: str) -> bool:
    """Return True if install failure for this package can be non-fatal.

    Args:
        package_name: The package name.

    Returns:
        True if the package is optional on this platform.
    """
    if sys.platform == "win32" and package_name in ("sam3", "triton-windows"):
        return True
    return False


def install_dependencies(
    venv_dir: str = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    cuda_enabled: bool = False,
) -> Tuple[bool, str]:
    """Install all required packages into the virtual environment.

    Args:
        venv_dir: Optional venv directory path. Uses VENV_DIR if None.
        progress_callback: Optional function called with (percent, message).
        cancel_check: Optional function returning True to cancel.
        cuda_enabled: Whether to install CUDA-enabled PyTorch.

    Returns:
        Tuple of (success, message).
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    if not venv_exists(venv_dir):
        return False, "Virtual environment does not exist"

    from .uv_manager import get_uv_path, uv_exists

    use_uv = uv_exists()
    uv_path = get_uv_path() if use_uv else None
    if use_uv:
        _log(f"Installing dependencies using uv: {uv_path}", Qgis.MessageLevel.Info)
    else:
        pip_path = get_venv_pip_path(venv_dir)
        _log(f"Installing dependencies using pip: {pip_path}", Qgis.MessageLevel.Info)
    if cuda_enabled:
        _log(
            "CUDA mode enabled - will install GPU-accelerated PyTorch",
            Qgis.MessageLevel.Info,
        )

    _cuda_fell_back = False
    _driver_too_old = False

    required_packages = _get_required_packages()
    base_progress = 20
    progress_range = 80
    python_path = get_venv_python_path(venv_dir)
    env = _get_clean_env_for_venv()
    subprocess_kwargs = _get_subprocess_kwargs()

    # -- Partition packages into CUDA (individual) and batch groups -----------
    cuda_packages: List[Tuple[str, str]] = []
    batch_packages: List[Tuple[str, str]] = []
    for pkg_name, ver_spec in required_packages:
        if cuda_enabled and pkg_name in ("torch", "torchvision"):
            cuda_packages.append((pkg_name, ver_spec))
        else:
            batch_packages.append((pkg_name, ver_spec))

    # Progress allocation
    cuda_progress_weight = 40 if cuda_packages else 0
    cuda_start = base_progress
    cuda_end = base_progress + int(progress_range * cuda_progress_weight / 100)
    batch_start = cuda_end
    batch_end = base_progress + progress_range  # == 100

    # -- Phase A: CUDA packages (individual installs) -------------------------
    _force_cuda_reinstall = False
    if cuda_packages:
        _precheck_env = _get_clean_env_for_venv()
        _precheck_kwargs = _get_subprocess_kwargs()
        if _is_cpu_torch_installed(python_path, _precheck_env, _precheck_kwargs):
            _force_cuda_reinstall = True
            _log(
                "CPU torch detected in venv, CUDA packages will use "
                "--force-reinstall",
                Qgis.MessageLevel.Info,
            )

        num_cuda = len(cuda_packages)
        for ci, (package_name, version_spec) in enumerate(cuda_packages):
            if cancel_check and cancel_check():
                _log("Installation cancelled by user", Qgis.MessageLevel.Warning)
                return False, "Installation cancelled"

            package_spec = f"{package_name}{version_spec}"
            pkg_start = cuda_start + int((cuda_end - cuda_start) * ci / num_cuda)
            pkg_end = cuda_start + int((cuda_end - cuda_start) * (ci + 1) / num_cuda)

            if package_name == "torch":
                label = "GPU PyTorch (~2.5 GB)"
            else:
                label = "{} (CUDA)".format(package_name)

            if progress_callback:
                progress_callback(
                    pkg_start,
                    "Installing GPU dependencies... ({}/{})".format(ci + 1, num_cuda),
                )
            _log(
                "[CUDA {}/{}] Installing {}...".format(ci + 1, num_cuda, package_spec),
                Qgis.MessageLevel.Info,
            )

            # Build install args
            if use_uv:
                pip_args = [
                    "pip",
                    "install",
                    "--python",
                    python_path,
                    "--upgrade",
                ]
                pip_args.extend(_get_uv_ssl_flags())
                pip_args.append(package_spec)
            else:
                pip_args = [
                    "install",
                    "--upgrade",
                    "--no-warn-script-location",
                    "--disable-pip-version-check",
                    "--prefer-binary",
                ]
                pip_args.extend(_get_pip_ssl_flags())
                pip_args.extend(_get_pip_proxy_args())
                pip_args.append(package_spec)

            is_cuda_package = True
            _, gpu_info = detect_nvidia_gpu()
            cuda_index = _select_cuda_index(gpu_info)
            if cuda_index is None:
                _log(
                    "Driver too old for CUDA, installing CPU {} instead".format(
                        package_name
                    ),
                    Qgis.MessageLevel.Warning,
                )
                is_cuda_package = False
                _driver_too_old = True
            else:
                pip_args.extend(
                    [
                        "--index-url",
                        "https://download.pytorch.org/whl/{}".format(cuda_index),
                        "--no-cache" if use_uv else "--no-cache-dir",
                    ]
                )
                _log(
                    "Using CUDA {} index for {}".format(cuda_index, package_name),
                    Qgis.MessageLevel.Info,
                )

            # Uninstall CPU torch before CUDA install
            if _force_cuda_reinstall and is_cuda_package:
                _log(
                    "Uninstalling CPU {} before CUDA install".format(package_name),
                    Qgis.MessageLevel.Info,
                )
                try:
                    if use_uv:
                        uninstall_cmd = [
                            uv_path,
                            "pip",
                            "uninstall",
                            "--python",
                            python_path,
                            package_name,
                        ]
                    else:
                        uninstall_cmd = [
                            python_path,
                            "-m",
                            "pip",
                            "uninstall",
                            "-y",
                            package_name,
                        ]
                    subprocess.run(  # nosec B603
                        uninstall_cmd,
                        capture_output=True,
                        text=True,
                        timeout=120,
                        env=env,
                        **subprocess_kwargs,
                    )
                except Exception as exc:
                    _log(
                        f"Failed to uninstall CPU {package_name}: {exc}",
                        Qgis.MessageLevel.Warning,
                    )

            if use_uv:
                base_cmd = [uv_path] + pip_args
            else:
                base_cmd = [python_path, "-m", "pip"] + pip_args

            install_failed = False
            install_error_msg = ""
            last_returncode = None
            pkg_timeout = 2400

            try:
                result = _run_pip_install(
                    cmd=base_cmd,
                    timeout=pkg_timeout,
                    env=env,
                    subprocess_kwargs=subprocess_kwargs,
                    label=label,
                    progress_start=pkg_start,
                    progress_end=pkg_end,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                )

                if (
                    result.returncode == -1
                    and "cancelled" in (result.stderr or "").lower()
                ):
                    return False, "Installation cancelled"

                # Retry on hash mismatch
                if result.returncode != 0:
                    error_output = result.stderr or result.stdout or ""
                    if _is_hash_mismatch(error_output):
                        _log(
                            "Hash mismatch, retrying with --no-cache...",
                            Qgis.MessageLevel.Warning,
                        )
                        nocache_flag = "--no-cache" if use_uv else "--no-cache-dir"
                        result = _run_pip_install(
                            cmd=base_cmd + [nocache_flag],
                            timeout=pkg_timeout,
                            env=env,
                            subprocess_kwargs=subprocess_kwargs,
                            label=label,
                            progress_start=pkg_start,
                            progress_end=pkg_end,
                            progress_callback=progress_callback,
                            cancel_check=cancel_check,
                        )

                # Retry on network errors
                if result.returncode != 0:
                    error_output = result.stderr or result.stdout or ""
                    if _is_network_error(error_output):
                        for attempt in range(1, 3):
                            _log(
                                "Network error, retrying in 5s "
                                "(attempt {}/2)...".format(attempt),
                                Qgis.MessageLevel.Warning,
                            )
                            if progress_callback:
                                progress_callback(
                                    pkg_start,
                                    "Network error, retrying {}...".format(
                                        package_name
                                    ),
                                )
                            time.sleep(5)
                            if cancel_check and cancel_check():
                                return False, "Installation cancelled"
                            result = _run_pip_install(
                                cmd=base_cmd,
                                timeout=pkg_timeout,
                                env=env,
                                subprocess_kwargs=subprocess_kwargs,
                                label=label,
                                progress_start=pkg_start,
                                progress_end=pkg_end,
                                progress_callback=progress_callback,
                                cancel_check=cancel_check,
                            )
                            if result.returncode == 0:
                                break

                if result.returncode == 0:
                    _log(
                        "Successfully installed {}".format(package_spec),
                        Qgis.MessageLevel.Success,
                    )
                    if progress_callback:
                        progress_callback(pkg_end, "{} installed".format(package_name))
                else:
                    error_msg = (
                        result.stderr
                        or result.stdout
                        or "Return code {}".format(result.returncode)
                    )
                    _log(
                        "Failed to install {}: {}".format(
                            package_spec, error_msg[:500]
                        ),
                        Qgis.MessageLevel.Critical,
                    )
                    install_failed = True
                    install_error_msg = error_msg
                    last_returncode = result.returncode

            except subprocess.TimeoutExpired:
                _log(
                    "Installation of {} timed out".format(package_spec),
                    Qgis.MessageLevel.Critical,
                )
                install_failed = True
                install_error_msg = "Installation of {} timed out".format(package_name)
            except Exception as e:
                _log(
                    "Exception installing {}: {}".format(package_spec, e),
                    Qgis.MessageLevel.Critical,
                )
                install_failed = True
                install_error_msg = "Error installing {}: {}".format(
                    package_name, str(e)[:200]
                )

            # CUDA -> CPU fallback
            if install_failed and is_cuda_package:
                _log(
                    "CUDA install of {} failed, falling back to CPU...".format(
                        package_name
                    ),
                    Qgis.MessageLevel.Warning,
                )
                if progress_callback:
                    progress_callback(
                        pkg_start,
                        "CUDA failed, installing {} (CPU)...".format(package_name),
                    )
                if use_uv:
                    cpu_pip_args = [
                        "pip",
                        "install",
                        "--python",
                        python_path,
                        "--upgrade",
                    ]
                    cpu_pip_args.extend(_get_uv_ssl_flags())
                    cpu_pip_args.append(package_spec)
                    cpu_cmd = [uv_path] + cpu_pip_args
                else:
                    cpu_pip_args = [
                        "install",
                        "--upgrade",
                        "--no-warn-script-location",
                        "--disable-pip-version-check",
                        "--prefer-binary",
                    ]
                    cpu_pip_args.extend(_get_pip_ssl_flags())
                    cpu_pip_args.append(package_spec)
                    cpu_cmd = [python_path, "-m", "pip"] + cpu_pip_args
                try:
                    cpu_result = _run_pip_install(
                        cmd=cpu_cmd,
                        timeout=600,
                        env=env,
                        subprocess_kwargs=subprocess_kwargs,
                        label="{} (CPU fallback)".format(package_name),
                        progress_start=pkg_start,
                        progress_end=pkg_end,
                        progress_callback=progress_callback,
                        cancel_check=cancel_check,
                    )
                    if cpu_result.returncode == 0:
                        _log(
                            "Successfully installed {} (CPU)".format(package_spec),
                            Qgis.MessageLevel.Success,
                        )
                        if progress_callback:
                            progress_callback(
                                pkg_end,
                                "{} installed (CPU)".format(package_name),
                            )
                        install_failed = False
                        _cuda_fell_back = True
                    else:
                        cpu_err = cpu_result.stderr or cpu_result.stdout or ""
                        install_error_msg = (
                            "CUDA and CPU install both failed for {}: {}".format(
                                package_name, cpu_err[:200]
                            )
                        )
                except subprocess.TimeoutExpired:
                    install_error_msg = (
                        "CUDA and CPU install both timed out for {}".format(
                            package_name
                        )
                    )
                except Exception as e:
                    install_error_msg = (
                        "CUDA and CPU install both failed for {}: {}".format(
                            package_name, str(e)[:200]
                        )
                    )

            if install_failed:
                _log(
                    "pip error output: {}".format(install_error_msg[:500]),
                    Qgis.MessageLevel.Critical,
                )
                if _is_ssl_error(install_error_msg):
                    return (
                        False,
                        "Failed to install {}: SSL certificate error".format(
                            package_name
                        ),
                    )
                if _is_proxy_auth_error(install_error_msg):
                    return (
                        False,
                        "Failed to install {}: proxy authentication "
                        "required".format(package_name),
                    )
                if _is_network_error(install_error_msg):
                    return (
                        False,
                        "Failed to install {}: network error".format(package_name),
                    )
                if _is_antivirus_error(install_error_msg):
                    return (
                        False,
                        "Failed to install {}: blocked by antivirus or "
                        "security policy".format(package_name),
                    )
                if last_returncode is not None and _is_windows_process_crash(
                    last_returncode
                ):
                    return (
                        False,
                        "Failed to install {}: process crashed "
                        "(code {})".format(package_name, last_returncode),
                    )
                return (
                    False,
                    "Failed to install {}: {}".format(
                        package_name, install_error_msg[:200]
                    ),
                )

    # -- Phase B: Batch install remaining packages ----------------------------
    if batch_packages:
        if cancel_check and cancel_check():
            _log("Installation cancelled by user", Qgis.MessageLevel.Warning)
            return False, "Installation cancelled"

        batch_specs = ["{}{}".format(name, ver) for name, ver in batch_packages]
        _log(
            "Installing {} packages in batch: {}".format(
                len(batch_specs), ", ".join(batch_specs)
            ),
            Qgis.MessageLevel.Info,
        )
        if progress_callback:
            progress_callback(batch_start, "Installing dependencies...")

        if use_uv:
            pip_args = [
                "pip",
                "install",
                "--python",
                python_path,
                "--upgrade",
            ]
            pip_args.extend(_get_uv_ssl_flags())
            pip_args.extend(batch_specs)
            base_cmd = [uv_path] + pip_args
        else:
            pip_args = [
                "install",
                "--upgrade",
                "--no-warn-script-location",
                "--disable-pip-version-check",
                "--prefer-binary",
            ]
            pip_args.extend(_get_pip_ssl_flags())
            pip_args.extend(_get_pip_proxy_args())
            pip_args.extend(batch_specs)
            base_cmd = [python_path, "-m", "pip"] + pip_args

        batch_timeout = 3600

        try:
            result = _run_pip_install(
                cmd=base_cmd,
                timeout=batch_timeout,
                env=env,
                subprocess_kwargs=subprocess_kwargs,
                label="dependencies",
                progress_start=batch_start,
                progress_end=batch_end,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )

            if result.returncode == -1 and "cancelled" in (result.stderr or "").lower():
                return False, "Installation cancelled"

            # Retry on hash mismatch
            if result.returncode != 0:
                error_output = result.stderr or result.stdout or ""
                if _is_hash_mismatch(error_output):
                    _log(
                        "Hash mismatch, retrying batch with --no-cache...",
                        Qgis.MessageLevel.Warning,
                    )
                    nocache_flag = "--no-cache" if use_uv else "--no-cache-dir"
                    result = _run_pip_install(
                        cmd=base_cmd + [nocache_flag],
                        timeout=batch_timeout,
                        env=env,
                        subprocess_kwargs=subprocess_kwargs,
                        label="dependencies (no-cache retry)",
                        progress_start=batch_start,
                        progress_end=batch_end,
                        progress_callback=progress_callback,
                        cancel_check=cancel_check,
                    )

            # Retry on network errors
            if result.returncode != 0:
                error_output = result.stderr or result.stdout or ""
                if _is_network_error(error_output):
                    for attempt in range(1, 3):
                        _log(
                            "Network error, retrying batch in 5s "
                            "(attempt {}/2)...".format(attempt),
                            Qgis.MessageLevel.Warning,
                        )
                        if progress_callback:
                            progress_callback(
                                batch_start,
                                "Network error, retrying...",
                            )
                        time.sleep(5)
                        if cancel_check and cancel_check():
                            return False, "Installation cancelled"
                        result = _run_pip_install(
                            cmd=base_cmd,
                            timeout=batch_timeout,
                            env=env,
                            subprocess_kwargs=subprocess_kwargs,
                            label="dependencies (retry {})".format(attempt),
                            progress_start=batch_start,
                            progress_end=batch_end,
                            progress_callback=progress_callback,
                            cancel_check=cancel_check,
                        )
                        if result.returncode == 0:
                            break

            # If batch failed, check if an optional package caused it
            if result.returncode != 0:
                error_output = result.stderr or result.stdout or ""
                failed_pkg = _classify_batch_error(error_output, batch_specs)
                if failed_pkg and _is_optional_install_package(failed_pkg):
                    _log(
                        "Optional package {} failed in batch; "
                        "retrying without it...".format(failed_pkg),
                        Qgis.MessageLevel.Warning,
                    )
                    retry_specs = [
                        s for s in batch_specs if not s.startswith(failed_pkg)
                    ]
                    if retry_specs:
                        if use_uv:
                            retry_args = [
                                "pip",
                                "install",
                                "--python",
                                python_path,
                                "--upgrade",
                            ]
                            retry_args.extend(_get_uv_ssl_flags())
                            retry_args.extend(retry_specs)
                            retry_cmd = [uv_path] + retry_args
                        else:
                            retry_args = [
                                "install",
                                "--upgrade",
                                "--no-warn-script-location",
                                "--disable-pip-version-check",
                                "--prefer-binary",
                            ]
                            retry_args.extend(_get_pip_ssl_flags())
                            retry_args.extend(_get_pip_proxy_args())
                            retry_args.extend(retry_specs)
                            retry_cmd = [
                                python_path,
                                "-m",
                                "pip",
                            ] + retry_args

                        result = _run_pip_install(
                            cmd=retry_cmd,
                            timeout=batch_timeout,
                            env=env,
                            subprocess_kwargs=subprocess_kwargs,
                            label="dependencies (without {})".format(failed_pkg),
                            progress_start=batch_start,
                            progress_end=batch_end,
                            progress_callback=progress_callback,
                            cancel_check=cancel_check,
                        )

            if result.returncode == 0:
                _log(
                    "Batch install succeeded for all packages",
                    Qgis.MessageLevel.Success,
                )
                if progress_callback:
                    progress_callback(batch_end, "All dependencies installed")
            else:
                error_output = result.stderr or result.stdout or ""
                failed_pkg = (
                    _classify_batch_error(error_output, batch_specs) or "dependencies"
                )
                _log(
                    "Batch install failed: {}".format(error_output[:500]),
                    Qgis.MessageLevel.Critical,
                )
                if _is_ssl_error(error_output):
                    return (
                        False,
                        "Failed to install {}: SSL certificate "
                        "error".format(failed_pkg),
                    )
                if _is_proxy_auth_error(error_output):
                    return (
                        False,
                        "Failed to install {}: proxy authentication "
                        "required".format(failed_pkg),
                    )
                if _is_network_error(error_output):
                    return (
                        False,
                        "Failed to install {}: network error".format(failed_pkg),
                    )
                if _is_antivirus_error(error_output):
                    return (
                        False,
                        "Failed to install {}: blocked by antivirus or "
                        "security policy".format(failed_pkg),
                    )
                if result.returncode is not None and _is_windows_process_crash(
                    result.returncode
                ):
                    return (
                        False,
                        "Failed to install {}: process crashed "
                        "(code {})".format(failed_pkg, result.returncode),
                    )
                return (
                    False,
                    "Failed to install {}: {}".format(failed_pkg, error_output[:200]),
                )

        except subprocess.TimeoutExpired:
            _log("Batch install timed out", Qgis.MessageLevel.Critical)
            return False, "Dependency installation timed out"
        except Exception as e:
            _log(
                "Exception during batch install: {}".format(e),
                Qgis.MessageLevel.Critical,
            )
            return False, "Error installing dependencies: {}".format(str(e)[:200])

    if progress_callback:
        progress_callback(100, "All dependencies installed")

    _log("=" * 50, Qgis.MessageLevel.Success)
    _log("All dependencies installed successfully!", Qgis.MessageLevel.Success)
    _log(f"Virtual environment: {venv_dir}", Qgis.MessageLevel.Success)
    _log("=" * 50, Qgis.MessageLevel.Success)

    if _driver_too_old:
        return True, "All dependencies installed successfully [DRIVER_TOO_OLD]"
    if _cuda_fell_back:
        return True, "All dependencies installed successfully [CUDA_FALLBACK]"
    return True, "All dependencies installed successfully"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _get_verification_timeout(package_name: str) -> int:
    """Get verification timeout for a package.

    Args:
        package_name: The package name.

    Returns:
        Timeout in seconds.
    """
    if package_name in ("torch", "torchvision"):
        return 120
    elif package_name in (
        "segment-geospatial",
        "sam3",
        "transformers",
        "triton-windows",
    ):
        return 120
    else:
        return 60


def _get_verification_code(package_name: str) -> str:
    """Get verification code that tests the package works.

    Args:
        package_name: The package name.

    Returns:
        Python code string to verify the package.
    """
    if package_name == "torch":
        return "import torch; t = torch.tensor([1, 2, 3]); print(t.sum())"
    elif package_name == "torchvision":
        return "import torchvision; print(torchvision.__version__)"
    elif package_name == "segment-geospatial":
        return "import samgeo; print(samgeo.__version__)"
    elif package_name == "sam3":
        return "import sam3; print('ok')"
    elif package_name == "scikit-image":
        return "import skimage; print(skimage.__version__)"
    elif package_name == "scikit-learn":
        return "import sklearn; print(sklearn.__version__)"
    elif package_name == "transformers":
        return "import transformers; print(transformers.__version__)"
    elif package_name == "triton-windows":
        return "import triton; print('ok')"
    else:
        import_name = package_name.replace("-", "_")
        return f"import {import_name}"


def verify_venv(
    venv_dir: str = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Tuple[bool, str]:
    """Verify all required packages are importable in the venv.

    Args:
        venv_dir: Optional venv directory path. Uses VENV_DIR if None.
        progress_callback: Optional function called with (percent, message).

    Returns:
        Tuple of (success, message).
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    if not venv_exists(venv_dir):
        return False, "Virtual environment not found"

    python_path = get_venv_python_path(venv_dir)
    env = _get_clean_env_for_venv()
    subprocess_kwargs = _get_subprocess_kwargs()

    required_packages = _get_required_packages()
    total_packages = len(required_packages)
    optional_failures: List[str] = []
    for i, (package_name, _) in enumerate(required_packages):
        if progress_callback:
            percent = int((i / total_packages) * 100)
            progress_callback(
                percent,
                f"Verifying {package_name}... ({i + 1}/{total_packages})",
            )

        verify_code = _get_verification_code(package_name)
        cmd = [python_path, "-c", verify_code]
        pkg_timeout = _get_verification_timeout(package_name)

        is_optional = sys.platform == "win32" and package_name in (
            "sam3",
            "triton-windows",
        )

        try:
            result = subprocess.run(  # nosec B603
                cmd,
                capture_output=True,
                text=True,
                timeout=pkg_timeout,
                env=env,
                **subprocess_kwargs,
            )

            if result.returncode != 0:
                error_detail = (
                    result.stderr[:300] if result.stderr else result.stdout[:300]
                )
                _log(
                    "Package {} verification failed: {}".format(
                        package_name, error_detail
                    ),
                    Qgis.MessageLevel.Warning,
                )
                if is_optional:
                    _log(
                        "Package {} verification failed but is optional on this "
                        "platform; continuing.".format(package_name),
                        Qgis.MessageLevel.Warning,
                    )
                    optional_failures.append(package_name)
                    continue
                return False, "Package {} is broken: {}".format(
                    package_name, error_detail[:200]
                )

        except subprocess.TimeoutExpired:
            _log(
                "Verification of {} timed out ({}s), retrying...".format(
                    package_name, pkg_timeout
                ),
                Qgis.MessageLevel.Info,
            )
            try:
                result = subprocess.run(  # nosec B603
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=pkg_timeout,
                    env=env,
                    **subprocess_kwargs,
                )
                if result.returncode != 0:
                    error_detail = (
                        result.stderr[:300] if result.stderr else result.stdout[:300]
                    )
                    if is_optional:
                        optional_failures.append(package_name)
                        continue
                    return False, "Package {} is broken: {}".format(
                        package_name, error_detail[:200]
                    )
            except subprocess.TimeoutExpired:
                if is_optional:
                    optional_failures.append(package_name)
                    continue
                return False, "Verification error: {} (timed out)".format(package_name)
            except Exception as e:
                if is_optional:
                    optional_failures.append(package_name)
                    continue
                return False, "Verification error: {} ({})".format(
                    package_name, str(e)[:100]
                )

        except Exception as e:
            _log(
                "Failed to verify {}: {}".format(package_name, str(e)),
                Qgis.MessageLevel.Warning,
            )
            return False, "Verification error: {}".format(package_name)

    if progress_callback:
        progress_callback(100, "Verification complete")

    if optional_failures:
        unique_optional = sorted(set(optional_failures))
        _log(
            "Virtual environment verified with optional package failures: {}".format(
                ", ".join(unique_optional)
            ),
            Qgis.MessageLevel.Warning,
        )
        return (
            True,
            "Virtual environment ready (optional packages unavailable: {})".format(
                ", ".join(unique_optional)
            ),
        )

    _log("Virtual environment verified successfully", Qgis.MessageLevel.Success)
    return True, "Virtual environment ready"


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------


def cleanup_old_venv_directories() -> List[str]:
    """Remove old venv directories that don't match current Python version.

    Returns:
        List of removed directory paths.
    """
    current_venv_name = f"venv_{PYTHON_VERSION}"
    removed = []

    try:
        if not os.path.exists(CACHE_DIR):
            return removed
        for entry in os.listdir(CACHE_DIR):
            entry_cmp = os.path.normcase(entry)
            current_cmp = os.path.normcase(current_venv_name)
            if (
                entry_cmp.startswith(os.path.normcase("venv_py"))
                and entry_cmp != current_cmp
            ):
                old_path = os.path.join(CACHE_DIR, entry)
                if os.path.isdir(old_path):
                    try:
                        shutil.rmtree(old_path)
                        _log(f"Cleaned up old venv: {old_path}", Qgis.MessageLevel.Info)
                        removed.append(old_path)
                    except Exception as e:
                        _log(
                            f"Failed to remove old venv {old_path}: {e}",
                            Qgis.MessageLevel.Warning,
                        )
    except Exception as e:
        _log(f"Error scanning for old venvs: {e}", Qgis.MessageLevel.Warning)

    return removed


# ---------------------------------------------------------------------------
# Quick check & status
# ---------------------------------------------------------------------------


def _quick_check_packages(venv_dir: str = None) -> Tuple[bool, str]:
    """Fast filesystem check that packages exist in site-packages.

    Does NOT spawn subprocesses -- safe for the main thread.

    Args:
        venv_dir: Optional venv directory path. Uses VENV_DIR if None.

    Returns:
        Tuple of (packages_found, message).
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    site_packages = get_venv_site_packages(venv_dir)
    if not os.path.exists(site_packages):
        return False, "site-packages directory not found"

    package_markers = {
        "torch": "torch",
        "torchvision": "torchvision",
        "samgeo": "samgeo",
        "sam3": "sam3",
    }

    for package_name, dir_name in package_markers.items():
        pkg_dir = os.path.join(site_packages, dir_name)
        if not os.path.exists(pkg_dir):
            _log(
                "Quick check: {} not found at {}".format(package_name, pkg_dir),
                Qgis.MessageLevel.Warning,
            )
            return False, "Package {} not found".format(package_name)

    _log(
        "Quick check: all packages found in {}".format(site_packages),
        Qgis.MessageLevel.Info,
    )
    return True, "All packages found"


def get_venv_status() -> Tuple[bool, str]:
    """Get the status of the complete installation.

    Performs a quick filesystem check (no subprocess calls).
    Safe to call from the main thread.

    Returns:
        Tuple of (is_ready, message).
    """
    from .python_manager import get_python_full_version, standalone_python_exists

    if not standalone_python_exists():
        if sys.platform == "win32" and venv_exists():
            pass  # venv was created with QGIS Python fallback
        else:
            _log("get_venv_status: standalone Python not found", Qgis.MessageLevel.Info)
            return False, "Dependencies not installed"

    if not venv_exists():
        _log(f"get_venv_status: venv not found at {VENV_DIR}", Qgis.MessageLevel.Info)
        return False, "Virtual environment not configured"

    is_present, msg = _quick_check_packages()
    if is_present:
        stored_hash = _read_deps_hash()
        current_hash = _compute_deps_hash()
        if stored_hash is not None and stored_hash != current_hash:
            _log(
                "get_venv_status: deps hash mismatch "
                "(stored={}, current={})".format(stored_hash, current_hash),
                Qgis.MessageLevel.Warning,
            )
            return False, "Dependencies need updating"
        if stored_hash is None:
            _log(
                "get_venv_status: no deps hash file, writing current hash",
                Qgis.MessageLevel.Info,
            )
            _write_deps_hash()
        python_version = get_python_full_version()
        _log("get_venv_status: ready (quick check passed)", Qgis.MessageLevel.Success)
        return True, "Ready (Python {})".format(python_version)
    else:
        _log(f"get_venv_status: quick check failed: {msg}", Qgis.MessageLevel.Warning)
        return False, "Virtual environment incomplete: {}".format(msg)


def remove_venv(venv_dir: str = None) -> Tuple[bool, str]:
    """Remove the virtual environment.

    Args:
        venv_dir: Optional venv directory path. Uses VENV_DIR if None.

    Returns:
        Tuple of (success, message).
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    if not os.path.exists(venv_dir):
        return True, "Virtual environment does not exist"

    try:
        shutil.rmtree(venv_dir)
        _log(f"Removed virtual environment: {venv_dir}", Qgis.MessageLevel.Success)
        return True, "Virtual environment removed"
    except Exception as e:
        _log(f"Failed to remove venv: {e}", Qgis.MessageLevel.Warning)
        return False, f"Failed to remove venv: {str(e)[:200]}"


# ---------------------------------------------------------------------------
# Full orchestration
# ---------------------------------------------------------------------------


def create_venv_and_install(
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    cuda_enabled: bool = False,
) -> Tuple[bool, str]:
    """Complete installation: download Python + uv + create venv + install.

    Progress breakdown:
    - 0-10%: Download Python standalone (~50MB)
    - 10-13%: Download uv package installer (~15MB)
    - 13-18%: Create virtual environment
    - 18-95%: Install packages
    - 95-100%: Verify installation

    Args:
        progress_callback: Optional function called with (percent, message).
        cancel_check: Optional function returning True to cancel.
        cuda_enabled: Whether to install CUDA-enabled PyTorch.

    Returns:
        Tuple of (success, message).
    """
    from .python_manager import (
        download_python_standalone,
        get_python_full_version,
        standalone_python_exists,
    )
    from .uv_manager import download_uv
    from .uv_manager import uv_exists as _uv_exists

    _log_system_info()

    # Early check: verify cache directory is writable
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        test_file = os.path.join(CACHE_DIR, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except OSError as e:
        if os.environ.get("SAMGEO_CACHE_DIR") or os.environ.get("SAMGEO_VENV_DIR"):
            hint = f"The SAMGEO_CACHE_DIR is set to: {CACHE_DIR}"
        else:
            hint = (
                "Set the SAMGEO_CACHE_DIR environment variable to a "
                "writable directory before launching QGIS."
            )
        return False, f"Cannot write to installation directory: {CACHE_DIR}\n{hint}"

    removed_venvs = cleanup_old_venv_directories()
    if removed_venvs:
        _log(
            f"Removed {len(removed_venvs)} old venv directories", Qgis.MessageLevel.Info
        )

    # Step 1: Download Python standalone (0-10%)
    if not standalone_python_exists():
        python_version = get_python_full_version()
        _log(
            f"Downloading Python {python_version} standalone...", Qgis.MessageLevel.Info
        )

        def python_progress(percent, msg):
            if progress_callback:
                progress_callback(int(percent * 0.10), msg)

        success, msg = download_python_standalone(
            progress_callback=python_progress,
            cancel_check=cancel_check,
        )

        if not success:
            if sys.platform == "win32":
                qgis_python = _get_qgis_python()
                if qgis_python:
                    _log(
                        "Standalone Python download failed, "
                        "falling back to QGIS Python: {}".format(msg),
                        Qgis.MessageLevel.Warning,
                    )
                    if progress_callback:
                        progress_callback(10, "Using QGIS Python (fallback)...")
                else:
                    return False, f"Failed to download Python: {msg}"
            else:
                return False, f"Failed to download Python: {msg}"

        if cancel_check and cancel_check():
            return False, "Installation cancelled"
    else:
        _log("Python standalone already installed", Qgis.MessageLevel.Info)
        if progress_callback:
            progress_callback(10, "Python standalone ready")

    # Step 1b: Download uv package installer (10-13%)
    if not _uv_exists():
        _log("Downloading uv package installer...", Qgis.MessageLevel.Info)
        if progress_callback:
            progress_callback(10, "Downloading uv package installer...")

        def uv_progress(percent, msg):
            if progress_callback:
                progress_callback(10 + int(percent * 0.03), msg)

        uv_success, uv_msg = download_uv(
            progress_callback=uv_progress,
            cancel_check=cancel_check,
        )
        if not uv_success:
            _log(
                "uv download failed (will use pip instead): {}".format(uv_msg),
                Qgis.MessageLevel.Warning,
            )
        else:
            _log("uv package installer ready", Qgis.MessageLevel.Info)

        if cancel_check and cancel_check():
            return False, "Installation cancelled"
    else:
        _log("uv package installer already installed", Qgis.MessageLevel.Info)
        if progress_callback:
            progress_callback(13, "uv package installer ready")

    # Step 2: Create venv (13-18%)
    if venv_exists():
        _log("Virtual environment already exists", Qgis.MessageLevel.Info)
        if progress_callback:
            progress_callback(18, "Virtual environment ready")
    else:
        success, msg = create_venv(progress_callback=progress_callback)
        if not success:
            return False, msg

        if cancel_check and cancel_check():
            return False, "Installation cancelled"

    # Step 3: Install dependencies (18-95%)
    def deps_progress(percent, msg):
        if progress_callback:
            mapped = 18 + int((percent - 20) * 77 / 80)
            progress_callback(min(mapped, 95), msg)

    success, msg = install_dependencies(
        progress_callback=deps_progress,
        cancel_check=cancel_check,
        cuda_enabled=cuda_enabled,
    )

    if not success:
        return False, msg

    _driver_too_old = "[DRIVER_TOO_OLD]" in msg
    _cuda_fell_back = "[CUDA_FALLBACK]" in msg

    # Step 4: Verify (95-100%)
    def verify_progress(percent: int, msg: str):
        if progress_callback:
            mapped = 95 + int(percent * 0.04)
            progress_callback(min(mapped, 99), msg)

    is_valid, verify_msg = verify_venv(progress_callback=verify_progress)

    if not is_valid and cuda_enabled:
        if _is_torch_related_verify_failure(verify_msg):
            _log(
                "Verification failed with CUDA torch, "
                "falling back to CPU: {}".format(verify_msg),
                Qgis.MessageLevel.Warning,
            )
            _reinstall_cpu_torch(VENV_DIR, progress_callback=progress_callback)
            is_valid, verify_msg = verify_venv(progress_callback=verify_progress)
            _cuda_fell_back = True

    # CUDA smoke test
    _cuda_smoke_failed = False
    if is_valid and cuda_enabled:
        if progress_callback:
            progress_callback(99, "Verifying CUDA functionality...")
        cuda_works = _verify_cuda_in_venv(VENV_DIR)
        if cuda_works and _cuda_fell_back:
            _cuda_fell_back = False
        elif not cuda_works and not _cuda_fell_back:
            _log(
                "CUDA smoke test failed after install.",
                Qgis.MessageLevel.Warning,
            )
            _cuda_smoke_failed = True

    if not is_valid:
        return False, f"Verification failed: {verify_msg}"

    _write_deps_hash()

    if cuda_enabled and not _cuda_fell_back and not _driver_too_old:
        _write_cuda_flag("cuda")
    elif cuda_enabled and _cuda_fell_back:
        _write_cuda_flag("cuda_fallback")
    else:
        _write_cuda_flag("cpu")

    if progress_callback:
        progress_callback(100, "All dependencies installed and verified")

    if _driver_too_old:
        return True, "Virtual environment ready [DRIVER_TOO_OLD]"
    if _cuda_fell_back:
        return True, "Virtual environment ready [CUDA_FALLBACK]"
    if _cuda_smoke_failed:
        return True, "Virtual environment ready [CUDA_VERIFY_FAILED]"
    return True, "Virtual environment ready"
