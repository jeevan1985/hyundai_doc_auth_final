"""Asset management utilities for Result GUI.

This module provides local-first asset handling for the GUI. It ensures the
presence of required static assets (Bootstrap, Bootstrap Icons, and related
files) under the module's static/ directory structure. If assets are missing
and network connectivity is available, it attempts to download them from known
CDNs. When offline, it fails gracefully without raising exceptions that would
break the application startup.

Design goals:
- Local-first: Prefer locally stored assets under static/.
- Auto-heal: If a required asset is missing and the internet is available,
  download it to static/.
- Air-gapped safe: If offline, do not crash. The UI can still render using
  default theme and any present assets.
- Minimal dependencies: Use Python standard library only.

Example:
    from pathlib import Path
    from assets_manager import ensure_assets

    base_dir = Path(__file__).resolve().parent
    ensure_assets(base_dir / "static")

"""
from __future__ import annotations

import contextlib
import ssl
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen


# CDN sources for fallback downloads. Prefer jsDelivr, then UNPKG.
CDN_BOOTSTRAP_CSS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
CDN_BOOTSTRAP_JS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"
CDN_BOOTSTRAP_ICONS_CSS = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"
CDN_BOOTSTRAP_ICONS_WOFF2 = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/fonts/bootstrap-icons.woff2"
CDN_BOOTSTRAP_ICONS_WOFF = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/fonts/bootstrap-icons.woff"

# Font assets (local theme fonts via @fontsource on jsDelivr)
CDN_FONT_JBM_400 = "https://cdn.jsdelivr.net/npm/@fontsource/jetbrains-mono@5.0.22/files/jetbrains-mono-latin-400-normal.woff2"
CDN_FONT_JBM_700 = "https://cdn.jsdelivr.net/npm/@fontsource/jetbrains-mono@5.0.22/files/jetbrains-mono-latin-700-normal.woff2"
CDN_FONT_ORBITRON_400 = "https://cdn.jsdelivr.net/npm/@fontsource/orbitron@5.0.20/files/orbitron-latin-400-normal.woff2"
CDN_FONT_ORBITRON_700 = "https://cdn.jsdelivr.net/npm/@fontsource/orbitron@5.0.20/files/orbitron-latin-700-normal.woff2"


@dataclass(frozen=True)
class Asset:
    """Descriptor for a required static asset.

    Attributes:
        relpath: Path relative to the static/ directory where this asset should reside.
        url: The CDN URL used to download the asset if it is missing locally.
    """

    relpath: Path
    url: str


def _default_required_assets(static_root: Path) -> List[Asset]:
    """Return the list of required assets with their target relative paths.

    Args:
        static_root: Absolute path to the GUI's static directory.

    Returns:
        A list of Asset descriptors indicating required local files and their
        corresponding CDN URLs.
    """
    return [
        Asset(relpath=Path("vendor/bootstrap/css/bootstrap.min.css"), url=CDN_BOOTSTRAP_CSS),
        Asset(relpath=Path("vendor/bootstrap/js/bootstrap.bundle.min.js"), url=CDN_BOOTSTRAP_JS),
        Asset(relpath=Path("vendor/bootstrap-icons/font/bootstrap-icons.min.css"), url=CDN_BOOTSTRAP_ICONS_CSS),
        Asset(relpath=Path("vendor/bootstrap-icons/font/fonts/bootstrap-icons.woff2"), url=CDN_BOOTSTRAP_ICONS_WOFF2),
        Asset(relpath=Path("vendor/bootstrap-icons/font/fonts/bootstrap-icons.woff"), url=CDN_BOOTSTRAP_ICONS_WOFF),
        # Theme fonts
        Asset(relpath=Path("fonts/jetbrains-mono/jetbrains-mono-latin-400-normal.woff2"), url=CDN_FONT_JBM_400),
        Asset(relpath=Path("fonts/jetbrains-mono/jetbrains-mono-latin-700-normal.woff2"), url=CDN_FONT_JBM_700),
        Asset(relpath=Path("fonts/orbitron/orbitron-latin-400-normal.woff2"), url=CDN_FONT_ORBITRON_400),
        Asset(relpath=Path("fonts/orbitron/orbitron-latin-700-normal.woff2"), url=CDN_FONT_ORBITRON_700),
    ]


def _ensure_directories(static_root: Path, assets: Iterable[Asset]) -> None:
    """Create parent directories for each asset if they do not exist.

    Args:
        static_root: The static root directory.
        assets: Iterable of Asset entries.
    """
    for asset in assets:
        (static_root / asset.relpath).parent.mkdir(parents=True, exist_ok=True)


def _internet_available(url: str = "https://cdn.jsdelivr.net", timeout: float = 2.0) -> bool:
    """Check if outbound network connectivity is available.

    This performs a lightweight HEAD request to a CDN host with a short
    timeout. Any exception results in a False return.

    Args:
        url: URL to probe for connectivity.
        timeout: Timeout in seconds for the probe.

    Returns:
        True if the probe succeeded, False otherwise.
    """
    try:
        req = Request(url, method="HEAD")
        # Some environments require a permissive SSL context
        with contextlib.closing(urlopen(req, timeout=timeout, context=ssl.create_default_context())) as resp:  # type: ignore[call-arg]
            return resp.status < 500  # noqa: PLR2004
    except Exception:
        return False


def _download(url: str, dest: Path, timeout: float = 15.0) -> bool:
    """Download a URL to a destination path.

    Args:
        url: Source URL.
        dest: Destination file path.
        timeout: Timeout in seconds for the download.

    Returns:
        True if the file was downloaded and is non-empty, False otherwise.
    """
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = Request(url, headers={"User-Agent": "ResultGUI/1.0"})
        with contextlib.closing(urlopen(req, timeout=timeout, context=ssl.create_default_context())) as resp:  # type: ignore[call-arg]
            data = resp.read()
        if not data:
            return False
        dest.write_bytes(data)
        return dest.exists() and dest.stat().st_size > 0
    except (HTTPError, URLError, TimeoutError, ssl.SSLError):  # pragma: no cover - environmental
        return False
    except Exception:  # pragma: no cover - defensive catch-all
        return False


def ensure_assets(static_root: Path) -> Dict[str, List[str]]:
    """Ensure required static assets are present, downloading if needed.

    This function is idempotent and safe to call at application startup. It will
    create the vendor directory structure under ``static_root`` as necessary,
    verify the existence of required files, and attempt to download any missing
    files when the internet is available. When offline, it records the missing
    files but does not raise.

    Args:
        static_root: Absolute path to the GUI static/ directory.

    Returns:
        A dictionary with keys:
        - "present": list of assets that were already present
        - "downloaded": list of assets that were downloaded in this run
        - "missing": list of assets still missing (e.g., offline)

    Raises:
        ValueError: If ``static_root`` is not an existing directory.
    """
    if not static_root.exists() or not static_root.is_dir():
        raise ValueError(f"Invalid static_root: {static_root}")

    assets = _default_required_assets(static_root)
    _ensure_directories(static_root, assets)

    present: List[str] = []
    downloaded: List[str] = []
    missing: List[str] = []

    online = _internet_available()

    for asset in assets:
        local_path = static_root / asset.relpath
        if local_path.exists() and local_path.stat().st_size > 0:
            present.append(str(asset.relpath))
            continue
        # Attempt download if we have connectivity
        if online:
            if _download(asset.url, local_path):
                downloaded.append(str(asset.relpath))
            else:
                missing.append(str(asset.relpath))
        else:
            missing.append(str(asset.relpath))

    # Optional: emit a concise, structured log on stdout (non-blocking)
    try:
        summary = {
            "present": present,
            "downloaded": downloaded,
            "missing": missing,
        }
        print(f"[assets] summary={summary}", file=sys.stdout)
    except Exception:
        pass

    return {"present": present, "downloaded": downloaded, "missing": missing}
