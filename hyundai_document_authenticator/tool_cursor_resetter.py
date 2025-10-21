"""Cursor IDE reset/cleanup utility.

This script removes Cursor IDE user data and caches across Windows, macOS, and
Linux. It targets common Electron application artifacts that may contain
identifiers and usage traces, such as network caches, cookies, local storage,
GPU caches, shader caches, and the Chromium "Local State" file.

Safety and behavior
- By default, the script performs an interactive confirmation before deletions.
- Use --yes to skip confirmation, and --dry-run to list targets without removal.
- Optionally attempt to terminate running Cursor processes before cleanup.

Scope
- The cleaner focuses on Cursor IDE directories only. It does NOT touch OS-level
  machine identifiers (e.g., Windows machine GUID, Linux /etc/machine-id), since
  those are outside Cursor and application scope.

Usage
  python tool_cursor_resetter.py --dry-run
  python tool_cursor_resetter.py --yes --kill-process

Args
  See argparse flags in main() for full options.

Raises
  This script logs errors and continues best-effort. It does not raise exceptions
  for missing paths.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


def _existing(paths: Iterable[Path]) -> List[Path]:
    """Return only those paths that currently exist.

    Args:
        paths: Candidate paths.

    Returns:
        A list of existing paths.
    """
    return [p for p in paths if p.exists()]


def _electrum_like_artifacts(base: Path) -> List[Path]:
    """Return common Electron/Chromium artifacts under a base directory.

    This includes caches and storage directories that may contain network traces,
    cookies, local storage, and GPU/shader caches. Not all will exist in all
    installations.

    Args:
        base: Base directory typically pointing to the app's local profile root
            (for Windows, often .../AppData/Local/Cursor or .../User Data).

    Returns:
        A list of candidate artifact paths under the base.
    """
    # Include both top-level and Default profile locations (common in Chromium).
    default = base / "Default"
    # Frequently present in Chromium/Electron apps
    candidates = [
        base / "User Data",  # on Windows often the true profile dir
        base / "Local State",
        base / "Code Cache",
        base / "GPUCache",
        base / "Cache",
        base / "Network",
        base / "Service Worker",
        base / "Local Storage",
        base / "IndexedDB",
        base / "DawnCache",
        base / "GrShaderCache",
        base / "ShaderCache",
        base / "SharedStorage",
        base / "blob_storage",
        # Profile-scoped subdirs
        default / "Network",
        default / "Service Worker",
        default / "Local Storage",
        default / "IndexedDB",
        default / "Code Cache",
        default / "GPUCache",
        default / "Cache",
        default / "SharedStorage",
        default / "blob_storage",
        # Cookie DB and session storage files (may or may not exist)
        default / "Network" / "Cookies",
        default / "Network" / "Cookies-journal",
        default / "Sessions",
    ]
    return candidates


def resolve_cursor_paths() -> List[Path]:
    """Resolve platform-specific Cursor IDE data and cache directories.

    The function discovers typical install paths and known Electron cache roots
    for Cursor IDE across Windows, macOS, and Linux.

    Returns:
        List of paths to remove.
    """
    sysname = platform.system()
    home = Path.home()
    paths: List[Path] = []

    if sysname == "Windows":
        roaming = Path(os.getenv("APPDATA", str(home / "AppData" / "Roaming")))
        local = Path(os.getenv("LOCALAPPDATA", str(home / "AppData" / "Local")))
        # Cursor typical locations
        cursor_roaming = roaming / "Cursor"
        cursor_local = local / "Cursor"
        user_data = cursor_local / "User Data"
        # High-level app directories
        paths.extend([cursor_roaming, cursor_local])
        # Electron artifacts from both cursor_local and user_data
        paths.extend(_electrum_like_artifacts(cursor_local))
        paths.extend(_electrum_like_artifacts(user_data))

    elif sysname == "Darwin":  # macOS
        app_support = home / "Library" / "Application Support"
        caches = home / "Library" / "Caches"
        logs = home / "Library" / "Logs"
        cursor_support = app_support / "Cursor"
        cursor_caches = caches / "Cursor"
        cursor_logs = logs / "Cursor"
        paths.extend([cursor_support, cursor_caches, cursor_logs])
        # Electron artifacts under Application Support/Cursor
        paths.extend(_electrum_like_artifacts(cursor_support))

    else:  # Linux and others
        config = home / ".config" / "Cursor"
        cache = home / ".cache" / "Cursor"
        data = home / ".local" / "share" / "Cursor"
        paths.extend([config, cache, data])
        # Electron artifacts under config/data
        paths.extend(_electrum_like_artifacts(config))
        paths.extend(_electrum_like_artifacts(data))

    # Deduplicate while preserving order
    seen: set[str] = set()
    uniq: List[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def safe_remove(path: Path, *, verbose: bool = True) -> None:
    """Remove a file or directory path safely.

    Args:
        path: File or directory to remove.
        verbose: When True, print progress messages.
    """
    try:
        if not path.exists():
            if verbose:
                print(f"SKIP  (missing): {path}")
            return
        # Remove files/symlinks
        if path.is_file() or path.is_symlink():
            try:
                path.unlink(missing_ok=True)  # Python 3.8+: ignore if race conditions
            except TypeError:
                # For Python < 3.8 compatibility
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            if verbose:
                print(f"OK    (file)   : {path}")
            return
        # Remove directories recursively
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=False)
            if verbose:
                print(f"OK    (dir)    : {path}")
    except Exception as exc:
        print(f"FAIL          : {path} -> {exc}")


def kill_cursor_processes(*, verbose: bool = True) -> None:
    """Attempt to terminate running Cursor IDE processes.

    On Windows, calls `taskkill /F /IM Cursor.exe`.
    On macOS, calls `pkill -f Cursor`.
    On Linux, calls `pkill -f cursor`.

    Args:
        verbose: When True, print outcome messages.
    """
    sysname = platform.system()
    try:
        if sysname == "Windows":
            result = subprocess.run(["taskkill", "/F", "/IM", "Cursor.exe"], capture_output=True, text=True)
        elif sysname == "Darwin":
            result = subprocess.run(["pkill", "-f", "Cursor"], capture_output=True, text=True)
        else:
            result = subprocess.run(["pkill", "-f", "cursor"], capture_output=True, text=True)
        if verbose:
            code = result.returncode
            print(f"Kill Cursor processes: rc={code}")
            if result.stdout:
                print(result.stdout.strip())
            if result.stderr and code != 0:
                print(result.stderr.strip())
    except FileNotFoundError:
        if verbose:
            print("Process kill utility not available on this system.")
    except Exception as exc:
        if verbose:
            print(f"Unable to terminate processes: {exc}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the cleaner.

    Returns:
        Configured ArgumentParser instance.
    """
    p = argparse.ArgumentParser(description="Reset/clean Cursor IDE data and caches")
    p.add_argument("--yes", action="store_true", help="Skip confirmation prompt and proceed with deletion")
    p.add_argument("--dry-run", action="store_true", help="List targets without removing anything")
    p.add_argument("--kill-process", action="store_true", help="Attempt to terminate running Cursor processes before cleanup")
    p.add_argument("--quiet", action="store_true", help="Reduce console output (only errors)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Application entry point.

    Args:
        argv: Optional argument vector (defaults to sys.argv[1:]).

    Returns:
        Process exit code (0 for success).
    """
    args = build_arg_parser().parse_args(argv)
    verbose = not args.quiet

    if args.kill_process:
        kill_cursor_processes(verbose=verbose)

    targets = resolve_cursor_paths()
    existing = _existing(targets)

    if not existing:
        if verbose:
            print("No Cursor IDE directories or cache artifacts found.")
        return 0

    print("The following Cursor IDE paths will be cleaned:")
    for p in existing:
        print(f"  - {p}")

    if args.dry_run:
        print("\nDry-run requested; no files will be removed.")
        return 0

    proceed = args.yes
    if not proceed:
        try:
            resp = input("Proceed with permanent deletion? (yes/no): ").strip().lower()
        except KeyboardInterrupt:
            print("\nAborted.")
            return 1
        proceed = resp == "yes"

    if not proceed:
        print("Aborted; no files were changed.")
        return 0

    for p in existing:
        safe_remove(p, verbose=verbose)

    if verbose:
        print("\nCleanup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
