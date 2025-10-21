"""Comprehensive logging utilities for the Image Similarity System.

This module centralizes logging setup for the application, providing a
production-ready configuration with:

- Rotating file logging (daily rotation, UTF-8, backup retention)
- Optional colored console output for readability
- Optional content-based filtering of log messages
- Integration with the Hugging Face ``transformers`` logger
- Periodic cleanup of older log files beyond the rotation window
- Basic third-party logger tuning (e.g., httpx via config)

Additionally, it provides a minimal, dependency-free JSON Lines helper to
append audit-style records (e.g., failed key-driven API filename requests)
under the app-level logs directory. This is designed to be resilient: write
errors are handled gracefully and will not impact the main pipeline.

All functions are fully typed and documented with Google-style docstrings.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, Set

from .constants import (
    DEFAULT_LOG_FOLDER,
    DEFAULT_LOG_FILENAME,
    DEFAULT_LOG_LEVEL,
)
from .utils import maintain_log_files

# Module-level logger for internal diagnostics
LOGGER = logging.getLogger(__name__)

# Global de-duplication set for once-per-message filtering
_SEEN_ONCE_MESSAGES: Set[str] = set()


class CustomFilter(logging.Filter):
    """Filter to suppress log messages that contain a given substring.

    When attached to a handler, this filter blocks records whose text contains
    ``filter_key``. When ``filter_key`` is None, the filter allows all messages.

    Args:
        filter_key (Optional[str]): Substring to block in log messages. If None,
            no filtering is applied.
    """

    def __init__(self, filter_key: Optional[str] = None) -> None:
        """Initialize the filter with an optional substring to suppress.

        Args:
            filter_key (Optional[str]): If provided, any log record whose message
                contains this substring will be filtered out (suppressed).
        """
        super().__init__()
        self.filter_key: Optional[str] = filter_key
        LOGGER.debug(
            "CustomFilter initialized with filter_key: %s",
            filter_key if filter_key is not None else "<none>",
        )

    def filter(self, record: logging.LogRecord) -> bool:
        """Return True to allow the record; False to suppress it.

        Args:
            record (logging.LogRecord): The log record under evaluation.

        Returns:
            bool: False when ``filter_key`` is set and present in the message;
                True otherwise.
        """
        if self.filter_key is not None and self.filter_key in record.getMessage():
            return False
        return True


class OncePerMessageFilter(logging.Filter):
    """Filter that allows a log message only once per logger and level.

    This filter suppresses subsequent occurrences of the exact same
    (logger.name, levelno, message) tuple across the process lifetime.
    When `allow_loggers` is provided, the filter only applies to those
    top-level logger names (e.g., {"ppocr"}); other loggers are unaffected.

    Args:
        allow_loggers (Optional[Set[str]]): Set of top-level logger names to
            which the filter should apply. When None, applies to all loggers.
    """

    def __init__(self, allow_loggers: Optional[Set[str]] = None) -> None:
        super().__init__()
        self.allow_loggers: Optional[Set[str]] = allow_loggers

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        """Filter that allows a log message only once per logger and level."""
        top_name = (record.name.split(".", 1)[0] if record.name else "")
        if self.allow_loggers is not None and top_name not in self.allow_loggers:
            return True
        key = f"{record.levelno}|{record.name}|{record.getMessage()}"
        if key in _SEEN_ONCE_MESSAGES:
            return False
        _SEEN_ONCE_MESSAGES.add(key)
        return True


class ConsoleLogColors:
    """ANSI escape codes used to colorize console log output.

    Note: Color rendering depends on terminal support. Codes are applied via
    the console formatter when console logging is enabled.
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"  # Reset color/style
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def setup_logging(
    log_folder: str = DEFAULT_LOG_FOLDER,
    log_filename: str = DEFAULT_LOG_FILENAME,
    filter_key: Optional[str] = None,
    log_level: int = DEFAULT_LOG_LEVEL,
    enable_console_logging: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> logging.Logger:
    """Configure a robust logging system for the application.

    This sets up a rotating file handler (daily rotation at midnight, UTF-8),
    an optional colored console handler, attaches an optional message filter,
    integrates with the Hugging Face ``transformers`` logger, and removes
    old log files beyond a 7-day window. It also tunes ``httpx`` logger level
    when provided via the config mapping.

    Args:
        log_folder (str): Directory to store log files.
        log_filename (str): Base filename for the primary log file.
        filter_key (Optional[str]): If provided, messages containing this
            substring are suppressed by a filter attached to handlers.
        log_level (int): Minimum level for the root logger (e.g., logging.INFO).
        enable_console_logging (bool): If True, also log to stdout with color.
        config (Optional[Dict[str, Any]]): Optional configuration mapping used
            to tune third-party loggers (e.g., httpx) with a path like
            config['logging']['loggers']['httpx']['level'].

    Returns:
        logging.Logger: The configured root logger.

    Raises:
        None
    """
    # Resolve log folder from APP_LOG_DIR when present; fallback to provided/default
    env_log_dir = os.getenv("APP_LOG_DIR")
    log_folder_effective = env_log_dir if env_log_dir else log_folder

    # Ensure log folder exists
    log_folder_path = Path(log_folder_effective).resolve()
    try:
        log_folder_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # Fall back: emit to stderr; continue with logging setup (may fail if
        # path remains invalid). This mirrors a resilient posture.
        print(
            f"ERROR: Could not create log folder at {log_folder_path}: {e}.",
            file=sys.stderr,
        )

    # File handler with daily rotation
    log_file_path = log_folder_path / log_filename
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)-30s L%(lineno)-4d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = TimedRotatingFileHandler(
        str(log_file_path), when="midnight", interval=1, backupCount=7, encoding="utf-8"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level)

    # Root logger: clear prior handlers to avoid duplication
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        LOGGER.debug(
            "Root logger already had handlers; clearing for fresh configuration."
        )
        root_logger.handlers.clear()

    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)

    # Attach once-per-message filter for known noisy third-party loggers (e.g., PaddleOCR "ppocr")
    try:
        once_filter = OncePerMessageFilter(allow_loggers={"ppocr"})
        root_logger.addFilter(once_filter)
        # Also attach to the ppocr logger directly and remove its default handlers
        ppocr_logger = logging.getLogger("ppocr")
        try:
            # Clear any existing handlers to avoid duplicate emissions
            for h in list(ppocr_logger.handlers):
                ppocr_logger.removeHandler(h)
        except Exception:
            pass
        try:
            ppocr_logger.addFilter(once_filter)
            # Ensure records propagate to root so our filters/handlers apply
            ppocr_logger.propagate = True
        except Exception:
            pass
    except Exception:
        pass

    # Console handler (optional)
    if enable_console_logging:
        console_formatter = logging.Formatter(
            (
                f"%(asctime)s {ConsoleLogColors.BOLD}[%(name)-25.25s]{ConsoleLogColors.ENDC} "
                f"{ConsoleLogColors.OKGREEN}%(levelname)-8s{ConsoleLogColors.ENDC} "
                f"{ConsoleLogColors.OKCYAN}L%(lineno)-4d{ConsoleLogColors.ENDC}: %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level)

        if filter_key:
            custom_filter = CustomFilter(filter_key)
            file_handler.addFilter(custom_filter)
            console_handler.addFilter(custom_filter)
            LOGGER.debug("Applied CustomFilter to file and console handlers: %s", filter_key)

        root_logger.addHandler(console_handler)
    elif filter_key:
        custom_filter = CustomFilter(filter_key)
        file_handler.addFilter(custom_filter)
        LOGGER.debug("Applied CustomFilter to file handler only (console disabled).")

    # Integrate with Hugging Face transformers logging
    try:
        from transformers import logging as hf_logging  # type: ignore

        if log_level <= logging.DEBUG:
            hf_logging.set_verbosity_debug()
        elif log_level <= logging.INFO:
            hf_logging.set_verbosity_info()
        elif log_level <= logging.WARNING:
            hf_logging.set_verbosity_warning()
        else:
            hf_logging.set_verbosity_error()

        hf_logging.enable_propagation()
        if not enable_console_logging:
            hf_logging.disable_default_handler()
        LOGGER.debug("Configured transformers logging at level %s.", log_level)

        # Reduce configuration noise specifically
        logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)
    except ImportError:
        LOGGER.debug("transformers not installed; skipping its logging configuration.")
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("Could not fully configure transformers logging: %s", e)

    # Config-driven log maintenance (backup + retention)
    try:
        lm_cfg = ((config or {}).get("logging") or {}).get("log_maintenance", {})
        backup_logs = bool(lm_cfg.get("backup_logs", False))
        remove_logs_days = int(lm_cfg.get("remove_logs_days", 7))
        removed = maintain_log_files(
            log_folder_path,
            stem=Path(log_filename).stem,
            remove_logs_days=remove_logs_days,
            backup_logs=backup_logs,
        )
        if removed == 0:
            LOGGER.debug("No old log files removed (backup_logs=%s, remove_logs_days=%s).", backup_logs, remove_logs_days)
    except Exception as e:
        LOGGER.error("Error during log maintenance: %s", e)

    # Tune httpx logger from config when present
    try:
        httpx_level = logging.INFO
        if config:
            level_str = (
                (config.get("logging") or {})
                .get("loggers", {})
                .get("httpx", {})
                .get("level", "INFO")
            )
            httpx_level = getattr(logging, str(level_str).upper(), logging.INFO)
            LOGGER.debug("httpx logger level set from config: %s", level_str)
        logging.getLogger("httpx").setLevel(httpx_level)
    except Exception as e:
        LOGGER.warning("Could not configure httpx logger level: %s", e)

    root_logger.info(
        "Logging initialized. File: %s | Level: %s",
        log_file_path.resolve(),
        logging.getLevelName(root_logger.level),
    )
    return root_logger


# ---------------------------------------------------------------------------
# App-level JSONL helpers (audit-style logs)
# ---------------------------------------------------------------------------

def _app_logs_dir() -> Path:
    """Return the centralized logs directory path for the application.

    Resolution order:
      1) Environment variable APP_LOG_DIR (absolute or relative to CWD)
      2) Package-level default: hyundai_document_authenticator/logs

    Returns:
        Path: Absolute path where application logs should be written.
    """
    env_dir = os.getenv("APP_LOG_DIR")
    if env_dir:
        return Path(env_dir).resolve()
    # Package default fallback
    return Path(__file__).resolve().parents[2] / "logs"


def append_json_line(
    entry: Dict[str, Any],
    *,
    log_filename: str = "failed_requests.log",
    logs_dir: Optional[Path] = None,
) -> bool:
    """Append one JSON-serialized line to a log file under the logs directory.

    The function ensures the directory exists, opens the file in append mode,
    writes a single line with a trailing newline, and closes the file. Any
    error is logged via the module logger and the function returns False.

    Args:
        entry (Dict[str, Any]): JSON-serializable mapping to write as one line.
        log_filename (str): Target filename inside the logs directory.
        logs_dir (Optional[Path]): Override logs directory; default is the
            package-level logs folder.

    Returns:
        bool: True when write succeeds; False on any error.
    """
    try:
        target_dir = logs_dir or _app_logs_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / log_filename
        with file_path.open("a", encoding="utf-8") as fp:
            json.dump(entry, fp, ensure_ascii=False)
            fp.write("\n")
        return True
    except Exception as e:  # noqa: BLE001
        try:
            LOGGER.warning("Failed to append JSON line to %s: %s", log_filename, e)
        except Exception:
            pass
        return False


def log_failed_key_request(
    requested_name: str,
    api_endpoint: str,
    *,
    status_code: Optional[int] = None,
    reason: Optional[str] = None,
    correlation_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    enabled: bool = True,
) -> bool:
    """Append a structured JSON line for a failed key-driven API request.

    Used by the key-input orchestrator/API fetcher to record failures such as
    non-2xx responses, JSON parsing failures, missing payload fields, URL
    download failures, or exceptions during save. The function is defensive:
    it will not raise and will quietly return False when disabled or on error.

    Args:
        requested_name (str): The filename key requested from the API.
        api_endpoint (str): API URL or endpoint string used for the request.
        status_code (Optional[int]): HTTP status code if known; otherwise None.
        reason (Optional[str]): Short reason or exception message.
        correlation_id (Optional[str]): Identifier from response headers or
            constructed; otherwise None.
        context (Optional[Dict[str, Any]]): Additional metadata for the event.
        enabled (bool): When False, the function is a no-op and returns False.

    Returns:
        bool: True if an entry was appended; False if disabled or an error
            occurred while writing.
    """
    if not enabled:
        return False

    payload: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "requested_name": requested_name,
        "api_endpoint": api_endpoint,
        "status_code": status_code,
        "reason": reason,
        "correlation_id": correlation_id,
    }
    if context:
        payload["context"] = context

    return append_json_line(payload, log_filename="failed_requests.log")
