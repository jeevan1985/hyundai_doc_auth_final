#!/usr/bin/env python3
"""Scheduler service for orchestrating execution of `doc_image_verifier.py`.

This module provides a production-ready scheduler layer that launches the
existing `doc_image_verifier.py` script as a separate subprocess on a
configurable interval. It emphasizes isolation, robust logging, configuration
via `.env`, error handling, and graceful shutdown.

Key design points:
- Runs the target script using a separate subprocess for isolation.
- Uses APScheduler for interval-based scheduling with timezone support.
- Prevents overlapping runs by default (configurable).
- Structured rotating-file logging capturing stdout, stderr, and exit codes.
- Graceful handling of SIGINT/SIGTERM for clean shutdowns.

Environment variables (via `.env`):
- SCHEDULE_INTERVAL_MINUTES: int, required. Interval between runs in minutes.
- PYTHON_EXECUTABLE_PATH: str, required. Absolute path to Python interpreter.
- SCRIPT_PATH: str, required. Absolute path to `doc_image_verifier.py`.
- TIMEZONE: str, optional. IANA timezone name (e.g., "UTC", "Asia/Kolkata").
            Use "local" to use the system local timezone. Defaults to "local".
- ALLOW_OVERLAP: bool, optional (true/false). Whether to allow overlapping
                 runs. Defaults to false.
- ENABLE_SCHEDULER: bool, optional (true/false). Startup gate; when false, the service logs and exits. Defaults to true.
- SCHEDULER_CAPTURES_OUTPUT: bool, optional (true/false). If true, buffer subprocess stdout/stderr and log at end.
                             If false or missing, stream subprocess stdout/stderr line-by-line to reduce memory.
- SCRIPT_ARGS: str, optional. Quoted argument string to pass to the script
               (e.g., 'search-doc --top-k 5'). If empty and the script is
               doc_image_verifier.py, defaults to 'search-doc'.

Usage:
    python scheduler_service.py

This script will read the configuration, start the scheduler, and run until a
termination signal is received.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
import signal
import subprocess
import shlex
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Final, Optional, List

import pytz
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, JobExecutionEvent
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from tzlocal import get_localzone_name

# Public API of this module.
__all__ = [
    "Config",
    "load_config",
    "setup_logging",
    "main",
]

# Logging constants configured for sensible defaults in most environments.
LOG_MAX_BYTES: Final[int] = 5_000_000
LOG_BACKUP_COUNT: Final[int] = 5

# Job identity used across the module to avoid string duplication.
DEFAULT_JOB_ID: Final[str] = "doc_image_verifier_job"


@dataclass(frozen=True)
class Config:
    """Immutable configuration for the scheduler service.

    Attributes:
        schedule_interval_minutes: Interval between each run in minutes.
        python_executable_path: Absolute path to the Python interpreter used to
            execute the target script.
        script_path: Absolute path to `doc_image_verifier.py`.
        timezone: Timezone to use for the scheduler (IANA name or "local").
        allow_overlap: Whether to allow job runs to overlap.
        enable_scheduler: Startup gate. When false, the service logs and exits without starting.
        log_dir: Directory where log files are stored.
        log_file: Full path to the scheduler log file.
        pause_flag_file: Path to a flag file that, when present, pauses job execution (skip runs).
        scheduler_captures_output: When True, buffer subprocess output; when False, stream line-by-line to reduce memory.
        script_args: Arguments to pass to the script (e.g., subcommand and options).
    """

    schedule_interval_minutes: int
    python_executable_path: str
    script_path: str
    timezone: str
    allow_overlap: bool
    enable_scheduler: bool
    log_dir: Path
    log_file: Path
    pause_flag_file: Path
    scheduler_captures_output: bool
    script_args: List[str]


def _str_to_bool(value: str) -> bool:
    """Convert a string to a boolean value.

    Accepts typical truthy/falsey strings (case-insensitive):
    true/false, yes/no, 1/0, y/n, on/off.

    Args:
        value: String representation of a boolean.

    Returns:
        Boolean interpretation of the input.

    Raises:
        ValueError: If the string cannot be interpreted as a boolean.
    """
    truthy = {"true", "1", "yes", "y", "on"}
    falsey = {"false", "0", "no", "n", "off"}
    v = value.strip().lower()
    if v in truthy:
        return True
    if v in falsey:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _resolve_timezone(tz_name: Optional[str]) -> pytz.tzinfo.BaseTzInfo:
    """Resolve a timezone string to a pytz timezone object.

    If tz_name is None or equals "local" (case-insensitive), the system's local
    timezone is used.

    Args:
        tz_name: IANA timezone name (e.g., "UTC", "Asia/Kolkata") or "local".

    Returns:
        A pytz timezone object.

    Raises:
        pytz.UnknownTimeZoneError: If the provided timezone name is invalid.
    """
    if not tz_name or tz_name.strip().lower() == "local":
        local_name = get_localzone_name()
        return pytz.timezone(local_name)
    return pytz.timezone(tz_name)


def load_config() -> Config:
    """Load and validate configuration from environment variables.

    Returns:
        A validated Config instance.

    Raises:
        ValueError: If required variables are missing or invalid.
        FileNotFoundError: If provided file paths do not exist.
    """
    load_dotenv()

    interval_str = os.getenv("SCHEDULE_INTERVAL_MINUTES")
    python_path = os.getenv("PYTHON_EXECUTABLE_PATH")
    script_path = os.getenv("SCRIPT_PATH")
    timezone = os.getenv("TIMEZONE", "local").strip()
    allow_overlap_str = os.getenv("ALLOW_OVERLAP", "false")
    enable_scheduler_str = os.getenv("ENABLE_SCHEDULER", "true")
    captures_output_str = os.getenv("SCHEDULER_CAPTURES_OUTPUT", "false")

    if not interval_str:
        raise ValueError("SCHEDULE_INTERVAL_MINUTES is required in .env")
    try:
        interval = int(interval_str)
        if interval <= 0:
            raise ValueError
    except ValueError as exc:
        raise ValueError(
            "SCHEDULE_INTERVAL_MINUTES must be a positive integer"
        ) from exc

    if not python_path:
        raise ValueError("PYTHON_EXECUTABLE_PATH is required in .env")
    if not script_path:
        raise ValueError("SCRIPT_PATH is required in .env")

    allow_overlap: bool
    try:
        allow_overlap = _str_to_bool(allow_overlap_str)
    except ValueError as exc:
        raise ValueError("ALLOW_OVERLAP must be a boolean (true/false)") from exc

    enable_scheduler: bool
    try:
        enable_scheduler = _str_to_bool(enable_scheduler_str)
    except ValueError as exc:
        raise ValueError("ENABLE_SCHEDULER must be a boolean (true/false)") from exc

    # Validate paths
    py_path = Path(python_path).expanduser().resolve()
    # If a directory was provided for PYTHON_EXECUTABLE_PATH, attempt to resolve the interpreter within it.
    if py_path.is_dir():
        candidate = py_path / ("python.exe" if os.name == "nt" else "bin/python")
        if candidate.exists():
            py_path = candidate.resolve()
    if not py_path.exists() or not py_path.is_file():
        raise FileNotFoundError(
            "Python executable not found. Ensure PYTHON_EXECUTABLE_PATH points to the interpreter binary, "
            f"e.g., 'C\\ProgramData\\anaconda3\\envs\\image-similarity-env\\python.exe' on Windows. Given: {py_path}"
        )

    sc_path = Path(script_path).expanduser().resolve()
    if not sc_path.exists() or not sc_path.is_file():
        raise FileNotFoundError(f"Script not found or not a file: {sc_path}")

    # Parse optional script arguments (subcommand and options)
    args_str = os.getenv("SCRIPT_ARGS", "").strip()
    script_args_list: List[str] = []
    if args_str:
        # Use POSIX mode on non-Windows for typical shell parsing rules
        script_args_list = shlex.split(args_str, posix=(os.name != "nt"))
    else:
        # Provide a safe default for the Typer CLI used by doc_image_verifier.py
        if sc_path.name == "doc_image_verifier.py":
            script_args_list = ["search-doc"]

    # Prepare logging paths honoring APP_LOG_DIR; default to container-friendly path
    env_log_dir = os.getenv("APP_LOG_DIR", "/home/appuser/app/logs")
    log_dir = Path(env_log_dir) / "scheduler"
    log_file = log_dir / "scheduler.log"
    pause_flag_file = log_dir / "pause.flag"

    # Output capture policy: default to streaming (False) to cap memory
    # TODO: Add configuration option for bounded buffer size when capture is true to avoid large memory usage.
    try:
        scheduler_captures_output = _str_to_bool(captures_output_str)
    except ValueError as exc:
        raise ValueError("SCHEDULER_CAPTURES_OUTPUT must be a boolean (true/false)") from exc

    return Config(
        schedule_interval_minutes=interval,
        python_executable_path=str(py_path),
        script_path=str(sc_path),
        timezone=timezone,
        allow_overlap=allow_overlap,
        log_dir=log_dir,
        log_file=log_file,
        enable_scheduler=enable_scheduler,
        pause_flag_file=pause_flag_file,
        scheduler_captures_output=scheduler_captures_output,
        script_args=script_args_list,
    )


def setup_logging(log_dir: Path, log_file: Path) -> logging.Logger:
    """Configure rotating file logging and return the module logger.

    Args:
        log_dir: Directory to write logs into.
        log_file: Full path to the log file.

    Returns:
        Configured logger instance for this module.

    Raises:
        OSError: If the log directory cannot be created.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("scheduler")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)s | %(name)s | "
            "%(threadName)s | %(message)s"
        ),
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    file_handler = RotatingFileHandler(
        filename=str(log_file), maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Avoid duplicate handlers if reconfigured.
    # Note: FileHandler is a subclass of StreamHandler; avoid using a broad
    # isinstance(h, logging.StreamHandler) check which would treat the file
    # handler as a stream handler and suppress console logging.
    has_file_handler = any(
        isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", None) == str(log_file)
        for h in logger.handlers
    )
    if not has_file_handler:
        logger.addHandler(file_handler)

    has_console_handler = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        and getattr(h, "stream", None) is sys.stdout
        for h in logger.handlers
    )
    if not has_console_handler:
        logger.addHandler(console_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("apscheduler").setLevel(logging.WARNING)

    # TODO: Consider offering an optional JSON log formatter for ingestion by log aggregators.
    return logger


# Synchronization primitives to control job overlap and termination.
_run_lock: threading.Lock = threading.Lock()
_stop_event: threading.Event = threading.Event()


def _run_script(logger: logging.Logger, config: Config) -> None:
    """Execute the target script as a subprocess and log full results.

    This function acquires a non-blocking lock when overlap is disallowed to
    ensure that concurrent executions are skipped, not queued.

    Args:
        logger: Logger for writing structured logs.
        config: Validated configuration values.

    Returns:
        None

    Raises:
        None
    """
    # Soft pause: if the pause flag file exists, skip this run without error.
    try:
        if config.pause_flag_file.exists():
            logger.info("Pause flag present at %s; skipping scheduled run", config.pause_flag_file)
            return
    except Exception:
        # Non-fatal: if we cannot stat the pause file, continue normally.
        pass

    if not config.allow_overlap:
        acquired = _run_lock.acquire(blocking=False)
        if not acquired:
            logger.warning(
                "Previous run is still in progress; skipping new run due to ALLOW_OVERLAP=false"
            )
            return

    start_time = datetime.now()
    script_cwd = Path(config.script_path).resolve().parent
    logger.info(
        "Job start | launching subprocess | script=%s | python=%s | args=%s | cwd=%s",
        config.script_path,
        config.python_executable_path,
        " ".join(config.script_args) if config.script_args else "(none)",
        str(script_cwd),
    )

    try:
        if config.scheduler_captures_output:
            # Buffering mode: capture stdout/stderr and log after completion.
            completed = subprocess.run(
                [config.python_executable_path, config.script_path, *config.script_args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(script_cwd),
            )
            retcode = completed.returncode
            duration_sec = (datetime.now() - start_time).total_seconds()
            logger.info("Job end | exit_code=%s | duration_sec=%.3f", retcode, duration_sec)
            if completed.stdout:
                logger.info("Captured stdout:\n%s", completed.stdout.rstrip())
            if completed.stderr:
                level = logging.ERROR if retcode != 0 else logging.WARNING
                logger.log(level, "Captured stderr:\n%s", completed.stderr.rstrip())
        else:
            # Streaming mode: stream stdout/stderr line-by-line to reduce memory usage.
            proc = subprocess.Popen(
                [config.python_executable_path, config.script_path, *config.script_args],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(script_cwd),
            )

            def _stream(pipe: Optional[IO[str]], level: int) -> None:
                # We stream lines to avoid building large in-memory buffers when
                # the child process emits substantial output over long runs.
                assert pipe is not None
                try:
                    for line in iter(pipe.readline, ""):
                        if not line:
                            break
                        logger.log(level, line.rstrip())
                except Exception:
                    logger.exception("Error while streaming subprocess output")
                finally:
                    try:
                        pipe.close()
                    except Exception:
                        pass

            t_out = threading.Thread(target=_stream, args=(proc.stdout, logging.INFO), name="stdout-reader", daemon=True)
            t_err = threading.Thread(target=_stream, args=(proc.stderr, logging.WARNING), name="stderr-reader", daemon=True)
            t_out.start(); t_err.start()
            retcode = proc.wait()
            t_out.join(timeout=2.0); t_err.join(timeout=2.0)
            duration_sec = (datetime.now() - start_time).total_seconds()
            logger.info("Job end | exit_code=%s | duration_sec=%.3f", retcode, duration_sec)

    except PermissionError as exc:
        logger.error(
            "Permission denied when launching subprocess | python=%s | script=%s | error=%s. "
            "Verify that PYTHON_EXECUTABLE_PATH points to the Python executable (e.g., python.exe on Windows) "
            "and that you have permission to execute it.",
            config.python_executable_path,
            config.script_path,
            exc,
        )
    except FileNotFoundError as exc:
        logger.error(
            "Component not found when launching subprocess | python=%s | script=%s | error=%s. "
            "Ensure both paths exist and are accessible.",
            config.python_executable_path,
            config.script_path,
            exc,
        )
    except Exception:
        # Log full traceback for unexpected failures
        logger.exception("Unhandled exception while running subprocess")

    finally:
        if not config.allow_overlap and _run_lock.locked():
            # Only release if we actually hold the lock
            try:
                _run_lock.release()
            except RuntimeError:
                # Defensive: ignore if lock wasn't acquired
                pass


def _job_listener(event: JobExecutionEvent) -> None:
    """Optional APScheduler job listener to hook into job lifecycle events.

    This logs job execution errors that occur outside the job function's try/except.

    Args:
        event: Job execution event from APScheduler.

    Returns:
        None

    Raises:
        None
    """
    logger = logging.getLogger("scheduler")
    if event.exception:
        logger.error(
            "APScheduler reported an error | job_id=%s | exception=%s",
            event.job_id,
            event.exception,
        )


def _install_signal_handlers(logger: logging.Logger, scheduler: BackgroundScheduler) -> None:
    """Register signal handlers for graceful shutdown.

    On receiving SIGINT or SIGTERM, this will stop the scheduler and block until
    it shuts down cleanly, then signal the main thread to exit.

    Args:
        logger: Logger instance for diagnostics.
        scheduler: Running APScheduler instance to be shut down on signal.

    Returns:
        None

    Raises:
        None
    """

    def _handle_signal(signum: int, _frame: object) -> None:
        logger.info("Received signal %s; initiating graceful shutdown", signum)
        try:
            scheduler.shutdown(wait=True)
        except Exception:
            logger.exception("Error during scheduler shutdown")
        finally:
            _stop_event.set()
            _log_shutdown_banner(logger)

    signal.signal(signal.SIGINT, _handle_signal)
    # SIGTERM may not be available on some platforms (e.g., older Windows)
    try:
        signal.signal(signal.SIGTERM, _handle_signal)  # type: ignore[attr-defined]
    except Exception:
        # Not critical; continue without SIGTERM handling on platforms lacking it
        pass


def _log_start_banner(logger: logging.Logger, config: Config, tzinfo: pytz.tzinfo.BaseTzInfo) -> None:
    """Emit a professional, human-readable startup banner.

    The banner provides a concise summary of the scheduler's key runtime
    parameters for operators to confirm configuration at a glance.

    Args:
        logger: Logger to write the banner to.
        config: Validated scheduler configuration.
        tzinfo: Resolved timezone used by the scheduler.

    Returns:
        None

    Raises:
        None
    """
    border = "=" * 72
    args_text = " ".join(config.script_args) if config.script_args else "(none)"
    logger.info(
        "%s\n"
        "�� Scheduler Service Starting\n"
        "   • Interval: %s minute(s)\n"
        "   • Timezone: %s\n"
        "   • Overlap:  %s\n"
        "   • Python:   %s\n"
        "   • Script:   %s\n"
        "   • Args:     %s\n"
        "%s",
        border,
        config.schedule_interval_minutes,
        tzinfo,
        config.allow_overlap,
        config.python_executable_path,
        config.script_path,
        args_text,
        border,
    )


def _log_next_run(logger: logging.Logger, scheduler: BackgroundScheduler, job_id: str = DEFAULT_JOB_ID) -> None:
    """Log the next scheduled run time for the configured job.

    Args:
        logger: Logger instance.
        scheduler: The running APScheduler instance.
        job_id: The identifier of the scheduled job.

    Returns:
        None

    Raises:
        None
    """
    try:
        job = scheduler.get_job(job_id)
        if job and job.next_run_time:
            logger.info("🕒 Next run scheduled at: %s", job.next_run_time)
        else:
            logger.warning("No next run is currently scheduled (job_id=%s)", job_id)
    except Exception:
        logger.exception("Failed to retrieve next run time (job_id=%s)", job_id)


def _log_shutdown_banner(logger: logging.Logger, note: str = "Stopped cleanly") -> None:
    """Emit a user-friendly shutdown banner with a clear status.

    Args:
        logger: Logger to write the banner to.
        note: Short note describing shutdown reason or status.

    Returns:
        None

    Raises:
        None
    """
    border = "=" * 72
    logger.info("%s\n🛑 Scheduler Service %s\n%s", border, note, border)


def main() -> int:
    """Entry point: configure, start the scheduler, and run until stopped.

    Returns:
        Process exit code (0 for normal termination, non-zero for configuration
        or startup failures).

    Raises:
        None
    """
    try:
        config = load_config()
    except Exception as exc:
        # Basic console logging for configuration errors (logger not yet set up)
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    logger = setup_logging(config.log_dir, config.log_file)

    # Startup gate
    if not config.enable_scheduler:
        logger.info("ENABLE_SCHEDULER=false; scheduler disabled by configuration. Exiting without starting.")
        return 0

    # Resolve scheduler timezone
    try:
        tzinfo = _resolve_timezone(config.timezone)
    except Exception as exc:
        logger.error("Invalid TIMEZONE value '%s': %s", config.timezone, exc)
        return 2

    _log_start_banner(logger, config, tzinfo)
    logger.info(
        "Scheduler starting | interval_minutes=%s | timezone=%s | overlap=%s",
        config.schedule_interval_minutes,
        tzinfo,
        config.allow_overlap,
    )

    scheduler = BackgroundScheduler(timezone=tzinfo)

    # Avoid overlapping via both lock and APScheduler max_instances to be safe.
    max_instances = 1 if not config.allow_overlap else 3

    scheduler.add_job(
        _run_script,
        trigger="interval",
        minutes=config.schedule_interval_minutes,
        args=[logger, config],
        id=DEFAULT_JOB_ID,
        coalesce=True,  # Collapse missed runs into one execution
        max_instances=max_instances,
        misfire_grace_time=60,  # seconds; tolerance for minor scheduler delays
        replace_existing=True,
    )

    scheduler.add_listener(_job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

    try:
        scheduler.start()
        logger.info("Scheduler started ✅")
        _install_signal_handlers(logger, scheduler)
        _log_next_run(logger, scheduler)

        # Block main thread while allowing background scheduler to run
        while not _stop_event.is_set():
            _stop_event.wait(timeout=1.0)

    except Exception:
        logger.exception("Fatal error in scheduler runtime loop")
        return 1

    finally:
        # Ensure shutdown if we break out unexpectedly
        if scheduler.running:
            try:
                scheduler.shutdown(wait=True)
            except Exception:
                logger.exception("Error during scheduler shutdown in finally")

        _log_shutdown_banner(logger, note="Exiting")

    return 0


if __name__ == "__main__":
    sys.exit(main())
