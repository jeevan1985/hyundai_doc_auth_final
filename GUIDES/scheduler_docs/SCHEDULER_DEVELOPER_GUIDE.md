> Refactor Notice (2025-10): Data directory renamed from `/appdat` to `/appdata`. Update any hardcoded paths or env variables in your local scheduler configs/scripts. Only lowercase `appdat` was corrected; do not change unrelated `data` terms.

# Scheduler Service: Developer Guide

## Project Structure
```
.
├── SCHEDULER_DEVELOPER_GUIDE.md
├── SCHEDULER_USER_GUIDE.md
├── requirements.txt
├── scheduler_service.py
├── .env.example
└── hyundai_document_authenticator/
    └── doc_image_verifier.py   # existing self-contained script (unchanged)
```

## Design Overview
- The scheduler is an orchestrator that never imports from `doc_image_verifier.py`. It launches it as a subprocess for strict process isolation.
- APScheduler provides interval scheduling with timezone handling. A background scheduler runs in the same process as the orchestrator.
- Overlap prevention is implemented using both:
  - A global non-blocking thread lock in `_run_script` when `ALLOW_OVERLAP=false` to skip new runs if one is in progress.
  - APScheduler's `max_instances=1` to add an additional guard against concurrent job execution.
- Logging uses `RotatingFileHandler` with a friendly, structured format to capture:
  - Scheduler lifecycle events (start/stop)
  - Job start/end with duration and exit code
  - Subprocess stdout and stderr
  - Unexpected exceptions with stack traces
- Graceful shutdown is achieved via SIGINT/SIGTERM handlers that shutdown the scheduler and signal the main thread to exit.

## Why Subprocess Isolation
- Prevents memory leaks or state contamination in the long-running scheduler process.
- Allows independent resource limits, environment isolation, and easier restarts.
- Capturing stdout/stderr provides a clear operational trail for debugging without entangling the scheduler code with application internals.

## Configuration and Validation
- `.env` is read on startup via `python-dotenv`.
- Paths are validated and resolved to absolute paths. Missing or invalid settings cause a clean startup failure with clear error messages.
- Timezone resolution uses `tzlocal` for system local time fallback and `pytz` for named timezones.

## Scheduling Behavior
- The job is scheduled with the specified minutes interval and `coalesce=True` so missed runs collapse to a single execution.
- `misfire_grace_time=60` provides a small tolerance for brief scheduler pauses.
- When overlap is disabled, any attempt to start a new run while an existing run is active will be skipped with a warning.

## Logging Strategy
- Main logger name: `scheduler`.
- Rotating file logs at `logs/scheduler.log` keep five backups of 5MB each.
- APScheduler log level is reduced to WARNING to minimize noise.

## Signal Handling and Shutdown
- SIGINT and SIGTERM trigger orderly shutdown of APScheduler (`wait=True`) and set an event to stop the main loop.
- On Windows, SIGTERM availability can vary; the code safely ignores if unsupported.

## Extending the Scheduler
- Add pre/post hooks: Wrap `_run_script` to call additional functions before or after the subprocess execution.
- Parameterize subprocess: If future requirements need CLI args, read them from `.env` (e.g., `SCRIPT_ARGS`) and extend the `subprocess.run` call accordingly.
- Alternate triggers: Replace the interval trigger with cron triggers via APScheduler (e.g., run at a specific time each day).
- Health checks: Add a heartbeat file or a small HTTP server endpoint to report scheduler status.

## Coding Standards
- All functions include type hints and Google-style docstrings.
- Non-obvious logic is documented inline, especially around overlap prevention and signal handling.

## Testing Suggestions
- Use a stub script that sleeps for a known duration to test overlap handling.
- Intentionally raise a non-zero exit code in the target script to verify stderr logging and scheduler resilience.
- Adjust `SCHEDULE_INTERVAL_MINUTES=1` for faster test cycles.
