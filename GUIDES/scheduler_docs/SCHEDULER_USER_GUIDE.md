> Refactor Notice (2025-10): Canonical persistent data path changed from `/appdat` to `/appdata`. If you maintain custom volume mounts or paths in .env or scripts, update them accordingly. Only lowercase `appdat` terms were renamed; do not alter `data` or `database` words.

# Scheduler Service: User Guide

Purpose
This service runs hyundai_document_authenticator/doc_image_verifier.py on a configurable schedule with robust logging and process isolation. The project now supports two execution modes from a single image:
- Scheduled Mode (default): Runs via a dedicated scheduler service.
- Manual Mode (on-demand): Runs the CLI for one-off tasks.

What This Service Does
- Launches doc_image_verifier.py as a subprocess using a specified Python interpreter.
- Prevents overlapping runs by default (configurable).
- Writes structured logs to logs/scheduler.log (start/end times, exit codes, stdout, stderr).
- Handles errors defensively and shuts down cleanly on SIGINT/SIGTERM.

Prerequisites
- Host: Python 3.9+ and project requirements, or
- Container: Images built from the provided Dockerfiles.
- The application script at:
  D:\frm_git\hyundai_document_authenticator\hyundai_document_authenticator\doc_image_verifier.py

Host Installation
1) From repository root (D:\frm_git\hyundai_document_authenticator):
   - conda env create -f environment.yml
   - conda activate image-similarity-env
2) Configure environment:
   - copy .env.example .env
   - Edit .env and set:
     - SCHEDULE_INTERVAL_MINUTES=15
     - PYTHON_EXECUTABLE_PATH=C:\\Python311\\python.exe (or your venv)
     - SCRIPT_PATH=D:\\frm_git\\hyundai_document_authenticator\\hyundai_document_authenticator\\doc_image_verifier.py
     - TIMEZONE=local (or UTC, Asia/Kolkata)
     - ALLOW_OVERLAP=false
3) Run locally:
   - python scheduler_service.py

Container Configuration
- Environment variables are read from the compose .env file.
- Recommended .env values inside the container:
  - SCHEDULE_INTERVAL_MINUTES=15
  - TIMEZONE=local
  - ALLOW_OVERLAP=false
  - PYTHON_EXECUTABLE_PATH=/opt/conda/envs/image-similarity-env/bin/python
  - SCRIPT_PATH=/home/appuser/app/hyundai_document_authenticator/doc_image_verifier.py

Start the scheduler in containers (canonical commands)
- Root CPU/Conda (production):
  - docker compose -f docker-compose.conda.yaml up -d app_scheduler
- Root GPU/Conda (production):
  - docker compose -f docker-compose.gpu.conda.yaml up -d app_scheduler
- Root Dev/Conda (live mounts):
  - docker compose -f docker-compose.conda.yaml -f docker-compose.conda.dev.yaml up -d app_scheduler
- Ubuntu backup (production):
  - docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_Ubuntu/docker-compose.yaml up -d app_scheduler
- RHEL backup (production):
  - docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.rhel.yaml up -d app_scheduler
- RHEL GPU backup (production):
  - docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.rhel.gpu.yaml up -d app_scheduler
- Mamba backup GPU (production):
  - docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_Ubuntu_mamba/docker-compose.gpu.mamba.yaml up -d app_scheduler
- Mamba Dev (live mounts):
  - docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_Ubuntu_mamba/docker-compose.mamba.yaml -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_Ubuntu_mamba/docker-compose.mamba.dev.yaml up -d app_scheduler

Manual Mode (on-demand CLI)
Use the cli_runner service and the entrypoint "cli" mode. The CLI prints output and the container is removed afterwards.
- CPU/Conda (root):
  - docker compose -f docker-compose.conda.yaml run --rm cli_runner cli hyundai_document_authenticator/doc_image_verifier.py search-doc --folder ./hyundai_document_authenticator/data_real --top-doc 5 --top-k 5
- GPU/Conda (root):
  - docker compose -f docker-compose.gpu.conda.yaml run --rm cli_runner cli hyundai_document_authenticator/doc_image_verifier.py search-doc --folder ./hyundai_document_authenticator/data_real --top-doc 5 --top-k 5
- Dev/Conda (root):
  - docker compose -f docker-compose.conda.yaml -f docker-compose.conda.dev.yaml run --rm cli_runner cli hyundai_document_authenticator/doc_image_verifier.py build-image-index --folder ./hyundai_document_authenticator/data_real
- Ubuntu backup:
  - docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_Ubuntu/docker-compose.yaml run --rm cli_runner cli hyundai_document_authenticator/doc_image_verifier.py search-doc --folder ./hyundai_document_authenticator/data_real --top-doc 5 --top-k 5
- RHEL backup:
  - docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.rhel.yaml run --rm cli_runner cli hyundai_document_authenticator/doc_image_verifier.py build-image-index --folder ./hyundai_document_authenticator/data_real
- Mamba backup:
  - docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_Ubuntu_mamba/docker-compose.mamba.yaml run --rm cli_runner cli hyundai_document_authenticator/doc_image_verifier.py search-doc --folder ./hyundai_document_authenticator/data_real --top-doc 5 --top-k 5

Notes on file availability inside containers
- The project root is mounted or copied to /home/appuser/app; ensure doc_image_verifier.py exists at SCRIPT_PATH.
- The scheduler reads env only; working directory does not need to match the script location.

Monitoring
- Tail logs from the scheduler service:
  - docker compose -f docker-compose.conda.yaml logs -f app_scheduler
- File-based logs (inside container volume):
  - logs/scheduler.log

Restart and Config Changes
- Restart without config changes:
  - docker compose -f docker-compose.conda.yaml restart app_scheduler
- Apply .env changes (e.g., schedule interval, overlap policy):
  - docker compose -f docker-compose.conda.yaml up -d app_scheduler
- Stop the scheduler service:
  - docker compose -f docker-compose.conda.yaml stop app_scheduler

In-application Controls (no container restart)
- Startup disable via config (requires restart to apply):
  - Set ENABLE_SCHEDULER=false in your .env (default is true)
  - Restart the service to apply: docker compose -f docker-compose.conda.yaml up -d app_scheduler
- Runtime pause/resume without restart:
  - Pause: docker compose -f docker-compose.conda.yaml exec app_scheduler sh -lc 'touch logs/pause.flag'
  - Resume: docker compose -f docker-compose.conda.yaml exec app_scheduler sh -lc 'rm -f logs/pause.flag'
  - In dev (bind-mounted ./logs): create/remove ./logs/pause.flag on the host.

Output Capture Policy
- Control how the scheduler captures the subprocess output of doc_image_verifier.py with SCHEDULER_CAPTURES_OUTPUT.
- Defaults to streaming when unset (recommended to cap memory for verbose output).
  - SCHEDULER_CAPTURES_OUTPUT=true: buffer stdout/stderr and log after completion (equivalent to capture_output=True).
  - SCHEDULER_CAPTURES_OUTPUT=false or unset: stream stdout/stderr line-by-line while running (lower memory; logs appear live).
- Where to view logs:
  - docker compose -f docker-compose.conda.yaml logs -f app_scheduler (console)
  - File-based: logs/scheduler.log inside the container (or mapped volume)

Graceful Shutdown
- Host: Ctrl+C terminates a foreground run.
- Docker: docker stop sends SIGTERM; tini forwards signals; the scheduler exits cleanly.

Troubleshooting
- Configuration error on startup: verify PYTHON_EXECUTABLE_PATH and SCRIPT_PATH refer to real in-container paths.
- Skipped runs: previous run still active and ALLOW_OVERLAP=false.
- No .env variables visible to the CLI: ensure the robust .env loader in doc_image_verifier.py is present (python-dotenv search + fallback), and that compose sets env_file: .env.
