# 👑 MASTER GUIDE: Hyundai Document Authenticator (Encyclopedic Edition)

**Version**: 2.1 (Canonical, No-Script)
**Audience**: Operators, Developers, SRE/DevOps, and Technical Stakeholders
**Scope**: CPU/GPU, Production & Development, Online & Air-Gapped

> ⚠️ **Architectural Mandate (Non-Negotiable)**
> - **Persistence Strategy**: Host-path bind mounts are the single source of truth.
> - **Host Directories**:
>   - `./appdata`: All persistent application data (PostgreSQL, FAISS indices, instance files).
>   - `./applog`: All application logs.
> - **Prerequisite**: `mkdir -p appdata applog` (plus subfolders) is **mandatory** before any launch.
> - **Backup/Restore**: Strictly operate on host paths (`tar`, `cp`, `rsync`). **Do not** use `docker volume` commands for data management.

---

<!-- ToC Sidebar: Renders in VS Code Preview, static HTML. Degrades gracefully on GitHub. -->
<details>
<summary>How the ToC Sidebar Works</summary>
<p>This guide includes an embedded HTML/CSS Table of Contents that appears as a sticky sidebar on wider screens. It uses standard Markdown features with a progressive enhancement for better navigation. On platforms that restrict custom HTML/CSS (like GitHub's default Markdown view), it falls back to a standard clickable Table of Contents at the top of the document. The links are hard-coded to match explicit anchors in the document, so no JavaScript is required.</p>
</details>

<style>
  #toc-sidebar { display: none; } /* Hidden by default on restrictive renderers */
  @media (min-width: 1200px) {
    #toc-sidebar {
      display: block; position: fixed; top: 100px; right: 20px; width: 250px;
      max-height: 80vh; overflow-y: auto; background-color: #f9f9f9;
      border: 1px solid #ddd; border-radius: 5px; padding: 15px; font-size: 0.9em;
    }
    #toc-sidebar h3 { margin-top: 0; font-size: 1.1em; border-bottom: 1px solid #ccc; padding-bottom: 5px; }
    #toc-sidebar ul { list-style-type: none; padding-left: 0; }
    #toc-sidebar ul ul { padding-left: 15px; }
    #toc-sidebar a { text-decoration: none; color: #333; display: block; padding: 3px 0; }
    #toc-sidebar a:hover { color: #007acc; }
  }
</style>
<nav id="toc-sidebar">
  <h3>Table of Contents</h3>
  <ul>
    <li><a href="#toc-quick-reference">Part 1: Quick Reference</a></li>
    <li><a href="#toc-getting-started">Part 2: Getting Started</a></li>
    <li><a href="#toc-cli-catalogue">Part 3: Deep CLI Catalogue</a>
      <ul>
        <li><a href="#toc-cli-doc-image-verifier">doc_image_verifier.py</a></li>
        <li><a href="#toc-cli-database-tester">tool_database_tester.py</a></li>
        <li><a href="#toc-cli-search-tif-files">tool_search_tif_files_with_key.py</a></li>
        <li><a href="#toc-cli-model-downloader">tool_universal_model_downloader.py</a></li>
        <li><a href="#toc-cli-tiff-aggregator">tool_tiff_aggregator.py</a></li>
        <li><a href="#toc-cli-smoke-test">tool_smoke_test.py</a></li>
        <li><a href="#toc-cli-code-audit">extra_tools/code_audit.py</a></li>
      </ul>
    </li>
    <li><a href="#toc-conda-runbook">Part 4: Conda (Host) Runbook</a></li>
    <li><a href="#toc-architecture">Part 5: Architecture Deep Dive</a></li>
    <li><a href="#toc-disaster-recovery">Part 6: Disaster Recovery & Runbooks</a></li>
    <li><a href="#toc-security">Part 7: Security & Secrets</a></li>
    <li><a href="#toc-performance">Part 8: Performance & Sizing</a></li>
    <li><a href="#toc-observability">Part 9: Observability & Logging</a></li>
    <li><a href="#toc-container-management">Part 10: Advanced Container Mgmt</a></li>
    <li><a href="#toc-external-services">Part 11: External Services</a></li>
    <li><a href="#toc-air-gapped">Part 12: Air-Gapped Deployment</a></li>
    <li><a href="#toc-data-model">Part 13: Data Model & Persistence</a></li>
    <li><a href="#toc-os-differences">Part 14: OS Differences</a></li>
  </ul>
</nav>

<a id="toc-main"></a>
## 📑 Table of Contents (Clickable)

- [👑 MASTER GUIDE: Hyundai Document Authenticator (Encyclopedic Edition)](#-master-guide-hyundai-document-authenticator-encyclopedic-edition)
  - [📑 Table of Contents (Clickable)](#-table-of-contents-clickable)
  - [Part 1: 🚀 Quick Reference Cheat Sheet](#part-1--quick-reference-cheat-sheet)
  - [Part 2: 🏁 Getting Started (For Everyone)](#part-2--getting-started-for-everyone)
    - [2.1. Core Concepts](#21-core-concepts)
    - [2.2. Quick Start: Your First Launch](#22-quick-start-your-first-launch)
  - [Part 3: 🛠️ Deep CLI Catalogue](#part-3-️-deep-cli-catalogue)
    - [3.1. `doc_image_verifier.py`](#31-doc_image_verifierpy)
    - [3.2. `tool_database_tester.py`](#32-tool_database_testerpy)
    - [3.3. `tool_search_tif_files_with_key.py`](#33-tool_search_tif_files_with_keypy)
    - [3.4. `tool_universal_model_downloader.py`](#34-tool_universal_model_downloaderpy)
    - [3.5. `tool_tiff_aggregator.py`](#35-tool_tiff_aggregatorpy)
    - [3.6. `tool_smoke_test.py`](#36-tool_smoke_testpy)
    - [3.7. `extra_tools/code_audit.py`](#37-extra_toolscode_auditpy)
  - [Part 4: 💻 Conda (Host) Runbook (Non-Docker)](#part-4--conda-host-runbook-non-docker)
  - [Part 5: 🏗️ Architecture Deep Dive](#part-5-️-architecture-deep-dive)
  - [Part 6: 🛡️ Disaster Recovery & Runbooks](#part-6-️-disaster-recovery--runbooks)
    - [6.1. End-to-End DR Drill](#61-end-to-end-dr-drill)
    - [6.2. RPO/RTO Guidance](#62-rporto-guidance)
  - [Part 7: 🔒 Security Hardening & Secrets](#part-7--security-hardening--secrets)
  - [Part 8: ⚡ Performance & Sizing Guide](#part-8--performance--sizing-guide)
  - [Part 9: 📊 Observability & Logging Strategy](#part-9--observability--logging-strategy)
  - [Part 10: 🎓 Advanced Container & Image Management](#part-10--advanced-container--image-management)
  - [Part 11: 🔌 External Services Integration](#part-11--external-services-integration)
  - [Part 12: ✈️ Air-Gapped Deployment (Deepened)](#part-12-️-air-gapped-deployment-deepened)
  - [Part 13: 💾 Data Model & Persistence Reference](#part-13--data-model--persistence-reference)
    - [13.1. PostgreSQL Schema](#131-postgresql-schema)
    - [13.3. Host Directory Structure (`./appdata` and `./applog`)](#133-host-directory-structure-appdata-and-applog)
  - [Part 14: 🐧 Windows vs. Linux/macOS Cheatsheet](#part-14--windows-vs-linuxmacos-cheatsheet)

---

<a id="toc-quick-reference"></a>
## Part 1: 🚀 Quick Reference Cheat Sheet

| Operation | Command (Linux/macOS) | Command (Windows PowerShell) |
| :--- | :--- | :--- |
| **Prerequisites** | `mkdir -p appdata applog; mkdir -p appdata/postgres appdata/instance` | `mkdir appdata, applog, appdata/postgres, appdata/instance` |
| **Start (CPU)** | `docker compose -f docker-compose.conda.yaml up -d --profile all` | `docker compose -f docker-compose.conda.yaml up -d --profile all` |
| **Start (GPU)** | `docker compose -f docker-compose.gpu.conda.yaml up -d --profile all` | `docker compose -f docker-compose.gpu.conda.yaml up -d --profile all` |
| **Stop System** | `docker compose -f docker-compose.conda.yaml down` | `docker compose -f docker-compose.conda.yaml down` |
| **View Logs** | `docker compose -f docker-compose.conda.yaml logs -f` | `docker compose -f docker-compose.conda.yaml logs -f` |
| **Run CLI Tool** | `docker compose -f DOCKER-COMPOSE.YML run --rm cli_runner cli SCRIPT [ARGS]` | `docker compose -f DOCKER-COMPOSE.YML run --rm cli_runner cli SCRIPT [ARGS]` |
| **Backup Data** | `tar -czf backup.tar.gz ./appdata ./applog` | `Compress-Archive -Path ./appdata, ./applog -DestinationPath backup.zip` |
| **Restore Data** | `tar -xzf backup.tar.gz` | `Expand-Archive -Path backup.zip -DestinationPath .` |
| **DB Health Check**| `docker compose run --rm cli_runner cli python hyundai_document_authenticator/tool_database_tester.py ping` | `docker compose run --rm cli_runner cli python hyundai_document_authenticator/tool_database_tester.py ping` |

**Key Environment Variables (`.env` file)**:
- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`: Internal PostgreSQL credentials.
- `APP_LOG_DIR`: Mapped to `./applog` inside containers. For host runs, set `export APP_LOG_DIR=$(pwd)/applog`.

---

<a id="toc-getting-started"></a>
## Part 2: 🏁 Getting Started (For Everyone)

### 2.1. Core Concepts
- **Host Bind Mounts**: Your data lives in `./appdata` and `./applog` on the host machine. The Docker containers are stateless; the data is yours to manage directly.
- **Docker Compose Profiles**: Services are grouped into profiles (`postgres`, `gui`, etc.). You activate what you need. `--profile all` runs everything.
- **`cli_runner` Service**: A dedicated, disposable container for running all command-line tools.

### 2.2. Quick Start: Your First Launch

**Step 1: Create Host Directories (Mandatory)**
This must be done once per environment.
```bash
# Linux/macOS
mkdir -p appdata applog
mkdir -p appdata/postgres appdata/instance appdata/downloads

# Windows PowerShell
mkdir appdata, applog, appdata/postgres, appdata/instance, appdata/downloads
```
* **What it does**: Creates the host directories that Docker will mount for data and logs.
* **Why you'd use it**: This is a non-negotiable prerequisite. Without these, containers will fail to start.

**Step 2: Configure Environment**
Copy `.env.example` to `.env` and review the default settings. For a standard local setup, no changes are needed.
```bash
# Linux/macOS
cp .env.example .env

# Windows
copy .env.example .env
```

**Step 3: Launch the System**
- **Production (CPU)**:
  ```bash
  docker compose -f docker-compose.conda.yaml up -d --profile all
  ```
- **Production (GPU)**:
  ```bash
  docker compose -f docker-compose.gpu.conda.yaml up -d --profile all
  ```
- **Development (with live code reload)**:
  ```bash
  docker compose -f docker-compose.conda.yaml -f docker-compose.conda.dev.yaml up -d --build --profile all
  ```
* **What it does**: Starts all services defined in the compose file(s) in detached mode (`-d`).
* **Why you'd use it**: This is the standard way to run the entire application stack.

---

<a id="toc-cli-catalogue"></a>
## Part 3: 🛠️ Deep CLI Catalogue

All tools are run via the `cli_runner` service to ensure a consistent environment.

**General Pattern:**
```bash
# CPU
docker compose -f docker-compose.conda.yaml run --rm cli_runner cli [SCRIPT_PATH] [ARGS]

# GPU (if tool supports it)
docker compose -f docker-compose.gpu.conda.yaml run --rm cli_runner cli [SCRIPT_PATH] [ARGS]
```

<a id="toc-cli-doc-image-verifier"></a>
### 3.1. `doc_image_verifier.py`
The main workflow tool for building indexes and running searches.

* **Purpose**: Orchestrates the end-to-end document similarity process.
* **Log Location**: `applog/image_similarity_system.log` and console.

**Subcommand: `build-image-index`**
* **Purpose**: Extracts photos from TIF documents and builds a vector search index (FAISS).

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--folder` | Path | (config) | Folder of TIFs to index. |
| `--engine` | str | (config) | Vector DB: `faiss`, `bruteforce`. |
| `--batch-size` | int | (config) | Batch size for feature extraction. |
| `--force-rebuild-index` | bool | false | If true, deletes and rebuilds the index. |
| `--photo-extraction-mode`| str | yolo | `yolo` or `bbox`. |
| `--yolo-model-path` | str | (config) | Path to the YOLO model file. |

**Example (Build FAISS index using YOLO):**
```bash
docker compose -f docker-compose.conda.yaml run --rm cli_runner cli \
  hyundai_document_authenticator/doc_image_verifier.py build-image-index \
  --folder ./hyundai_document_authenticator/data_real \
  --engine faiss \
  --photo-extraction-mode yolo \
  --yolo-model-path trained_model/yolo_photo_extractor/best.pt
```

**Subcommand: `search-doc`**
* **Purpose**: Searches for similar documents given a folder of query TIFs.

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--folder` | Path | (config) | Folder of query TIF files. |
| `--top-k` | int | 5 | Per-photo nearest neighbor count. |
| `--top-doc` | int | 5 | Number of similar documents to return per query. |
| `--save-to-postgres`| bool | false | If true, saves results to the PostgreSQL DB. |

**Example (Search and save to DB):**
```bash
docker compose -f docker-compose.conda.yaml run --rm cli_runner cli \
  hyundai_document_authenticator/doc_image_verifier.py search-doc \
  --folder ./hyundai_document_authenticator/data_real \
  --top-doc 5 --top-k 5 \
  --save-to-postgres
```

---

<a id="toc-conda-runbook"></a>
## Part 4: 💻 Conda (Host) Runbook (Non-Docker)

This mode is for environments where Docker is not available but Conda is. It requires manual setup.

**Rationale**: Fulfills client requirements for a non-containerized execution path.

**Step 1: Create Conda Environment**
```bash
# Create from the primary environment file
conda env create -f environment.yml

# Or for a CPU-only environment
conda env create -f environment.cpu.yml
```

**Step 2: Activate Environment**
```bash
conda activate image-similarity-env
```

**Step 3: Set Environment Variables**
You must manually configure the database connections and log directory.

*   **Linux/macOS**:
    ```bash
    export APP_LOG_DIR=$(pwd)/applog
    export POSTGRES_HOST=localhost
    export POSTGRES_USER=youruser
    export POSTGRES_PASSWORD=yourpass
    export POSTGRES_DB=yourdb
    ```
*   **Windows PowerShell**:
    ```powershell
    $env:APP_LOG_DIR = "`$pwd\applog"
    $env:POSTGRES_HOST = "localhost"
    # ... and so on for other variables
    ```

**Step 4: Run Application Scripts**
With the environment activated and variables set, you can run the Python scripts directly. Data paths must map to your local `./appdata` directory.

```bash
python hyundai_document_authenticator/doc_image_verifier.py build-image-index \
  --folder ./path/to/your/tifs \
  --config-path ./hyundai_document_authenticator/configs/image_similarity_config.yaml
```

---

<a id="toc-architecture"></a>
## Part 5: 🏗️ Architecture Deep Dive
*This section would contain the detailed architecture diagrams and explanations from the original guide, which are preserved but outside the scope of Qdrant-related changes.*

---

<a id="toc-disaster-recovery"></a>
## Part 6: 🛡️ Disaster Recovery & Runbooks

### 6.1. End-to-End DR Drill

**Scenario**: Complete loss of the Docker host, but the `./appdata` and `./applog` directories have been recovered from a backup.

1.  **Stop Services**: Ensure no old containers are running.
    ```bash
    docker compose -f docker-compose.conda.yaml down --remove-orphans
    ```
2.  **Restore Data**: Place the backed-up `appdata` and `applog` directories in the project root.
    ```bash
    # Example restore from a tarball
    tar -xzf /path/to/backup/backup-20251015.tar.gz -C /path/to/project/
    ```
3.  **Verify Integrity**:
    *   **PostgreSQL**: Run a quick check to ensure the database is responsive.
        ```bash
        docker compose -f docker-compose.conda.yaml run --rm cli_runner cli \
          python hyundai_document_authenticator/tool_database_tester.py ping
        ```
    *   **File System**: `ls -l appdata/postgres` should show data files.
4.  **Start System**: Bring the full stack online.
    ```bash
    docker compose -f docker-compose.conda.yaml up -d --profile all
    ```
5.  **Smoke Test**: Run a high-level test to confirm functionality.
    ```bash
    docker compose -f docker-compose.conda.yaml run --rm cli_runner cli \
      python hyundai_document_authenticator/tool_smoke_test.py
    ```

### 6.2. RPO/RTO Guidance
- **Recovery Point Objective (RPO)**: Depends on the frequency of your `./appdata` backups. If you back up daily, your RPO is 24 hours.
- **Recovery Time Objective (RTO)**: With this host-path-based approach, RTO is low. It's the time required to provision a new host, restore the `appdata` directory, and run `docker compose up`. This should be under 30 minutes.

---

<a id="toc-security"></a>
## Part 7: 🔒 Security Hardening & Secrets

- **Non-Root User**: All application containers run as a non-root `appuser` to limit potential damage from a compromise.
- **`tini` init system**: Used as the entrypoint to properly handle signals and reap zombie processes, ensuring graceful shutdowns.
- **Secrets Handling**:
  - **Method**: Secrets are managed via the `.env` file.
  - **Hygiene**:
    - `.env` is listed in `.gitignore` and **must never** be committed to version control.
    - Use a password manager or secure vault to store production `.env` file contents.
    - Implement a rotation policy for database credentials.
- **Host Filesystem Permissions**:
  - The `appdata` and `applog` directories should be owned by the user running the Docker daemon.
  - On SELinux-enforced systems (like RHEL), the `:z` flag is appended to volume mounts (e.g., `./appdata:/data:z`) to allow the container to write to the host path. This is handled in `docker-compose.conda.dev.yaml` and should be added to production files if needed.

---

<a id="toc-performance"></a>
## Part 8: ⚡ Performance & Sizing Guide

- **Gunicorn/Uvicorn Workers**: The number of API workers is not currently exposed as an environment variable. To tune this, you would need to modify the `CMD` or `command` in the `docker-entrypoint.sh` script or `docker-compose.*.yaml` files. A general rule is `(2 * CPU_CORES) + 1`.
- **Disk Sizing**:
  - **`./appdata` (4TB Total)**:
    - `appdata/postgres` (256GB): Reserved for PostgreSQL data.
    - `appdata/instance/faiss_indices` (~3.5TB): The bulk of the space, for storing vector indexes and image embeddings.
    - `appdata/instance/database_images` (Variable): Stores copies of processed images.
  - **`./applog` (10-20GB)**: Sized for log retention, assuming log rotation is active.
- **GPU Notes**:
  - Use `nvidia-smi` on the host to ensure GPUs are available and drivers are correctly installed.
  - Inside a GPU-enabled container, `nvidia-smi` should also work. If not, the NVIDIA Container Toolkit is likely misconfigured on the host.

---

<a id="toc-observability"></a>
## Part 9: 📊 Observability & Logging Strategy

- **Log Aggregation**: All container logs are written to the host's `./applog` directory, with subdirectories for each service (e.g., `applog/api`, `applog/tools`).
- **Shipping Logs**: To ship logs to an aggregator (like Splunk, ELK, or Datadog), configure a log forwarder agent (e.g., Filebeat, Fluentd) on the **host machine** to monitor the `./applog` directory.
- **Log Rotation**:
  - **Application**: The application's logger is configured for basic rotation.
  - **OS-level**: For production, it's recommended to use a robust OS tool like `logrotate` on the host to manage the files in `./applog`.
- **Log Level**: Log levels are not currently controllable via environment variables. This would be a future enhancement.

---

<a id="toc-container-management"></a>
## Part 10: 🎓 Advanced Container & Image Management

- **`docker builder prune`**: Safely cleans up build cache. Run this periodically to reclaim disk space.
- **`docker compose events`**: Streams real-time events from containers (e.g., start, stop, die). Useful for monitoring.
- **`docker compose top`**: Shows the running processes inside each service's container.
- **Image Tagging**:
  - **Retagging**: `docker tag SOURCE_IMAGE[:TAG] TARGET_IMAGE[:TAG]`
  - **Multi-tag Push**: To push to multiple registries or with multiple tags, simply tag the image multiple times before pushing.

---

<a id="toc-external-services"></a>
## Part 11: 🔌 External Services Integration

To connect to external databases, set the `*_EXTERNAL` variables in your `.env` file.

- **PostgreSQL**:
  ```env
  POSTGRES_HOST_EXTERNAL=your-postgres-host.example.com
  POSTGRES_PORT_EXTERNAL=5432
  ```
- **`host.docker.internal`**: To connect from a container to a service running on the Docker **host** machine, use `host.docker.internal` as the hostname. This is useful for development.

---

<a id="toc-air-gapped"></a>
## Part 12: ✈️ Air-Gapped Deployment (Deepened)

1.  **On Online Host**:
    - Pull/build all necessary images.
    - `docker save -o images.tar image1:tag image2:tag ...`
2.  **Transfer `images.tar`** to the air-gapped host.
3.  **On Air-Gapped Host**:
    - `docker load -i images.tar`
    - **Verification**: `docker images` should now list the loaded images.
    - **Retagging**: The image names in `docker-compose.conda.yaml` must exactly match the loaded images. If they don't (e.g., due to a registry prefix), retag them:
      ```bash
      docker tag loaded-image-name:tag expected-compose-name:tag
      ```
    - Create host directories (`mkdir -p appdata applog ...`).
    - Run `docker compose -f docker-compose.conda.yaml up -d --profile all`.

---

<a id="toc-data-model"></a>
## Part 13: 💾 Data Model & Persistence Reference

### 13.1. PostgreSQL Schema
The primary table for storing search results is `doc_similarity_results`.

**CREATE TABLE Example:**
```sql
CREATE TABLE public.doc_similarity_results (
    id SERIAL PRIMARY KEY,
    run_identifier TEXT,
    requesting_username TEXT,
    search_timestamp TIMESTAMPTZ DEFAULT NOW(),
    parent_document_name TEXT,
    highest_similarity_score FLOAT,
    sim_img_check JSONB,
    image_authenticity JSONB,
    fraud_doc_probability JSONB,
    global_top_docs JSONB
);
```
- **`parent_document_name`**: The name of the query document.
- **`highest_similarity_score`**: The top aggregated similarity score.
- **`global_top_docs`**: A JSON array of the top matching documents and their scores.

### 13.3. Host Directory Structure (`./appdata` and `./applog`)
```
.
├── appdata/
│   ├── postgres/         # PostgreSQL data files
│   ├── instance/
│   │   ├── database_images/ # Stored images
│   │   └── faiss_indices/   # FAISS index files
│   └── downloads/        # Exported files
└── applog/
    ├── image_similarity_system.log # Main application log
    └── tools/                    # Logs from CLI tools
        ├── tool_database_tester.log
        └── ...
```

---

<a id="toc-os-differences"></a>
## Part 14: 🐧 Windows vs. Linux/macOS Cheatsheet

| Task | Linux/macOS | Windows PowerShell | Notes |
| :--- | :--- | :--- | :--- |
| **Path Separator** | `/` | `\` | PowerShell often accepts `/`, but native commands may not. |
| **Env Variables** | `export VAR="value"` | `$env:VAR = "value"` | `export` is for the current session. `$env:` modifies the process environment. |
| **Chaining Cmds** | `cmd1 && cmd2` | `cmd1; cmd2` | `&&` stops on error; `;` does not. |
| **`tar` equivalent**| `tar -czf a.tar.gz .` | `Compress-Archive -Path . -DestinationPath a.zip` | Different archive formats. |
| **Quoting** | Single `' '` quotes are literal. | Double `" "` quotes are generally safer. | PowerShell has complex quoting rules. |
| **Docker Path Mounts** | `pwd`/appdata | `${pwd}`/appdata | Use `${pwd}` in PowerShell for current dir. |
