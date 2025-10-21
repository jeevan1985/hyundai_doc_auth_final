> Refactor Notice (2025-10): The canonical data directory has been renamed from `/appdat` to `/appdata`. Update any custom environment files, volume mounts, or scripts accordingly. Only lowercase `appdat` occurrences were changed; unrelated terms like `data`/`database` are unaffected.

# 📘 Comprehensive System Guide: Image Similarity System from Novice to Professional

**Version:** 1.0.1
**Last Updated:** October 2023

![Image Similarity System](https://via.placeholder.com/800x200?text=Image+Similarity+System)  

This guide is your ultimate companion to mastering the Image Similarity System. Crafted with care, it bridges the gap between beginners and experts, offering crystal-clear explanations, step-by-step instructions, and pro tips. Whether you're running scripts locally or in Docker, we've got you covered.

---

## 📑 Table of Contents

- [📘 Comprehensive System Guide: Image Similarity System from Novice to Professional](#-comprehensive-system-guide-image-similarity-system-from-novice-to-professional)
  - [📑 Table of Contents](#-table-of-contents)
  - [1. 🚀 Introduction](#1--introduction)
    - [1.1. 🎯 Purpose of This Guide](#11--purpose-of-this-guide)
    - [1.2. 👥 Who Should Use This Guide](#12--who-should-use-this-guide)
    - [1.3. 🔍 System Overview](#13--system-overview)
    - [1.4. 🛠 Key Components](#14--key-components)
  - [2. 🛫 Getting Started](#2--getting-started)
    - [2.1. 📋 Prerequisites](#21--prerequisites)
    - [2.2. 🖥 Non-Docker Environment Setup](#22--non-docker-environment-setup)
    - [2.3. 🐳 Docker Environment Setup](#23--docker-environment-setup)
    - [2.4. ⚙ Configuration Files](#24--configuration-files)
    - [2.5. 🌍 Environment Variables](#25--environment-variables)
  - [3. 🛠 Core Scripts and Commands](#3--core-scripts-and-commands)
    - [3.1. 🔎 find\_sim\_images.py](#31--find_sim_imagespy)
      - [3.1.1. 🎯 Purpose](#311--purpose)
      - [3.1.2. 📋 Prerequisites](#312--prerequisites)
      - [3.1.3. 🖥 Non-Docker Usage](#313--non-docker-usage)
      - [3.1.4. 🐳 Docker Usage](#314--docker-usage)
      - [3.1.5. ⚠ Troubleshooting](#315--troubleshooting)
    - [3.2. 🌐 find\_sim\_images\_api.py](#32--find_sim_images_apipy)
      - [3.2.1. 🎯 Purpose](#321--purpose)
      - [3.2.2. 📋 Prerequisites](#322--prerequisites)
      - [3.2.3. 🖥 Non-Docker Usage](#323--non-docker-usage)
      - [3.2.4. 🐳 Docker Usage](#324--docker-usage)
      - [3.2.5. ⚠ Troubleshooting](#325--troubleshooting)
    - [3.3. 🔑 generate\_secret\_key.py](#33--generate_secret_keypy)
      - [3.3.1. 🎯 Purpose](#331--purpose)
      - [3.3.2. 📋 Prerequisites](#332--prerequisites)
      - [3.3.3. 🖥 Non-Docker Usage](#333--non-docker-usage)
      - [3.3.4. 🐳 Docker Usage](#334--docker-usage)
      - [3.3.5. ⚠ Troubleshooting](#335--troubleshooting)
    - [3.4. 🧪 professional\_database\_tester.py](#34--professional_database_testerpy)
      - [3.4.1. 🎯 Purpose](#341--purpose)
      - [3.4.2. 📋 Prerequisites](#342--prerequisites)
      - [3.4.3. 🖥 Non-Docker Usage](#343--non-docker-usage)
      - [3.4.4. 🐳 Docker Usage](#344--docker-usage)
      - [3.4.5. ⚠ Troubleshooting](#345--troubleshooting)
    - [3.5. 📦 setup.py](#35--setuppy)
      - [3.5.1. 🎯 Purpose](#351--purpose)
      - [3.5.2. 📋 Prerequisites](#352--prerequisites)
      - [3.5.3. 🖥 Usage](#353--usage)
      - [3.5.4. ⚠ Troubleshooting](#354--troubleshooting)
    - [3.6. 🧪 test\_postgresql.py](#36--test_postgresqlpy)
      - [3.6.1. 🎯 Purpose](#361--purpose)
      - [3.6.2. 📋 Prerequisites](#362--prerequisites)
      - [3.6.3. 🖥 Non-Docker Usage](#363--non-docker-usage)
      - [3.6.4. 🐳 Docker Usage](#364--docker-usage)
      - [3.6.5. ⚠ Troubleshooting](#365--troubleshooting)
  - [4. 🌐 API and Application Management](#4--api-and-application-management)
    - [4.1. 👤 manage\_api\_users.py](#41--manage_api_userspy)
      - [4.1.1. 🎯 Purpose](#411--purpose)
      - [4.1.2. 📋 Prerequisites](#412--prerequisites)
      - [4.1.3. 🖥 Non-Docker Usage](#413--non-docker-usage)
      - [4.1.4. 🐳 Docker Usage](#414--docker-usage)
      - [4.1.5. ⚠ Troubleshooting](#415--troubleshooting)
    - [4.2. 👤 manage\_api\_users\_postgresql.py](#42--manage_api_users_postgresqlpy)
      - [4.2.1. 🎯 Purpose](#421--purpose)
      - [4.2.2. 📋 Prerequisites](#422--prerequisites)
      - [4.2.3. 🖥 Non-Docker Usage](#423--non-docker-usage)
      - [4.2.4. 🐳 Docker Usage](#424--docker-usage)
      - [4.2.5. ⚠ Troubleshooting](#425--troubleshooting)
    - [4.3. 🖥 run\_api\_Server.py](#43--run_api_serverpy)
      - [4.3.1. 🎯 Purpose](#431--purpose)
      - [4.3.2. 📋 Prerequisites](#432--prerequisites)
      - [4.3.3. 🖥 Non-Docker Usage](#433--non-docker-usage)
      - [4.3.4. 🐳 Docker Usage](#434--docker-usage)
      - [4.3.5. ⚠ Troubleshooting](#435--troubleshooting)
    - [4.4. 📱 app.py](#44--apppy)
      - [4.4.1. 🎯 Purpose](#441--purpose)
      - [4.4.2. 📋 Prerequisites](#442--prerequisites)
      - [4.4.3. 🖥 Non-Docker Usage](#443--non-docker-usage)
      - [4.4.4. 🐳 Docker Usage](#444--docker-usage)
      - [4.4.5. ⚠ Troubleshooting](#445--troubleshooting)
  - [5. 🐳 Docker-Specific Operations](#5--docker-specific-operations)
    - [5.1. 📄 Understanding Docker Files](#51--understanding-docker-files)
    - [5.2. ⚖ Development vs. Production Modes](#52--development-vs-production-modes)
    - [5.3. 🛠 Common Docker Commands](#53--common-docker-commands)
      - [Getting an Interactive Development Shell (The "Conda" Method")](#getting-an-interactive-development-shell-the-conda-method)
    - [5.4. 🔄 Docker Lifecycle Management](#54--docker-lifecycle-management)
    - [5.5. ⚠ Docker Troubleshooting](#55--docker-troubleshooting)
  - [6. ⚠ General Troubleshooting and Best Practices](#6--general-troubleshooting-and-best-practices)
    - [6.1. 🐞 Common Errors and Solutions](#61--common-errors-and-solutions)
    - [6.2. 📝 Logging and Monitoring](#62--logging-and-monitoring)
    - [6.3. 💡 Best Practices](#63--best-practices)
    - [6.4. ❓ Frequently Asked Questions (FAQ)](#64--frequently-asked-questions-faq)
  - [7. 📚 Appendix](#7--appendix)
    - [7.1. 📖 Glossary](#71--glossary)
    - [7.2. 🔗 Resources and Further Reading](#72--resources-and-further-reading)
    - [7.3. 📝 License and Credits](#73--license-and-credits)

---

## 1. 🚀 Introduction

Welcome to the **Image Similarity System** – a state-of-the-art platform designed to detect and retrieve visually similar images from vast datasets using advanced AI and vector database technologies. This guide is meticulously crafted to be your go-to resource, ensuring you can harness the full potential of the system regardless of your expertise level.

### 1.1. 🎯 Purpose of This Guide

This document aims to provide an exhaustive, user-friendly manual that covers every aspect of the system. From basic setup to advanced usage, including detailed command examples for both local and Docker environments, troubleshooting tips, and best practices. We've incorporated emojis for visual appeal, structured sections for easy navigation, and comprehensive explanations to make learning enjoyable and effective. Even if you're a complete beginner, you'll find step-by-step instructions; for professionals, there are advanced tips and customization options.

### 1.2. 👥 Who Should Use This Guide

- **🧑‍🎓 Novice Users:** Start with basics like setup and simple commands.
- **🧑‍💻 Intermediate Users:** Dive into script details and Docker integrations.
- **🧑‍🔬 Professional Developers/Administrators:** Explore extensions, troubleshooting, and optimization.

No matter your level, this guide scales with you!

### 1.3. 🔍 System Overview

The system leverages deep learning models (e.g., ResNet) for feature extraction and vector databases (FAISS or Qdrant) for efficient similarity searches. It supports CLI tools, API servers, and GUI applications, all configurable via YAML files. Key features include air-gapped support, batch processing, and scalable deployments.

### 1.4. 🛠 Key Components

- **Core Engine:** Handles feature extraction and searching.
- **CLI Scripts:** For direct command-line interactions.
- **API Server:** For remote access.
- **Databases:** PostgreSQL for metadata, FAISS/Qdrant for vectors.
- **Docker Integration:** For containerized, reproducible environments.

---

## 2. 🛫 Getting Started

Let's set up your environment. Follow these steps carefully to avoid common pitfalls.

### 2.1. 📋 Prerequisites

- **Hardware:** Modern CPU (GPU recommended for speed), at least 8GB RAM.
- **Software:**
  - Python 3.8+ (download from [python.org](https://www.python.org)).
  - pip (comes with Python).
  - Git (for cloning repo).
  - Docker Desktop (for Docker usage) – install from [docker.com](https://www.docker.com).
- **Optional:** NVIDIA GPU with CUDA for accelerated computations.

**Pro Tip:** Use a virtual machine if testing in isolated environments.

### 2.2. 🖥 Non-Docker Environment Setup

1. **Clone the Repository:**
   ```bash
   git clone https://your-repo-url.git
   cd project_folder
   ```
   *Note:* Replace with your actual repo URL and project folder name.

2. **Create and Activate Virtual Environment:**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # Linux/macOS
   myenv\Scripts\activate     # Windows
   ```
   This isolates dependencies.

3. **Install Dependencies:**
   ```bash
   pip install -e .
   ```
   Or use `pip install -r requirements.txt` if setup.py is not used.

4. **Verify Installation:**
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```
   Expect no errors.

**Common Issue:** If modules are missing, re-run pip install.

### 2.3. 🐳 Docker Environment Setup

Docker ensures consistency across machines. We use Docker Compose for multi-service orchestration.

1. **Install Docker:** Follow official docs for your OS.

2. **Build Images:**
   ```bash
   docker compose -f docker-compose.conda.yaml -f docker-compose.conda.dev.yaml build
   ```
   This builds dev images with volume mounts for live code changes.

3. **Start Containers:**
   ```bash
   docker compose -f docker-compose.conda.yaml -f docker-compose.conda.dev.yaml up -d
   ```
   `-d` runs in background.

4. **Verify:**
   ```bash
   docker ps
   ```
   See running services like postgres, api_server, etc.

**Pro Tip:** Use `docker compose -f docker-compose.conda.yaml logs -f` to monitor real-time logs.

### 2.4. ⚙ Configuration Files

Most scripts use `config.yaml` in `configs/`. Example structure:
```yaml
feature_extractor:
  model_name: resnet
vector_database:
  provider: faiss
```
Customize as needed. Always back up before editing.

### 2.5. 🌍 Environment Variables

Set in `.env` file:
```bash
POSTGRES_USER=youruser
POSTGRES_PASSWORD=yourpass
```
Load with `source .env` or let docker-compose handle it.

---

## 3. 🛠 Core Scripts and Commands

Each script is detailed below with purposes, usages, examples, and tips.

### 3.1. 🔎 doc_image_verifier.py (TIF-only CLI)

#### 3.1.1. 🎯 Purpose

This flagship script builds vector indexes from image folders and performs similarity searches. It's the core CLI for image similarity tasks, supporting FAISS or Qdrant backends. Ideal for batch processing or single queries, with options for customization via config or CLI flags.

#### 3.1.2. 📋 Prerequisites

- Configured `config.yaml` with image paths.
- Images in `instance/database_images/`.
- Installed dependencies (torch, faiss, etc.).

#### 3.1.3. 🖥 Non-Docker Usage

**Basic Command Structure:**
```bash
python doc_image_verifier.py --help
```
- TIF batch search:
```bash
python doc_image_verifier.py search-doc --folder ./data_real --top-doc 7 --top-k 5 --aggregation-strategy max
```
- Build index from TIFs:
```bash
python doc_image_verifier.py build-image-index --folder ./data_real --photo-extraction-mode bbox --bbox-list "[[171,236,1480,1100],[171,1168,1480,2032]]" --bbox-format xyxy
```

**Getting Help:**
```bash
python doc_image_verifier.py --help
```
Output: Detailed list of all flags, e.g., `--top_k`, `--query_image_path`.

**Building Index (Example):**
```bash
python doc_image_verifier.py build-image-index --folder ./data_real --force-rebuild-index true
```
- `--force_rebuild_index`: Overwrites existing index.
- Expected Output: Progress logs, e.g., "Indexing 100 images... Done in 5s".

**Performing Search (Single Image):**
```bash
python doc_image_verifier.py search-doc --folder ./data_real --top-k 10 --top-doc 7
```
- Results saved in `instance/search_results/` with JSON summary.

**Batch Search:**
Set `batch_query_image_folder_path` in config or via flag.

**Advanced Options:**
- `--show_config_help`: Displays config explanations.
- Output Interpretation: JSON includes scores, paths; higher score = more similar.

**Example Output Snippet:**
```
Search completed. Results in instance/search_results/2023-10-01_12-00/
Top match: score 0.95, path: database_images/cat.jpg
```

#### 3.1.4. 🐳 Docker Usage

Use `cli_runner` service for one-off commands.

**Getting Help:**
```bash
docker-compose -f docker-compose.yaml -f docker-compose.dev.yaml run --rm cli_runner cli doc_image_verifier.py search-doc --help
```
```bash
docker-compose -f docker-compose.yaml -f docker-compose.dev.yaml run --rm cli_runner cli doc_image_verifier.py build-image-index --help
```

**Building Index in Dev Mode:**
```bash
docker-compose -f docker-compose.yaml -f docker-compose.dev.yaml run --rm cli_runner cli doc_image_verifier.py build-image-index --folder /home/appuser/app/data_real
```
Note: Paths are container-internal, e.g., `/app/`.

**Production Mode:**
Use `-f docker-compose.yaml` without `.dev` for optimized builds.

**Batch Search Example:**
```bash
docker-compose run --rm cli_runner cli doc_image_verifier.py search-doc --folder /home/appuser/app/data_real --top-k 5 --top-doc 7
```

**Tips:** Mount volumes for persistent data: `-v ./instance:/app/instance`.

#### 3.1.5. ⚠ Troubleshooting

- **Error: Index not found** – Run build_index first.
- **Slow performance** – Switch to GPU device in config.
- **FileNotFound** – Check image paths; use absolute paths.
- **Pro Tip:** Enable verbose logging with `--verbose` if available.

### 3.2. 🌐 find_sim_images_api.py

#### 3.2.1. 🎯 Purpose

This script exposes similarity search functionality via API endpoints. It's integrated into the API server for remote queries, supporting POST requests with image data. Perfect for integrating into web apps or microservices.

#### 3.2.2. 📋 Prerequisites

- Running API server.
- Valid config with vector database set.
- API key if authentication is enabled.

#### 3.2.3. 🖥 Non-Docker Usage

Typically not run standalone; called by run_api_Server.py. For testing:
```bash
python find_sim_images_api.py --test_mode
```
(If implemented; otherwise, use API calls via curl/postman.)

**Example API Call (using curl):**
```bash
curl -X POST http://localhost:5000/search -F 'image=@/path/to/image.jpg' -H 'Authorization: Bearer yourkey'
```
Output: JSON with similar images and scores.

**Detailed Explanation:** The script loads the index, extracts features from uploaded image, queries the database, and returns ranked results.

#### 3.2.4. 🐳 Docker Usage

Integrated into `api_server` service.

**Start Server:**
```bash
docker-compose up -d api_server
```

**API Call Example:** Same as non-Docker, but use container IP if needed.

**Dev Mode:** Changes to script reflect immediately due to volume mounts.

#### 3.2.5. ⚠ Troubleshooting

- **401 Unauthorized** – Check API key generation (see generate_secret_key.py).
- **500 Error** – Verify index exists; logs in container via `docker logs`.
- **Pro Tip:** Use Postman for testing complex payloads.

### 3.3. 🔑 generate_secret_key.py

#### 3.3.1. 🎯 Purpose

Generates secure secret keys for API authentication or encryption. Essential for securing the API server against unauthorized access.

#### 3.3.2. 📋 Prerequisites

- Python environment ready.
- No additional configs needed.

#### 3.3.3. 🖥 Non-Docker Usage

**Basic Run:**
```bash
python generate_secret_key.py
```
Output: A random string, e.g., "xYz123AbC".

**With Options (if implemented):**
```bash
python generate_secret_key.py --length 32 --output file.txt
```
Saves to file for easy copy-paste into .env.

**Explanation:** Uses cryptographically secure random generation.

#### 3.3.4. 🐳 Docker Usage

```bash
docker-compose run --rm cli_runner cli generate_secret_key.py
```
Output appears in terminal; copy for use.

#### 3.3.5. ⚠ Troubleshooting

- **No output** – Ensure script is executable; check Python version.
- **Pro Tip:** Generate multiple and store securely.

### 3.4. 🧪 professional_database_tester.py

#### 3.4.1. 🎯 Purpose

Performs advanced testing on PostgreSQL databases, including connection checks, query performance, and integrity tests. Designed for professional validation in production setups.

#### 3.4.2. 📋 Prerequisites

- PostgreSQL running (local or container).
- Credentials in .env or config.

#### 3.4.3. 🖥 Non-Docker Usage

**Run Tests:**
```bash
python professional_database_tester.py --host localhost --port 5432 --user youruser --password yourpass --database yourdb
```
Output: Detailed report, e.g., "Connection successful. Query time: 0.5s".

**Full Suite:**
```bash
python professional_database_tester.py --run_all_tests
```
Includes stress tests, schema validation.

**Explanation:** Connects, runs queries, measures metrics, reports anomalies.

#### 3.4.4. 🐳 Docker Usage

```bash
docker-compose run --rm cli_runner cli professional_database_tester.py --host postgres --user $POSTGRES_USER --password $POSTGRES_PASSWORD
```
Uses environment vars for seamless integration.

#### 3.4.5. ⚠ Troubleshooting

- **Connection failed** – Check if DB is running; verify credentials.
- **Timeout** – Increase timeout flag if available.
- **Pro Tip:** Run in verbose mode for detailed logs.

### 3.5. 📦 setup.py

#### 3.5.1. 🎯 Purpose

Installs the project as a Python package, managing dependencies and making scripts executable system-wide.

#### 3.5.2. 📋 Prerequisites

- Python and pip installed.

#### 3.5.3. 🖥 Usage

**Editable Install:**
```bash
pip install -e .
```
Allows development changes without reinstall.

**Standard Install:**
```bash
python setup.py install
```

**Explanation:** Reads requirements, sets up entry points for CLI scripts.

#### 3.5.4. ⚠ Troubleshooting

- **Dependency errors** – Run `pip check` to verify.
- **Pro Tip:** Use virtualenvs to avoid conflicts.

### 3.6. 🧪 test_postgresql.py

#### 3.6.1. 🎯 Purpose

Basic connection test for PostgreSQL, verifying setup before advanced operations.

#### 3.6.2. 📋 Prerequisites

- DB credentials.

#### 3.6.3. 🖥 Non-Docker Usage

```bash
python test_postgresql.py --host localhost --user youruser --password yourpass
```
Output: "Connection successful!" or error.

#### 3.6.4. 🐳 Docker Usage

```bash
docker-compose run --rm cli_runner cli test_postgresql.py --host postgres
```

#### 3.6.5. ⚠ Troubleshooting

- **Auth error** – Double-check password.
- **Pro Tip:** Combine with professional tester for full validation.

---

## 4. 🌐 API and Application Management

This section covers tools for managing APIs and the GUI app.

### 4.1. 👤 manage_api_users.py

#### 4.1.1. 🎯 Purpose

Manages user accounts for API authentication using SQLite. Supports add, delete, list, update operations.

#### 4.1.2. 📋 Prerequisites

- SQLite DB file (e.g., users.db).

#### 4.1.3. 🖥 Non-Docker Usage

**Help:**
```bash
python manage_api_users.py --help
```

**Add User:**
```bash
python manage_api_users.py add --username admin --password secret --role admin
```

**List Users:**
```bash
python manage_api_users.py list
```
Output: Table of users.

**Delete User:**
```bash
python manage_api_users.py delete --username admin
```

**Update Password:**
```bash
python manage_api_users.py update --username admin --new_password newsecret
```

**Explanation:** Hashes passwords, stores in DB for secure auth.

#### 4.1.4. 🐳 Docker Usage

```bash
docker-compose run --rm cli_runner manage api_server.manage_api_users add --username admin --password secret
```

#### 4.1.5. ⚠ Troubleshooting

- **DB locked** – Close other connections.
- **Pro Tip:** Backup DB before operations.

### 4.2. 👤 manage_api_users_postgresql.py

#### 4.2.1. 🎯 Purpose

Similar to above but for PostgreSQL, offering scalable user management.

#### 4.2.2. 📋 Prerequisites

- PostgreSQL connection details.

#### 4.2.3. 🖥 Non-Docker Usage

**Add User:**
```bash
python manage_api_users_postgresql.py add --username user --password pass --db_host localhost
```

**List:**
```bash
python manage_api_users_postgresql.py list --db_host localhost
```

#### 4.2.4. 🐳 Docker Usage

```bash
docker-compose run --rm cli_runner manage api_server.manage_api_users_postgresql add --username user --password pass --db_host postgres
```

#### 4.2.5. ⚠ Troubleshooting

- **Schema mismatch** – Ensure tables exist; run init scripts.

### 4.3. 🖥 run_api_Server.py

#### 4.3.1. 🎯 Purpose

Launches the API server, hosting endpoints for similarity searches and more.

#### 4.3.2. 📋 Prerequisites

- Configured API users.
- Index built.

#### 4.3.3. 🖥 Non-Docker Usage

```bash
python run_api_Server.py --host 0.0.0.0 --port 5000 --debug
```
Access at http://localhost:5000.

**Explanation:** Uses Flask or similar to serve routes like /search.

#### 4.3.4. 🐳 Docker Usage

```bash
docker-compose up -d api_server
```

#### 4.3.5. ⚠ Troubleshooting

- **Port in use** – Change port or kill process.
- **Pro Tip:** Use gunicorn for production.

### 4.4. 📱 app.py

#### 4.4.1. 🎯 Purpose

Main entry for the GUI application, providing a web-based interface for searches.

#### 4.4.2. 📋 Prerequisites

- Flask installed.

#### 4.4.3. 🖥 Non-Docker Usage

```bash
flask run --app app.py --debug
```
Open browser to localhost:5000.

#### 4.4.4. 🐳 Docker Usage

```bash
docker-compose up -d web
```

#### 4.4.5. ⚠ Troubleshooting

- **Template not found** – Check gui_app/ templates folder.

---

## 5. 🐳 Docker-Specific Operations

### 5.1. 📄 Understanding Docker Files

- **Dockerfile:** Base image with prod setup.
- **Dockerfile.dev:** Adds dev tools, volume mounts.
- **docker-compose.yaml:** Prod config.
- **docker-compose.dev.yaml:** Dev overrides.

### 5.2. ⚖ Development vs. Production Modes

- **Dev:** Hot-reloading, debugging enabled.
- **Prod:** Optimized, no mounts.
Switch with compose file combinations.

### 5.3. 🛠 Common Docker Commands

- Build: `docker-compose build`
- Up: `docker-compose up -d`
- Down: `docker-compose down -v` (removes volumes)
- Exec: `docker-compose exec cli_runner bash`
- Logs: `docker-compose logs -f`

#### Getting an Interactive Development Shell (The "Conda" Method")

For many development tasks, you'll want an interactive shell inside the container to run multiple commands, explore the filesystem, or debug your code. This is like activating a Conda environment. The `cli_runner` service is designed for exactly this.

**To start a new, clean, interactive session:**

```bash
docker-compose -f docker-compose.yaml -f docker-compose.dev.yaml run --rm cli_runner
```

**What this command does:**

- It starts a new, temporary container using your development image (which has all libraries installed).
- It mounts your local project folder for live code syncing.
- It drops you directly into a `bash` shell inside the container.
- When you type `exit`, the container is automatically removed (--rm), keeping your system clean.

**When to use `docker-compose run` vs. `docker-compose exec`:**

- Use **`docker-compose run`** to start a **new, temporary container** for an interactive session or a one-off script. This is the most common method for your workflow.
- Use **`docker-compose exec`** (which you already have in your guide) to get a shell inside a **service that is already running in the background** (e.g., a web server you started with `docker-compose up -d`).

### 5.4. 🔄 Docker Lifecycle Management

1. Build -> Up -> Run commands -> Monitor logs -> Down.
For updates: `docker-compose pull && docker-compose up -d --build`

### 5.5. ⚠ Docker Troubleshooting

- **Volume issues** – Check mounts in compose files.
- **Network errors** – Use `docker network inspect`.
- **Pro Tip:** Use `docker system prune` to clean up.

---

## 6. ⚠ General Troubleshooting and Best Practices

### 6.1. 🐞 Common Errors and Solutions

- **ModuleNotFoundError:** Reinstall dependencies.
- **ConnectionRefusedError:** Verify service is up (e.g., postgres).
- **Out of Memory:** Increase Docker resources or optimize queries.

### 6.2. 📝 Logging and Monitoring

Logs in `logs/` or Docker console. Use tools like ELK stack for advanced monitoring.

### 6.3. 💡 Best Practices

- Always use virtualenvs/Docker for isolation.
- Backup data before destructive operations.
- Test in dev before prod.
- Keep configs version-controlled.

### 6.4. ❓ Frequently Asked Questions (FAQ)

- **Q: How to switch databases?** A: Edit config.yaml provider.
- **Q: GPU support?** A: Set device: cuda in config.
- **Q: Scaling?** A: Use Qdrant for distributed setups.

---

## 7. 📚 Appendix

### 7.1. 📖 Glossary

- **FAISS:** Facebook AI Similarity Search.
- **Qdrant:** Vector database for embeddings.
- **Docker Compose:** Tool for multi-container apps.

### 7.2. 🔗 Resources and Further Reading

- Official Docs: [FAISS](https://github.com/facebookresearch/faiss), [Qdrant](https://qdrant.tech).
- Tutorials: Docker basics on docker.com.

### 7.3. 📝 License and Credits

This system is open-source under MIT License. Credits to JK for development.

---

Thank you for using this guide! If you have feedback, contribute to the repo. 🚀