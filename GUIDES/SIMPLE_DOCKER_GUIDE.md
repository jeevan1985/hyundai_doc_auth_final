> Refactor Notice (2025-10): Canonical data directory changed from `/appdat` to `/appdata`. Update any custom volume mounts or scripts accordingly. Only lowercase `appdat` occurrences are affected; `data`/`database` terms are not changed.

---
---

# 📜 The Unified Operations Guide for the JK Image Similarity System

## 🌟 Introduction

Welcome to the definitive operational guide for the JK Image Similarity System. This document is engineered to be your single source of truth for every stage of the application's lifecycle, from initial development to final production deployment.

It is meticulously structured to serve two primary audiences:

1.  **👨‍💻 The Development Team:** A complete handbook for both the daily "hot-reloading" development workflow and the final process of "baking" the application into secure, distributable packages for both **Debian** and **Red Hat Enterprise Linux (RHEL)**.
2.  **🚀 The Client & System Administrator:** A clear, step-by-step installation and management guide for deploying the system in any environment, including secure, offline ("air-gapped") servers on both Debian-based and RHEL-based hosts.

This guide focuses on practical, real-world commands and provides the "why" behind them, empowering you to operate the system with confidence.

## 📋 Table of Contents

*   **Part 1: The Developer's Handbook: From Code to Package**
    *   [1.1. Prerequisite: Setting Up Your Development Environment](#11-prerequisite-setting-up-your-development-environment)
    *   [1.2. Workflow A: The Live-Reload Development Loop (For Daily Coding)](#12-workflow-a-the-live-reload-development-loop-for-daily-coding)
    *   [1.3. Workflow B: "Baking" the Final Production Package](#13-workflow-b-baking-the-final-production-package)

*   **Part 2: The Client's Deployment & Management Guide**
    *   [2.1. Prerequisites for Deployment](#21-prerequisites-for-deployment)
    *   [2.2. Step 1: Load the Docker Images](#22-step-1-load-the-docker-images)
    *   [2.3. Step 2: Prepare the Configuration File (`.env`)](#23-step-2-prepare-the-configuration-file-env)
    *   [2.4. Step 3: Choose Your Architecture & Start the System](#24-step-3-choose-your-architecture--start-the-system)
    *   [2.5. Step 4: Verify the System is Running](#25-step-4-verify-the-system-is-running)
    *   [2.6. Step 5: Access the Services](#26-step-5-access-the-services)
    *   [2.7. Step 6: Running Command-Line Tools (CLI)](#27-step-6-running-command-line-tools-cli)
    *   [2.8. Step 7: Stopping the System](#28-step-7-stopping-the-system)
    *   [2.9. Step 8: Updating the Application](#29-step-8-updating-the-application)

---

## Part 1: The Developer's Handbook: From Code to Package

This section is for the software development team. It covers the complete lifecycle from writing code to packaging the final product for delivery.

### 1.1. Prerequisite: Setting Up Your Development Environment

Ensure your development machine has Docker Engine, Docker Compose, a text editor/IDE, and Git.

### 1.2. Workflow A: The Live-Reload Development Loop (For Daily Coding)

This workflow is your daily driver. It creates a fast feedback loop, allowing you to see code changes instantly without rebuilding the entire Docker image. This is achieved using special `dev` Dockerfiles and Compose override files.

➡️ **Action: Start Your Development Environment**

Choose the command that matches your host operating system and your current task.

#### **Full Stack Development (Running Everything)**
Use this when you need all services running simultaneously.

*   **For Debian/Ubuntu-based Systems:**
    ```bash
    docker compose -f docker-compose.conda.yaml -f docker-compose.conda.dev.yaml --profile postgres --profile qdrant --profile gui --profile fastapi_gui up --build
    ```
*   **For RHEL/CentOS/Rocky Linux Systems:**
    ```bash
    docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.rhel.yaml -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.dev.rhel.yaml --profile postgres --profile qdrant --profile gui --profile fastapi_gui up --build
    ```

#### **Focused Development (e.g., Working only on the Flask API)**
Use this for a faster, more lightweight setup.

*   **For Debian/Ubuntu-based Systems:**
    ```bash
    docker compose -f docker-compose.conda.yaml -f docker-compose.conda.dev.yaml --profile flask_api --profile postgres up --build
    ```
*   **For RHEL/CentOS/Rocky Linux Systems:**
    ```bash
    docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.rhel.yaml -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.dev.rhel.yaml --profile flask_api --profile postgres up --build
    ```
These commands do two magical things:
1.  **They build from a `dev` Dockerfile**, which only installs dependencies and creates an empty environment.
2.  **They create a "live link"** (a bind mount) between your local project folder and the code inside the container.

#### The Development Rhythm: "Code, Save, Refresh"
Once the environment is running, **you do not need to run more Docker commands while coding.**

| Action You Take on Your PC | What Happens Inside the Container | Why It's Powerful |
| :--- | :--- | :--- |
| ✏️ You edit and save a file, like `api_server/routes.py`. | The container sees the change **instantly**. Because the server is running in debug/reload mode, it automatically restarts itself. | ⚡ **Hot Reloading:** You see your changes in seconds without rebuilding the image or restarting the container manually. |

#### When DO You Rebuild During Development?
You only need to stop (Ctrl+C) and re-run the `up --build` command when you change the *environment*, not the code:

*   When you change **`requirements.txt`** (adding, removing, or updating a library).
*   When you change a **`Dockerfile.dev...`** file itself (e.g., to install a new system package).

### 1.3. Workflow B: "Baking" the Final Production Package

This workflow is performed **only when development of a new version is complete.** You will take your finished code and "bake" it into secure, immutable, and portable packages for the client.

#### Step 1: Versioning
First, update the version number in the `image:` tag within your production compose files: **`docker-compose.conda.yaml`** and the air-gapped RHEL compose under **`hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.rhel.yaml`**. Adhere to semantic versioning (e.g., `v1.1.0`).

#### Step 2: Build the Production Images
From the project root, build the final, self-contained images for both platforms.

*   **Build the Debian-based production image:**
    ```bash
    docker build -f Dockerfile -t jk-image-similarity-app:v1.1.0 .
    ```
*   **Build the RHEL-based production image:**
    ```bash
    docker build -f Dockerfile.rhel -t jk-image-similarity-app:rhel-v1.1.0 .
    ```

#### Step 3: Save Images into an Air-Gap Bundle
To create a single, portable file for offline installation, bundle all required images into a `.tar` archive.

```bash
# Note: Ensure the versions here match your build commands and compose files.
docker save \
  jk-image-similarity-app:v1.1.0 \
  jk-image-similarity-app:rhel-v1.1.0 \
  qdrant/qdrant:v1.9.1-avx-disabled \
  postgres:15-alpine \
  -o jk-image-similarity-system-v1.1.0.tar
```

#### Step 4: Assemble the Client Package
Create a final `dist/` folder and copy only the essential files the client will need.

```
dist/
├── jk-image-similarity-system-v1.1.0.tar   (The bundled Docker images)
├── docker-compose.conda.yaml               (Main compose file for Debian/Ubuntu)
├── Docker_for_airgapped/
│   └── Dockerfiles_RHEL/
│       └── docker-compose.rhel.yaml        (RHEL-specific compose file)
├── .env.template                           (The user configuration template)
├── GUIDES/
│   └── SIMPLE_DOCKER_GUIDE.md              (This operations guide)
├── configs/                                (Configuration templates)
└── hyundai_document_authenticator/
    └── postgres-init/
        └── init-multiple-databases.sh      (Database initialization script)
```
Zip this `dist` folder. This is the final package to deliver.

---

## Part 2: The Client's Deployment & Management Guide

This guide will walk you through deploying and operating the application using the package provided to you.

### 2.1. Prerequisites for Deployment

Ensure the target machine has **Docker Engine** and **Docker Compose** installed.

### 2.2. Step 1: Load the Docker Images

First, load all application images from the provided `.tar` archive into your local Docker instance. This only needs to be done once, or when updating.

1.  Unzip the application package (e.g., `dist.zip`) and navigate into the directory.
2.  Run the `docker load` command:
    ```bash
    docker load -i jk-image-similarity-system-v1.1.0.tar
    ```

### 2.3. Step 2: Prepare the Configuration File (`.env`)

You must set up your environment's configuration before starting the services. This file controls passwords, security keys, and database connections.

1.  Create your personal configuration file by copying the template:
    ```bash
    cp .env.template .env
    ```
2.  Open the new `.env` file in a text editor and set the required values.

### 2.4. Step 3: Choose Your Architecture & Start the System

Below are the supported deployment scenarios. For each one, **first set the required `.env` variables**, and then **run the corresponding command**.

> 💡 **What are Docker Compose Profiles?**
> Profiles are like light switches for groups of services (`--profile <name>`). The command you run determines which services are activated. Your `.env` file must contain the correct settings for the profiles you choose to activate.

> 🚨 **IMPORTANT NOTE FOR RHEL HOSTS**
> If your server runs **Red Hat (RHEL), CentOS, or Rocky Linux**, you must use the RHEL-specific compose file. Add `-f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.rhel.yaml` to the beginning of **every `docker-compose` command**.
>
> *Example:* `docker-compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.rhel.yaml --profile ... up -d`

---
#### **Scenario A: 🚀 All-in-One (Internal PostgreSQL & Qdrant)**
*This runs the full application stack, creating new, dedicated database containers.*

**1. Configure your `.env` file:**
*   **CRITICAL:** Set new, secure, random values for `SECRET_KEY` and `ADMIN_API_USER_KEY`.
*   Set the `POSTGRES_DB`, `POSTGRES_USER`, and `POSTGRES_PASSWORD` for your new database.
*   Set `QDRANT_MODE=server`.
*   Leave all `..._EXTERNAL` variables commented out.

**2. Run the command:**
```bash
# On Debian/Ubuntu:
docker compose -f docker-compose.conda.yaml --profile postgres --profile qdrant --profile gui --profile fastapi_gui up -d

# On RHEL/CentOS:
docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.rhel.yaml --profile postgres --profile qdrant --profile gui --profile fastapi_gui up -d
```
---
#### **Scenario B: 🍃 Lightweight (SQLite & Embedded Qdrant)**
*Runs only the application services. Data is stored in local files. Good for simple tests or low-load situations.*

**1. Configure your `.env` file:**
*   **CRITICAL:** Set new, secure, random values for `SECRET_KEY` and `ADMIN_API_USER_KEY`.
*   Leave all `POSTGRES_...` variables blank or comment them out.
*   Set `QDRANT_MODE=embedded`.

**2. Run the command:**
```bash
# On Debian/Ubuntu:
docker compose -f docker-compose.conda.yaml --profile gui --profile fastapi_gui up -d

# On RHEL/CentOS:
docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.rhel.yaml --profile gui --profile fastapi_gui up -d
```
> 💡 **How this works:** The `gui` services depend on the API services. Docker Compose intelligently sees this `depends_on` relationship and automatically activates the required API profiles (`flask_api`, `fastapi_api`) for you.

---
#### **Scenario C: 🔗 Hybrid / Connect to External Databases**
*Runs the application services and connects them to existing PostgreSQL and/or Qdrant servers on your network.*

**1. Configure your `.env` file:**
*   **CRITICAL:** Set `SECRET_KEY` and `ADMIN_API_USER_KEY`.
*   **To connect to an external PostgreSQL:**
    *   Uncomment `POSTGRES_HOST_EXTERNAL` and set it to your database server's IP address or hostname.
    *   Set `POSTGRES_DB`, `POSTGRES_USER`, and `POSTGRES_PASSWORD` to match the external database credentials.
*   **To connect to an external Qdrant:**
    *   Uncomment `QDRANT_HOST_EXTERNAL` and set it to your Qdrant server's IP address or hostname.

**2. Run the command:**
*   Choose profiles for the services you want to run. **Do not include profiles for services you are connecting to externally.**
    *   *Example: External PostgreSQL, but Internal Qdrant:*
        ```bash
        # On Debian/Ubuntu:
        docker compose -f docker-compose.conda.yaml --profile qdrant --profile gui --profile fastapi_gui up -d
        # On RHEL/CentOS:
        docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.rhel.yaml --profile qdrant --profile gui --profile fastapi_gui up -d
        ```

---
#### **Scenario D: 🗓 Scheduled Mode (Default)**
Run the dedicated scheduler service. The image defaults to scheduler; the compose service `app_scheduler` starts the scheduler and runs in the background.

- **CPU/Conda (root):**
```bash
docker compose -f docker-compose.conda.yaml up -d app_scheduler
```

- **Dev/Conda (root):**
```bash
docker compose -f docker-compose.conda.yaml -f docker-compose.conda.dev.yaml up -d app_scheduler
```

- **RHEL (production):**
```bash
docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.rhel.yaml up -d app_scheduler
```

> **Note:** The scheduler service runs `doc_image_verifier.py` at configured intervals. Check the scheduler configuration in your `.env` file and application logs for scheduling details.

### 2.5. Step 4: Verify the System is Running

Check the status of your running services:
```bash
# On Debian/Ubuntu:
docker compose -f docker-compose.conda.yaml ps

# On RHEL/CentOS:
docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.rhel.yaml ps
```
All started services should show a state of `Up` or `running`. To view real-time logs:
```bash
# Example: View logs for the Flask API server
# Remember to add the RHEL compose file on RHEL hosts
docker compose -f docker-compose.conda.yaml logs -f flask_api

# Example: View logs for the scheduler service
docker compose -f docker-compose.conda.yaml logs -f app_scheduler

# Example: View logs for all services
docker compose -f docker-compose.conda.yaml logs -f
```

**Common service names you might see:**
- `app_scheduler`: The scheduler service (runs by default)
- `flask_api`: Flask API service (if `flask_api` profile is active)
- `fastapi_api`: FastAPI service (if `fastapi_api` profile is active)
- `gui`: Flask GUI service (if `gui` profile is active)
- `fastapi_gui`: FastAPI GUI service (if `fastapi_gui` profile is active)
- `db`: PostgreSQL database (if `postgres` profile is active)
- `qdrant`: Qdrant vector database (if `qdrant` profile is active)

### 2.6. Step 5: Access the Services

*   **GUI:** `http://<your_server_ip>:8501` (if `gui` profile is active)
*   **FastAPI GUI:** `http://<your_server_ip>:8502` (if `fastapi_gui` profile is active)
*   **Flask API:** `http://<your_server_ip>:5001` (if `flask_api` profile is active)
*   **FastAPI:** `http://<your_server_ip>:8000` (if `fastapi_api` profile is active)
*   **Qdrant Dashboard:** `http://<your_server_ip>:6333/dashboard` (if `qdrant` profile is active)

> **Note:** The specific API endpoints available depend on your application implementation. Check the application logs or documentation for available endpoints.

### 2.7. Step 6: Running Command-Line Tools (CLI)

Use `docker compose run` for one-off tasks. This starts a temporary container for a single command and removes it when finished (`--rm`).

Manual Mode (on-demand) uses the entrypoint "cli" to run doc_image_verifier.py subcommands.

#### **Available CLI Commands:**
- `search-doc`: Run TIF batch similarity search (photo extraction + image similarity + document aggregation)
- `build-image-index`: Build an image index from photos extracted from TIF documents

#### **Examples:**

- **CPU/Conda (root) - Document Search:**
```bash
docker compose -f docker-compose.conda.yaml run --rm cli_runner \
  cli hyundai_document_authenticator/doc_image_verifier.py search-doc \
  --folder ./hyundai_document_authenticator/data_real --top-doc 5 --top-k 5
```

- **CPU/Conda (root) - Build Index:**
```bash
docker compose -f docker-compose.conda.yaml run --rm cli_runner \
  cli hyundai_document_authenticator/doc_image_verifier.py build-image-index \
  --folder ./hyundai_document_authenticator/data_real
```

- **Dev/Conda (root) - Document Search:**
```bash
docker compose -f docker-compose.conda.yaml -f docker-compose.conda.dev.yaml run --rm cli_runner \
  cli hyundai_document_authenticator/doc_image_verifier.py search-doc \
  --folder ./hyundai_document_authenticator/data_real --top-doc 7 --top-k 5
```

- **Dev/Conda (root) - Build Index:**
```bash
docker compose -f docker-compose.conda.yaml -f docker-compose.conda.dev.yaml run --rm cli_runner \
  cli hyundai_document_authenticator/doc_image_verifier.py build-image-index \
  --folder ./hyundai_document_authenticator/data_real
```

- **RHEL - Document Search:**
```bash
docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.rhel.yaml run --rm cli_runner \
  cli hyundai_document_authenticator/doc_image_verifier.py search-doc \
  --folder ./hyundai_document_authenticator/data_real --top-doc 5 --top-k 5
```

- **RHEL - Build Index:**
```bash
docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.rhel.yaml run --rm cli_runner \
  cli hyundai_document_authenticator/doc_image_verifier.py build-image-index \
  --folder ./hyundai_document_authenticator/data_real
```

#### **Additional CLI Options:**
You can add various options to customize the behavior:

**For search-doc:**
- `--aggregation-strategy max|sum|mean`: How to combine similarity scores
- `--photo-extraction-mode yolo|bbox`: Method for extracting photos from documents
- `--yolo-model-path <path>`: Path to YOLO model for photo detection
- `--bbox-list <json>`: JSON list of bounding boxes for photo extraction
- `--output-folder <path>`: Where to save results

**For build-image-index:**
- `--engine faiss|qdrant|bruteforce`: Vector database provider
- `--batch-size <number>`: Feature extraction batch size
- `--force-rebuild-index`: Force rebuilding the index even if it exists

**Example with additional options:**
```bash
docker compose -f docker-compose.conda.yaml run --rm cli_runner \
  cli hyundai_document_authenticator/doc_image_verifier.py search-doc \
  --folder ./data_real --top-doc 7 --top-k 5 --aggregation-strategy max \
  --photo-extraction-mode yolo --yolo-model-path trained_model/weights/best.pt
```

Existing administrative examples (e.g., user management) remain valid; wrap them with `docker compose run --rm cli_runner` as shown above when applicable.

### 2.8. Step 7: Stopping the System

To safely stop and remove all application containers without deleting your data volumes:
```bash
# On Debian/Ubuntu:
docker compose -f docker-compose.conda.yaml down

# On RHEL/CentOS:
docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.rhel.yaml down
```

### 2.9. Step 8: Updating the Application

1.  Receive the new application package (e.g., `dist-v1.2.0.zip`).
2.  Stop the currently running application using the `down` command from the previous step.
3.  Load the new images: `docker load -i jk-image-similarity-system-v1.2.0.tar`.
4.  Replace the old `docker-compose...yml` files with the new ones. Review your `.env` file and merge any configuration changes from the new `.env.template`.
5.  Start the new version using the appropriate `up` command from Step 2.4.
