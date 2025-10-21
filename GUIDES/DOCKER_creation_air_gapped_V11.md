---
---

# 🚢 The Ultimate Enterprise Masterclass: Docker for the JK Image Similarity System

**Project:** JK Image Similarity System
**Document Version:** 12.1 (Definitive, Unabridged, Hybrid-Cloud Architecture)
**Audience:** Developers, System Administrators, DevOps Engineers, and End-Users
**Last Updated:** [Current Date]

## 📜 Introduction: A New Standard for Enterprise Deployment

Welcome to the definitive, enterprise-grade masterclass for packaging, deploying, and managing the JK Image Similarity System using Docker. This document is your single source of truth, providing a complete, line-by-line walkthrough for creating and running a self-contained, portable, and secure application suite. It is meticulously engineered to run on any machine with Docker—from standard **Debian/Ubuntu** servers to security-hardened **Red Hat Enterprise Linux (RHEL)** systems, including those that are completely disconnected from the internet (an "air-gapped" environment).

This is far more than just a setup guide; it's a comprehensive tutorial on the architecture and the industry-best practices that power this solution. We will explore every file, every command, and every design decision, empowering you to not only run the system but to understand, customize, and troubleshoot it with complete confidence.

**Why is this Docker architecture the gold standard for modern application deployment?**

*   **🚀 Unparalleled Flexibility & Conditional Logic:** This architecture is built on a powerful, **profile-driven model**. You can run the application in a simple, embedded mode, or enable full-fledged, high-performance **PostgreSQL** and **Qdrant** database servers with simple command-line flags (`--profile postgres`, `--profile qdrant`). You can choose to run the databases inside Docker or connect seamlessly to **existing, external database servers**, all controlled from a single configuration file. This hybrid capability is essential for integration into diverse enterprise IT landscapes.
*   **📦 True, Universal Portability:** The entire application—including all its code, complex dependencies, the optional **Qdrant vector database**, the optional **PostgreSQL metadata database(s)**, and all configurations—is bundled into a single, transportable package. It runs identically on a developer's laptop, a cloud testing server, or a secure, air-gapped production machine. The dream of "build once, run anywhere" is fully realized here.
*   **⚙️ Ironclad Consistency & Multi-OS Support:** This architecture permanently eliminates the "it works on my machine" problem. The environment is defined once, as code, and is replicated perfectly every time. We provide distinct, optimized `Dockerfile`s for both **Debian-based** and **RHEL-based** host systems, using their respective universal base images and best practices to ensure native performance and compliance.
*   **🔒 Hardened Security by Design:** Security is not an afterthought; it is woven into the fabric of this design. Each application component runs in its own isolated container as a dedicated, **non-root user**, drastically reducing the potential attack surface. We use SELinux-aware volume mounts (`:z` flag) for seamless and secure operation on RHEL systems and provide CPU-compatible images to ensure stability across different hardware platforms.
*   **💡 Professional Maintainability & Extensibility:** By leveraging modern Docker Compose features like **YAML Anchors** (`x-app-base`) and **Profiles**, we've created a clean, Don't-Repeat-Yourself (DRY) configuration that is easy to understand, maintain, and extend. The same set of files powers both local development with live-reloading and secure, immutable production deployments.

This guide is your trusted companion, designed for the **Developer** who will build and package the final product, and the **Client or System Administrator** who will deploy and manage it with confidence in their own environment.

***

## 📋 Table of Contents

*   **Part 1: A Deep Dive into the Docker Architecture**
    *   [1.1. Architecture Overview Diagram](#11-architecture-overview-diagram)
    *   [1.2. The Production Blueprints: `Dockerfile` & `Dockerfile.rhel` Explained](#12-the-production-blueprints-dockerfile--dockerfilerhel-explained)
    *   [1.3. The Production Conductors: `docker-compose.yaml` & `docker-compose.rhel.yml` Explained](#13-the-production-conductors-docker-composeyaml--docker-composerhelyml-explained)
    *   [1.4. The Development Overrides: `docker-compose.dev.yaml` & `docker-compose.dev.rhel.yml` Explained](#14-the-development-overrides-docker-composedevyaml--docker-composedevrhelyml-explained)
    *   [1.5. The Container's Brain: `docker-entrypoint.sh` Explained](#15-the-containers-brain-docker-entrypointsh-explained)
    *   [1.6. The Database Magician: `init-multiple-databases.sh` Explained](#16-the-database-magician-init-multiple-databasessh-explained)
    *   [1.7. The Master Control Panel: `.env` File Explained](#17-the-master-control-panel-env-file-explained)

*   **Part 2: The Developer's Handbook: Building and Packaging**
    *   [2.1. Prerequisite: Setting Up Your Development Environment](#21-prerequisite-setting-up-your-development-environment)
    *   [2.2. Automating Secure Configuration (`.env`)](#22-automating-secure-configuration-env)
    *   [2.3. Workflow A: The Live-Reload Development Loop](#23-workflow-a-the-live-reload-development-loop)
    *   [2.4. Workflow B: "Baking" the Final Production Package](#24-workflow-b-baking-the-final-production-package)

*   **Part 3: The Client's Deployment Guide: Step-by-Step Installation**
    *   [3.1. Prerequisites & Package Contents](#31-prerequisites--package-contents)
    *   [3.2. Step 1: Load the Application into Docker](#32-step-1-load-the-application-into-docker)
    *   [3.3. Step 2: Configure Your Deployment](#33-step-2-configure-your-deployment)
    *   [3.4. Step 3: Choose Your Architecture & Start the System](#34-step-3-choose-your-architecture--start-the-system)
    *   [3.5. Step 4: Accessing the Services](#35-step-4-accessing-the-services)
    *   [3.6. Step 5: Verify the System is Running](#36-step-5-verify-the-system-is-running)
    *   [3.7. Step 6: Using the Command-Line Tools (CLI)](#37-step-6-using-the-command-line-tools-cli)
    *   [3.8. Step 7: Stopping the System](#38-step-7-stopping-the-system)
    *   [3.9. Step 8: Updating the Application (The Basic Workflow)](#39-step-8-updating-the-application-the-basic-workflow)
    *   [3.10. Step 9: Post-Deployment Verification](#310-step-9-post-deployment-verification)

*   **Part 4: Production Operations & Best Practices**
    *   [4.1. 📈 Production-Grade Update Strategy: Database Migrations](#41--production-grade-update-strategy-database-migrations)
    *   [4.2. 🗄️ Critical Operations: Backup and Recovery Strategy](#42--critical-operations-backup-and-recovery-strategy)
    *   [4.3. 🔐 Security Hardening & Advanced Secrets Management](#43--security-hardening--advanced-secrets-management)
    *   [4.4. 📊 System Observability: Monitoring and Logging](#44--system-observability-monitoring-and-logging)
    *   [4.5. 🚀 Preparing for Growth: Scaling the System](#45--preparing-for-growth-scaling-the-system)

*   **Appendix: Deeper Dives & Foundational Knowledge**
    *   [A.1. 🐳 Docker Fundamentals (A Primer for Beginners)](#a1--docker-fundamentals-a-primer-for-beginners)
    *   [A.2. Understanding the Dynamic Host Configuration (`${VAR:-default}`)](#a2-understanding-the-dynamic-host-configuration-vardefault)
    *   [A.3. CPU Architecture Explained (AVX2 vs. AVX-Disabled)](#a3-cpu-architecture-explained-avx2-vs-avx-disabled)
    *   [A.4. Connecting to External Services (`host.docker.internal`)](#a4-connecting-to-external-services-hostdockerinternal)
    *   [A.5. Idempotency Explained: Why Our Scripts Are Robust](#a5-idempotency-explained-why-our-scripts-are-robust)
    *   [A.6. Docker Command Reference: `up`, `down`, `run`, `exec`](#a6-docker-command-reference-up-down-run-exec)
    *   [A.7. Essential Troubleshooting Commands](#a7-essential-troubleshooting-commands)
    *   [A.8. Managing Data and Volumes](#a8-managing-data-and-volumes)
    *   [A.9. Deep Dive into Docker Networking](#a9-deep-dive-into-docker-networking)
    *   [A.10. Security Hardening Best Practices](#a10-security-hardening-best-practices)
    *   [A.11. Adapting for Automated CI/CD Pipelines](#a11-adapting-for-automated-cicd-pipelines)
    *   [A.12. 🖥️ Cross-Platform Considerations (Windows/macOS)](#a12--cross-platform-considerations-windowsmacos)
    *   [A.13. ⚖️ Clarifying Qdrant Modes: Server vs. Embedded](#a13--clarifying-qdrant-modes-server-vs-embedded)

***

## Part 1: A Deep Dive into the Docker Architecture

This section provides an exhaustive explanation of each configuration file and a high-level view of the system's architecture. Understanding this core logic will empower you to confidently modify, adapt, and troubleshoot the system in any environment.

### 1.1. Architecture Overview Diagram

A visual diagram helps clarify how all the components interact. This diagram shows the data flow and relationships between the user, the containers, and the persistent data volumes.

```mermaid
graph TD
    subgraph "User's Browser"
        User
    end

    subgraph "Docker Host Machine"
        subgraph "Docker Network (e.g., myproject_default)"
            subgraph "Application Services"
                Flask_API[🚀 Flask API<br>(:5001)]
                FastAPI_API[🚀 FastAPI<br>(:8000)]
                Flask_GUI[🖥️ Flask GUI<br>(:8501)]
                FastAPI_GUI[🖥️ FastAPI GUI<br>(:8502)]
            end

            subgraph "Optional Database Services"
                Postgres[🐘 PostgreSQL<br>(:5432)]
                Qdrant[🧠 Qdrant<br>(:6333)]
            end
        end

        subgraph "Persistent Data (Docker Named Volumes)"
            PG_Data[🗄️ postgres_data]
            Qdrant_Data[🗄️ qdrant_data]
            Logs_Data[📄 logs_data]
            Instance_Data[⚙️ instance_data]
        end
    end

    User -- "HTTP Requests" --> Flask_GUI
    User -- "HTTP Requests" --> FastAPI_GUI

    Flask_GUI -- "API Calls" --> Flask_API
    FastAPI_GUI -- "API Calls" --> FastAPI_API

    Flask_API -- "Reads/Writes" --> Postgres
    FastAPI_API -- "Reads/Writes" --> Postgres
    Flask_API -- "Vector Search" --> Qdrant
    FastAPI_API -- "Vector Search" --> Qdrant

    Postgres -- "Persists Data To" --> PG_Data
    Qdrant -- "Persists Data To" --> Qdrant_Data
    
    Flask_API -- "Writes Logs To" --> Logs_Data
    FastAPI_API -- "Writes Logs To" --> Logs_Data
    Flask_API -- "Runtime Files" --> Instance_Data
    FastAPI_API -- "Runtime Files" --> Instance_Data
```

### 1.2. The Production Blueprints: `Dockerfile` & `Dockerfile.rhel` Explained

A `Dockerfile` is the master recipe for creating a Docker image. It's a text file containing a sequence of commands that assemble a portable, self-contained environment for our application. We employ a **multi-stage build**, an industry best practice that creates a final image that is dramatically smaller, more efficient, and more secure.

---

#### 🏗️ **Debian-based Blueprint: `Dockerfile`**

*This is the standard, general-purpose file for building on systems like Ubuntu, Debian, or in most cloud CI/CD environments.*

```dockerfile
# ==============================================================================
#                 Production Dockerfile (Debian-based)
# ==============================================================================
# This Dockerfile builds a secure, efficient, and production-ready image for the
# application using a multi-stage build process on a Debian base.

# ==============================================================================
# --- Stage 1: The "Builder" Workshop                                        ---
# ==============================================================================
# Purpose: This stage is like a temporary workshop. We install all dependencies,
# including heavy build-time tools (like C++ compilers) that we don't need in
# our final, lean production image.
#
# We start with a slim Debian "Bookworm" image. It provides a stable,
# well-supported, and minimal base.
# `AS builder` gives this stage a name we can reference in the next stage.
FROM python:3.10-slim-bookworm AS builder

# --- Environment Variables ---
# These are best practices for running Python in containers.
#   - PYTHONUNBUFFERED=1: Forces Python to print output directly to the console
#     without delay. This is essential for seeing logs in real-time from `docker logs`.
#   - PYTHONDONTWRITEBYTECODE=1: Prevents Python from creating .pyc files,
#     keeping our image clean and slightly smaller.
#   - PIP_NO_CACHE_DIR=off: Disables the pip cache, saving significant space.
#   - PIP_DISABLE_PIP_VERSION_CHECK=on: Supresses non-critical warnings about
#     pip versions during builds, keeping logs clean and builds faster.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=off PIP_DISABLE_PIP_VERSION_CHECK=on

# --- System Dependencies ---
# 'RUN' executes a command inside the container, creating a new "layer" in the image.
# We install 'build-essential', which contains C/C++ compilers (like gcc) needed
# to build some Python libraries (e.g., numpy, faiss) from source. This is a heavy
# dependency that we will discard before creating the final image.
#   - --no-install-recommends: A powerful optimization that avoids installing
#     optional "recommended" packages, keeping the image significantly leaner.
#   - && rm -rf /var/lib/apt/lists/*: Crucial cleanup step to remove package
#     lists after installation, reducing the layer size.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# --- Python Dependencies ---
# 'WORKDIR' sets the current working directory for all subsequent commands.
WORKDIR /app

# 'COPY' copies files from your local machine into the container.
# ⭐ PRO-TIP: We copy ONLY the requirements file first. This is a critical
# optimization that leverages Docker's layer caching. If this file doesn't change
# on a future build, Docker reuses the cached layer from the next command, skipping
# the time-consuming dependency installation and making builds much faster.
COPY requirements.txt .

# Install all Python libraries from our requirements file.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ==============================================================================
# --- Stage 2: The "Final" Production Image                                  ---
# ==============================================================================
# Purpose: This stage creates the final, minimal, and secure image. It starts
# from a fresh base and cherry-picks only the necessary, pre-compiled artifacts
# from the "Builder" stage.
FROM python:3.10-slim-bookworm AS final

# --- Environment Variables ---
# 'PYTHONPATH' tells Python where to find our application modules. This allows
# your code to use clean, absolute imports like `from core_engine...` from anywhere.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/appuser/app

# --- System Dependencies ---
# Install only the runtime dependencies needed. 'curl' is added for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- User and Group Setup (Security Best Practice) ---
# 🔒 CRITICAL SECURITY STEP: We create a dedicated, non-root user to run the
# application. Running containers as 'root' is a major security risk; if an
# attacker compromises the app, they would have root privileges inside the container.
# '-r' creates a system user, which is appropriate for services.
RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup appuser

# --- Copy Artifacts from Builder Stage ---
# 'COPY --from=builder' is the magic of multi-stage builds. We copy only the
# installed Python packages, leaving the compilers and build tools behind.
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# --- Application Setup ---
WORKDIR /home/appuser/app

# ⭐ OPTIMAL OWNERSHIP PATTERN: 'COPY --chown' sets the correct owner AS the files
# are copied. This is significantly more efficient than a separate `RUN chown -R`
# command because it creates a single, faster filesystem layer.
COPY --chown=appuser:appgroup . .
COPY --chown=appuser:appgroup docker-entrypoint.sh .
RUN chmod +x ./docker-entrypoint.sh

# --- Final User Switch ---
# 🔒 FINAL SECURITY STEP: Switch the active user from 'root' to our unprivileged
# 'appuser'. All subsequent commands and the final running container will now
# execute as this user.
USER appuser

# --- Port Exposure ---
# 'EXPOSE' documents the network ports the application listens on. It does NOT
# publish them to the host. It's metadata for developers and tools.
EXPOSE 5001 8000 8501 8502

# --- Execution ---
# 'ENTRYPOINT' configures the main executable for the container.
ENTRYPOINT ["/home/appuser/app/docker-entrypoint.sh"]
# 'CMD' provides the default argument to the ENTRYPOINT. If a user runs `docker run <image>`,
# this is equivalent to running `/app/docker-entrypoint.sh flask-api`.
CMD ["flask-api"]
```

#### 📜 Detailed Breakdown: `Dockerfile`

*   **`FROM python:3.10-slim-bookworm AS builder`**: This begins our first stage, named `builder`. We start with an official Python image that is `-slim` (has fewer non-essential packages) and is based on Debian "Bookworm". Using a specific version like `3.10-slim-bookworm` instead of `python:latest` is a best practice for reproducible builds.
*   **`ENV ...`**: We set several environment variables that are standard best practices for Python in Docker. `PYTHONUNBUFFERED=1` ensures that logs are sent directly to the console for real-time viewing with `docker logs`. `PYTHONDONTWRITEBYTECODE=1` prevents Python from creating `.pyc` files, keeping the image clean. The `PIP_` variables optimize the build process by disabling caches and version checks.
*   **`RUN apt-get update && apt-get install ...`**: This command installs system-level dependencies. In the `builder` stage, we install `build-essential` which includes compilers like `gcc`. These are necessary to build certain Python libraries (like `numpy`) from source code. `--no-install-recommends` is a crucial optimization that prevents `apt` from installing optional packages, significantly reducing image size. We chain commands with `&&` and end with `rm -rf /var/lib/apt/lists/*` to clean up package manager caches in the same layer, a key optimization to keep image layers small. `curl` is included as it may be needed for health checks or other scripts.
*   **`WORKDIR /app`**: This sets the working directory for subsequent commands.
*   **`COPY requirements.txt .`**: This is a critical caching optimization. We copy *only* the `requirements.txt` file first. Docker builds images in layers. If this file doesn't change between builds, Docker will reuse the cached layers for all subsequent steps, skipping the time-consuming dependency installation and dramatically speeding up development.
*   **`RUN pip install -r requirements.txt`**: This installs all our Python dependencies. Because of the previous step, this layer will only be rebuilt if `requirements.txt` has changed.
*   **`FROM python:3.10-slim-bookworm AS final`**: This begins our second and final stage. We start fresh from the same clean base image. This ensures that none of the build tools from the `builder` stage (like `gcc`) will be in our final image.
*   **`RUN groupadd ... && useradd ...`**: This is a critical security step. We create a dedicated, unprivileged user (`appuser`) and group (`appgroup`) to run our application. Running containers as the `root` user is a major security risk.
*   **`COPY --from=builder ...`**: This is the core magic of multi-stage builds. We copy only the necessary artifacts from the `builder` stage—in this case, the installed Python packages from `site-packages`—into our final image. The build tools and temporary files are left behind.
*   **`WORKDIR /home/appuser/app`**: We set the final working directory inside our non-root user's home directory.
*   **`COPY --chown=appuser:appgroup . .`**: We copy the application source code into the container. Using the `--chown` flag is more efficient than a separate `RUN chown` command as it sets ownership in the same filesystem layer.
*   **`USER appuser`**: This command switches the active user for all subsequent commands and for the final running container. From this point on, everything executes as our unprivileged `appuser`.
*   **`EXPOSE 5001 8000 8501 8502`**: This command serves as documentation, informing the user which ports the application is designed to listen on. It does not actually publish the ports to the host machine; that is done in `docker-compose.yaml`.
*   **`ENTRYPOINT [...]`**: This specifies the main command that will be executed when the container starts. We point it to our smart entrypoint script.
*   **`CMD ["flask-api"]`**: This provides the *default argument* to the `ENTRYPOINT`. If a user runs `docker run <image>` without any arguments, the container will execute `/home/appuser/app/docker-entrypoint.sh flask-api`. This can be easily overridden from the `docker-compose.yaml` file.

---

#### 🌶️ **RHEL-based Blueprint: `Dockerfile.rhel`**

*This file is specifically for building a native image on Red Hat Enterprise Linux or its derivatives (CentOS, Fedora). It uses Red Hat's Universal Base Image (UBI) and follows RHEL conventions for security and file paths.*

```dockerfile
# ==============================================================================
#                 Production Dockerfile for RHEL Environments
# ==============================================================================
# This file is optimized for RHEL-based systems, ensuring compliance and
# native performance by using Red Hat's Universal Base Image (UBI).

# --- Stage 1: The "Builder" - Uses Red Hat's UBI 9 and `microdnf` ---
FROM registry.access.redhat.com/ubi9/python-311 AS builder
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=off PIP_DISABLE_PIP_VERSION_CHECK=on

# Install RHEL build tools like gcc using the minimal `microdnf` package manager.
RUN microdnf install -y gcc-toolset-12 python3.11-devel curl && microdnf clean all

# Use the standard RHEL application directory `/opt/app-root/src`.
WORKDIR /opt/app-root/src
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# --- Stage 2: The "Final" Image - Starts fresh from the clean UBI 9 base ---
FROM registry.access.redhat.com/ubi9/python-311 AS final
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/app-root/src

# Install only runtime dependencies needed. `curl` is added for the healthcheck.
RUN microdnf install -y curl && microdnf clean all

# 💡 RHEL CONVENTION: No need to create a user; UBI provides a default non-root
# user (UID 1001) by convention, which enhances security out-of-the-box.
# Copy installed packages from the builder stage.
COPY --from=builder /opt/app-root/lib/python3.11/site-packages /opt/app-root/lib/python3.11/site-packages
COPY --from=builder /opt/app-root/bin /opt/app-root/bin

WORKDIR /opt/app-root/src
# Copy application code, setting ownership to the default non-root user (1001) and root group (0).
COPY --chown=1001:0 . .
COPY --chown=1001:0 docker-entrypoint.sh .
RUN chmod +x ./docker-entrypoint.sh

# Switch to the non-root user by its UID.
USER 1001
EXPOSE 5001 8000 8501 8502
ENTRYPOINT ["/opt/app-root/src/docker-entrypoint.sh"]
CMD ["flask-api"]
```

#### 📜 Detailed Breakdown: `Dockerfile.rhel`

*   **`FROM registry.access.redhat.com/ubi9/python-311`**: This uses Red Hat's Universal Base Image (UBI) for Python 3.11. UBI images are freely distributable and are built with RHEL security and performance standards.
*   **`RUN microdnf install ...`**: Instead of `apt-get`, RHEL-based images use `microdnf` (a lightweight version of `dnf`). We install the RHEL-equivalent build tools (`gcc-toolset-12`) and development headers (`python3.11-devel`).
*   **`WORKDIR /opt/app-root/src`**: We adhere to the RHEL convention of placing application source code in `/opt/app-root/src`.
*   **`USER 1001`**: UBI images come with a pre-configured non-root user (with User ID 1001). We don't need to create one; we just switch to it. This is an excellent security feature provided out-of-the-box.
*   **`COPY --chown=1001:0 . .`**: When copying the application code, we set the owner to the UID `1001` and the group to `0` (the root group). This is a standard practice on OpenShift/RHEL platforms that allows the non-root user to run the application while maintaining compatibility with certain storage and system integrations.

The rest of the logic (multi-stage build, caching, entrypoint) is identical to the Debian-based `Dockerfile`, demonstrating how the core principles can be applied across different base operating systems.

### 1.3. The Production Conductors: `docker-compose.yaml` & `docker-compose.rhel.yml` Explained

These files are the orchestra conductors for the entire application stack. They define the `services`, `networks`, and `volumes` that constitute your application. They have been engineered for maximum flexibility, allowing you to run a simple, all-in-one setup or a complex, distributed system with external databases, all from the same file.

---

#### 🎶 **Standard Conductor: `docker-compose.yaml`**

*This is the primary, universal file for defining the application stack on Debian-based systems.*

```yaml
# ==============================================================================
#      DUAL-PURPOSE Production & Development Compose File (Debian-based)
# ==============================================================================
# This file defines the application stack with optional, profile-driven services.
# It is designed for two primary scenarios:
#
# 1. PRODUCTION (Default): It runs a pre-built, versioned Docker image.
# 2. DEVELOPMENT: By using an override file, this exact same file can be used
#    to build the image locally and enable live-reloading.
#
# --- DEPLOYMENT SCENARIOS (Controlled by .env and --profile flags) ---
#
# 1. ALL-IN-ONE (Internal PG, Qdrant, APIs):
#    - .env: Fill PG credentials. QDRANT_MODE=server.
#    - CMD:  docker-compose --profile gui --profile fastapi_gui --profile postgres --profile qdrant up -d
#
# 2. FULLY EMBEDDED (SQLite & Embedded Qdrant, no separate DB containers):
#    - .env: Leave PG credentials blank. QDRANT_MODE=embedded.
#    - CMD:  docker-compose --profile gui --profile fastapi_gui up -d
#
# 3. HYBRID (e.g., Internal GUI, External PG & Qdrant):
#    - .env: Set _EXTERNAL variables for PG & Qdrant.
#    - CMD:  docker-compose --profile gui --profile fastapi_gui up -d
# ==============================================================================

# ==============================================================================
#                      --- TEMPLATE DEFINITIONS (YAML ANCHORS) ---
# ==============================================================================
# This section defines a reusable template for our application services.
# The 'x-' prefix is a standard convention indicating that this is NOT a
# service to be run, just a block of reusable configuration.
x-app-base: &app-base
  # image: should be updated for new releases
  image: jk-image-similarity-app:v1.0.0 
  restart: unless-stopped
  env_file: .env
  volumes:
    # We mount the configs folder as read-only (:ro) for security.
    - ./configs:/home/appuser/app/configs:ro
    # We use named volumes for all runtime data for portability and safety.
    - downloads_data:/home/appuser/app/downloads
    - instance_data:/home/appuser/app/instance
    - logs_data:/home/appuser/app/logs
  environment:
    # --- DYNAMIC HOST & MODE CONFIGURATION ---
    # This block dynamically configures the application at startup. It uses
    # the ":- default" syntax to fall back to safe, internal values if
    # variables are not set in the .env file. See Appendix A.2 for details.
    - POSTGRES_HOST=${POSTGRES_HOST_EXTERNAL:-db}
    - POSTGRES_USER_HOST=${POSTGRES_HOST_EXTERNAL:-db}
    - QDRANT_HOST=${QDRANT_HOST_EXTERNAL:-qdrant}
    - QDRANT_MODE=${QDRANT_MODE:-server} # Defaults to 'server' mode.

# ==============================================================================
#                            --- CORE SERVICES ---
# ==============================================================================
services:
  # --- 1. PostgreSQL Database Service (OPTIONAL) ---
  db:
    # 'profiles: ["postgres"]' makes this service optional. It will only start
    # if you run docker-compose with the '--profile postgres' flag.
    profiles: ["postgres"]
    image: postgres:15-alpine
    container_name: postgres_db_prod_service
    restart: unless-stopped
    volumes:
      - postgres_data:/var/lib/postgresql/data
      # We mount the initialization script as read-only for security.
      - ./postgres-init:/docker-entrypoint-initdb.d:ro
    env_file: .env 
    environment: 
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      # This extra variable is picked up by our custom init script.
      POSTGRES_USER_DB_NAME: ${POSTGRES_USER_DB}
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 5

  # --- 2. Qdrant Vector Database Service (OPTIONAL) ---
  qdrant:
    profiles: ["qdrant"]
    # We use the AVX-disabled image for maximum CPU compatibility. See Appendix A.3.
    image: qdrant/qdrant:v1.9.1-avx-disabled
    container_name: qdrant_db_prod_service
    restart: unless-stopped
    ports: ["6333:6333", "6334:6334"]
    volumes: ["qdrant_data:/qdrant/storage"]
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:6333/healthz || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  # --- 3. Flask API Service (OPTIONAL) ---
  flask_api:
    profiles: ["flask_api"] 
    # The '<<:' is a YAML Merge Key. It merges the entire &app-base anchor here.
    <<: *app-base
    container_name: flask_api_prod_service
    # The command passed to the docker-entrypoint.sh script.
    command: flask-api
    ports: ["5001:5001"]
    # 'depends_on' controls startup order. 'condition: service_healthy' ensures this
    # service waits until the databases are ready before starting. 'required: false'
    # allows this service to start even if the db/qdrant profiles are inactive.
    depends_on:
      db: { condition: service_healthy, required: false }
      qdrant: { condition: service_healthy, required: false }
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:5001/health || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 5
      start_period: 30s

  # --- 4. FastAPI Service (OPTIONAL) ---
  fastapi_api:
    profiles: ["fastapi_api"]
    <<: *app-base
    container_name: fastapi_api_prod_service
    command: fastapi-api
    ports: ["8000:8000"]
    depends_on:
      db: { condition: service_healthy, required: false }
      qdrant: { condition: service_healthy, required: false }
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 5
      start_period: 30s

  # --- 5. GUI Service (OPTIONAL) ---
  gui:
    profiles: ["gui"]
    <<: *app-base
    container_name: gui_prod_service
    command: gui
    ports: ["8501:8501"]
    environment:
      # This sets an environment variable INSIDE the GUI container, telling it
      # how to reach the API. It dynamically uses external overrides if they
      # exist, otherwise falls back to internal service discovery names.
      - API_BASE_URL=http://${FLASK_API_HOST_EXTERNAL:-flask_api}:${FLASK_API_PORT_EXTERNAL:-5001}/api/v1
    depends_on:
      # ⭐ CRITICAL DEPENDENCY: This GUI requires the API to be healthy before it starts.
      # Docker Compose is smart: enabling the 'gui' profile will now automatically
      # enable the 'flask_api' profile as well, because of this dependency.
      flask_api: { condition: service_healthy }
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8501/_stcore/health || exit 1"]
      interval: 15s
      timeout: 10s
      retries: 5
      start_period: 30s

  # --- 6. FastAPI GUI Service (OPTIONAL) ---
  fastapi_gui:
    profiles: ["fastapi_gui"]
    <<: *app-base
    container_name: fastapi_gui_prod_service
    command: fastapi-gui
    ports: ["8502:8502"]
    environment:
      - API_BASE_URL=http://${FASTAPI_API_HOST_EXTERNAL:-fastapi_api}:${FASTAPI_API_PORT_EXTERNAL:-8000}/api/v1
    depends_on:
      fastapi_api: { condition: service_healthy }
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8502/_stcore/health || exit 1"]
      interval: 15s
      timeout: 10s
      retries: 5
      start_period: 30s

  # --- 7. CLI Helper Service (FOR ONE-OFF TASKS) ---
  # This is a "dummy" service. It uses the same application image but has no
  # command and is never meant to be run permanently. Its only purpose is to
  # provide a clean target for `docker-compose run` commands.
  cli_runner:
    profiles: ["cli"] # It has its own profile, so it never starts by default.
    <<: *app-base
    # It still declares dependencies so `docker-compose run` can wait for them if they are active.
    depends_on:
      db: { condition: service_healthy, required: false }
      qdrant: { condition: service_healthy, required: false }

# ==============================================================================
#                      --- NAMED VOLUME DEFINITIONS ---
# ==============================================================================
# This section formally declares the named volumes used by the services.
# See Appendix A.8 for a deep dive on why named volumes are essential.
volumes:
  postgres_data:
  qdrant_data:
  downloads_data:
  instance_data:
  logs_data:
```

#### 📜 Detailed Breakdown: `docker-compose.yaml`

*   **`x-app-base: &app-base`**: This is a YAML Anchor. It defines a reusable block of configuration that we can apply to all our application services. This follows the Don't-Repeat-Yourself (DRY) principle, making the file much cleaner and easier to maintain.
*   **`image: jk-image-similarity-app:v1.0.0`**: Specifies the pre-built, versioned production image to use. This should be updated by the developer for each new release.
*   **`restart: unless-stopped`**: A policy that tells Docker to automatically restart the container if it crashes, but not to start it automatically on Docker daemon startup if it was manually stopped by the user.
*   **`env_file: .env`**: Tells Docker Compose to load environment variables from a file named `.env` in the same directory. This is how we inject all our configuration.
*   **`volumes:`**: This section defines the data mounts.
    *   `./configs:/...:ro`: This is a "bind mount", linking a directory on the host (`./configs`) to a directory in the container. `:ro` makes it read-only for security.
    *   `downloads_data:/...`: This is a "named volume". Docker manages the storage for this volume, making it portable and safe. See Appendix A.8 for a full explanation.
*   **`environment:`**: Defines environment variables inside the container. The `${VAR:-default}` syntax provides a default value if the variable isn't set in the `.env` file, which is the key to our flexible architecture.
*   **`services:`**: The main section where each component (container) of our application is defined.
*   **`db:` / `qdrant:`**: These are our optional database services.
    *   **`profiles: ["postgres"]`**: This is the core of our conditional logic. This service will *only* be started if the user runs `docker-compose` with the `--profile postgres` flag.
    *   **`healthcheck:`**: A critical feature for production. Docker will periodically run the `test` command inside the container to check if it's healthy. Other services can then use `depends_on: { condition: service_healthy }` to wait until the database is fully initialized and ready to accept connections before they start.
*   **`flask_api:` / `fastapi_api:` / etc.**: These are our application services.
    *   **`<<: *app-base`**: This is a YAML Merge Key. It instructs the parser to inject the entire `&app-base` anchor here, so we don't have to repeat all the common settings.
    *   **`command: flask-api`**: This overrides the default `CMD` from the `Dockerfile` and passes `flask-api` as the argument to our `docker-entrypoint.sh` script, telling it which process to launch.
    *   **`ports: ["5001:5001"]`**: This publishes a port, mapping port `5001` on the host machine to port `5001` inside the container, making the service accessible from outside Docker.
    *   **`depends_on:`**: Defines the startup order. For example, `gui` depends on `flask_api`, so `flask_api` will be started first.
*   **`cli_runner:`**: This is a special "dummy" service that is never meant to be run as a long-lived process. Its sole purpose is to provide a pre-configured entrypoint for running one-off tasks using the `docker-compose run` command.
*   **`volumes:` (at the bottom)**: This top-level key formally declares the named volumes used in the stack. This is a best practice that allows Docker to manage them properly.

---

#### 🌶️ **RHEL Conductor: `docker-compose.rhel.yml`**

*This mirrors the structure of `docker-compose.yaml`, but with RHEL-specific paths, volume names, and SELinux flags. The core logic is identical.*

```yaml
# ==============================================================================
#      DUAL-PURPOSE Production & Development Compose File for RHEL
# ==============================================================================
# This file defines the application stack for a Red Hat Enterprise Linux (RHEL)
# or compatible system (e.g., CentOS, Rocky Linux).
#
# Key RHEL-specific features:
#   - Uses RHEL-standard paths (e.g., /opt/app-root/src).
#   - Includes the ':z' flag on bind mounts for SELinux compatibility.
#   - Uses RHEL-specific names for containers and volumes to avoid conflicts.
# ==============================================================================

x-app-base: &app-base
  # image: should be updated for new releases
  image: jk-image-similarity-app:rhel-v1.0.0 
  restart: unless-stopped
  env_file: .env
  volumes:
    # 🔒 The ':z' flag is CRITICAL for RHEL. It tells Docker to apply the correct
    # SELinux security label to the host directory, allowing the container's
    # processes to access it without being blocked by SELinux policies.
    - ./configs:/opt/app-root/src/configs:ro:z
    - downloads_data_rhel:/opt/app-root/src/downloads
    - instance_data_rhel:/opt/app-root/src/instance
    - logs_data_rhel:/opt/app-root/src/logs
  environment:
    - POSTGRES_HOST=${POSTGRES_HOST_EXTERNAL:-db}
    - POSTGRES_USER_HOST=${POSTGRES_HOST_EXTERNAL:-db}
    - QDRANT_HOST=${QDRANT_HOST_EXTERNAL:-qdrant}
    - QDRANT_MODE=${QDRANT_MODE:-server}

services:
  db:
    profiles: ["postgres"]
    image: postgres:15-alpine
    container_name: postgres_db_rhel_prod_service
    restart: unless-stopped
    volumes:
      - postgres_data_rhel:/var/lib/postgresql/data
      - ./postgres-init:/docker-entrypoint-initdb.d:ro:z
    env_file: .env
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_USER_DB_NAME: ${POSTGRES_USER_DB}
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 5

  qdrant:
    profiles: ["qdrant"]
    image: qdrant/qdrant:v1.9.1-avx-disabled
    container_name: qdrant_db_rhel_prod_service
    restart: unless-stopped
    ports: ["6333:6333", "6334:6334"]
    volumes: ["qdrant_data_rhel:/qdrant/storage"]
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:6333/healthz || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  flask_api:
    profiles: ["flask_api"]
    <<: *app-base
    container_name: flask_api_rhel_prod_service
    command: flask-api
    ports: ["5001:5001"]
    depends_on:
      db: { condition: service_healthy, required: false }
      qdrant: { condition: service_healthy, required: false }
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:5001/health || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 5
      start_period: 30s

  fastapi_api:
    profiles: ["fastapi_api"]
    <<: *app-base
    container_name: fastapi_api_rhel_prod_service
    command: fastapi-api
    ports: ["8000:8000"]
    depends_on:
      db: { condition: service_healthy, required: false }
      qdrant: { condition: service_healthy, required: false }
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 5
      start_period: 30s

  gui:
    profiles: ["gui"]
    <<: *app-base
    container_name: gui_rhel_prod_service
    command: gui
    ports: ["8501:8501"]
    environment:
      - API_BASE_URL=http://${FLASK_API_HOST_EXTERNAL:-flask_api}:${FLASK_API_PORT_EXTERNAL:-5001}/api/v1
    depends_on:
      flask_api: { condition: service_healthy }
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8501/_stcore/health || exit 1"]
      interval: 15s
      timeout: 10s
      retries: 5
      start_period: 30s

  fastapi_gui:
    profiles: ["fastapi_gui"]
    <<: *app-base
    container_name: fastapi_gui_rhel_prod_service
    command: fastapi-gui
    ports: ["8502:8502"]
    environment:
      - API_BASE_URL=http://${FASTAPI_API_HOST_EXTERNAL:-fastapi_api}:${FASTAPI_API_PORT_EXTERNAL:-8000}/api/v1
    depends_on:
      fastapi_api: { condition: service_healthy }
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8502/_stcore/health || exit 1"]
      interval: 15s
      timeout: 10s
      retries: 5
      start_period: 30s

  cli_runner:
    profiles: ["cli"]
    <<: *app-base
    container_name: cli_runner_rhel_service
    depends_on:
      db: { condition: service_healthy, required: false }
      qdrant: { condition: service_healthy, required: false }

volumes:
  postgres_data_rhel:
  qdrant_data_rhel:
  downloads_data_rhel:
  instance_data_rhel:
  logs_data_rhel:
```

#### 📜 Detailed Breakdown: `docker-compose.rhel.yml`

This file is nearly identical to its Debian counterpart, with a few key distinctions for RHEL compatibility:
*   **Distinct Naming**: All containers (`container_name`) and named volumes (`postgres_data_rhel`) have an `_rhel` suffix. This prevents naming collisions if a developer were to accidentally try to run both Debian and RHEL stacks on the same machine.
*   **RHEL-specific Paths**: The volume mounts in the `x-app-base` anchor point to the correct RHEL path (`/opt/app-root/src`) inside the container, matching the `Dockerfile.rhel`.
*   **The `:z` Flag**: The most critical difference is the `:z` flag appended to all bind mounts (e.g., `./configs:/...:ro:z`). This flag instructs Docker to relabel the host directory with the correct SELinux context, allowing the container's process to read/write to it. Without this flag, SELinux (which is enabled by default on RHEL) would block the container from accessing the mounted host directories, causing permission errors.

### 1.4. The Development Overrides: `docker-compose.dev.yaml` & `docker-compose.dev.rhel.yml` Explained

> **💡 FOR DEVELOPERS ONLY:** These files should **never** be given to the client. They temporarily modify the production setup to enable a fast, iterative development workflow with **live code reloading** and an **isolated development database**. When you run `docker-compose -f <base_file> -f <dev_file> up`, Docker Compose intelligently merges them, with the `dev` file's settings taking precedence.

```yaml
# ==============================================================================
#            DEVELOPMENT OVERRIDE for Debian-based Environments
# ==============================================================================
# This file contains the overrides needed for a local development environment.
# It MERGES with 'docker-compose.yaml' to create a complete dev setup.
# ==============================================================================

# ==============================================================================
#                 --- DEVELOPMENT OVERRIDE TEMPLATE (YAML ANCHOR) ---
# ==============================================================================
x-dev-app-overrides: &dev-overrides
  # OVERRIDE 1: Build the image locally instead of using a production image.
  build:
    context: .
    dockerfile: Dockerfile # The base Dockerfile works for dev too

  # OVERRIDE 2: Enable Live Code Reloading.
  # This **replaces** the `volume` section from the base file. The production
  # named volumes are ignored, and a direct bind mount of your source code is
  # used instead. This is the key to live-reloading.
  volumes:
    - .:/home/appuser/app

services:
  # --- 1. PostgreSQL Development Database (OVERRIDE) ---
  db:
    profiles: ["postgres"]
    # Use a SEPARATE named volume to keep development data isolated from production.
    container_name: postgres_db_dev_service
    volumes:
      - postgres_data_dev:/var/lib/postgresql/data
      - ./postgres-init:/docker-entrypoint-initdb.d:ro
    ports: ["5433:5432"] # Map to 5433 to avoid conflicts with a local host DB
    # Security note: exposing DB ports on the host increases attack surface. Prefer docker exec or internal network access in dev.
    # See MASTER_GUIDE.md → Development database exposure (dev-only): ../GUIDES/MASTER_GUIDE.md#development-database-exposure-dev-only

  # --- 2. Qdrant Development Database (OVERRIDE) ---
  qdrant:
    profiles: ["qdrant"]
    container_name: qdrant_db_dev_service
    volumes:
      - qdrant_data_dev:/qdrant/storage

  # --- 3. Flask API Service (Development Mode) ---
  flask_api:
    <<: *dev-overrides
    container_name: flask_api_dev_service
    # Use Flask's built-in debug server with auto-reloading.
    command: flask run --host=0.0.0.0 --port=5001 --debug
    depends_on:
      db: { condition: service_healthy, required: false }
      qdrant: { condition: service_healthy, required: false }

  # --- 4. FastAPI Service (Development Mode) ---
  fastapi_api:
    <<: *dev-overrides
    container_name: fastapi_api_dev_service
    # Use Uvicorn's live-reloading feature.
    command: uvicorn fastapi_app.main:app --host 0.0.0.0 --port 8000 --reload
    depends_on:
      db: { condition: service_healthy, required: false }
      qdrant: { condition: service_healthy, required: false }

  # --- 5. GUI Service (Development Mode) ---
  gui:
    <<: *dev-overrides
    container_name: gui_dev_service
    # Run streamlit directly for live reloading.
    command: streamlit run gui_app/app.py --server.port 8501 --server.address 0.0.0.0
    depends_on:
      flask_api: { condition: service_started }

  # --- 6. FastAPI GUI Service (Development Mode) ---
  fastapi_gui:
    <<: *dev-overrides
    container_name: fastapi_gui_dev_service
    command: streamlit run fastapi_gui/main.py --server.port 8502 --server.address 0.0.0.0
    depends_on:
      fastapi_api: { condition: service_started }

# ==============================================================================
#                      --- DEVELOPMENT NAMED VOLUMES ---
# ==============================================================================
# Declare the separate volumes to keep dev data safe from prod data.
volumes:
  postgres_data_dev:
  qdrant_data_dev:
```

#### 📜 Detailed Breakdown: `docker-compose.dev.yaml`

When this file is used with a base `docker-compose.yaml`, Docker Compose merges them. If a service or setting is defined in both files, the value from this `dev` file wins.

*   **`x-dev-app-overrides`**: A YAML anchor specific to development.
    *   **`build:`**: This directive **overrides** the `image:` directive from the base file. Instead of pulling a pre-built image, it tells Docker Compose to build the image locally from the `Dockerfile`.
    *   **`volumes: - .:/home/appuser/app`**: This is the key to live reloading. It **replaces** the volume definitions from the base `x-app-base` anchor. It mounts the entire project directory from your host machine (`.`) directly into the container's working directory. When you change a `.py` file on your host, it's instantly changed inside the container.
*   **`services: db:`**: We override the `db` service to use a separate named volume (`postgres_data_dev`). This is a critical safety measure to prevent your development experiments from corrupting your production data, and vice-versa. We also expose the port to `5433` on the host to avoid conflicts with a potential PostgreSQL instance running natively on the host at `5432`.
*   **`services: flask_api:` / `fastapi_api:`**: We override the `command` for each API service. Instead of running the production-grade `gunicorn` server, we run the framework's built-in development server (`flask run --debug` or `uvicorn --reload`). These servers are designed to watch for file changes in the mounted volume and automatically restart, giving you an instant feedback loop.

### 1.5. The Container's Brain: `docker-entrypoint.sh` Explained

This shell script is the single most important part of the runtime architecture. It acts as a smart, dynamic bootloader for the container, configuring the application based on environment variables before launching the final process. It is a masterpiece of robust, defensive shell scripting.

```bash
#!/bin/sh
# ==============================================================================
#                 PROFESSIONAL PRODUCTION DOCKER ENTRYPOINT
# ==============================================================================
#
# This script serves as the single, robust entrypoint for the application
# container. It is designed with best practices for production environments.
#
# Its responsibilities are:
#
#   1. RUNTIME CONFIGURATION:
#      - It dynamically inspects environment variables (`$VAR`) at container
#        startup.
#      - It modifies configuration files (`.yaml`) to apply settings for
#        databases or other services without requiring a new image build.
#      - This makes the Docker image a flexible, reusable artifact.
#
#   2. PROCESS MANAGEMENT:
#      - It determines which application component to launch based on the
#        command passed to `docker run` or `docker-compose`.
#      - It uses 'exec' to replace itself with the final application process,
#        ensuring correct signal handling (e.g., for Ctrl+C or `docker stop`).
#
# ==============================================================================

# --- SAFETY FIRST: Exit immediately if a command exits with a non-zero status.
# This is a critical safety feature to prevent the container from starting in
# a broken or unpredictable state if any of the setup commands fail.
set -e

# --- VISUALS: Define ANSI color codes for readable and organized log output.
# Using these variables makes the script's output much easier to follow.
C_BLUE='\033[94m'
C_GREEN='\033[92m'
C_YELLOW='\033[93m'
C_RED='\033[91m'
C_BOLD='\033[1m'
C_RESET='\033[0m' # Resets all text formatting

# ==============================================================================
#                          CONFIGURATION FUNCTIONS
# ==============================================================================

# --- Function: Configure Qdrant Vector Database ---
# Switches the vector database mode between 'server' (network) and 'embedded'
# (local file) by commenting/uncommenting lines in the config files.
configure_qdrant() {
    API_CONFIG="configs/api_config.yaml"
    CLI_CONFIG="configs/image_similarity_config.yaml"

    # Robustness Check: If config files are missing, warn and exit the function.
    if [ ! -f "$API_CONFIG" ] || [ ! -f "$CLI_CONFIG" ]; then
        echo -e "${C_YELLOW}🟡 Warning: One or more Qdrant config files not found. Skipping Qdrant configuration.${C_RESET}"
        return
    fi

    # Read QDRANT_MODE from the environment, defaulting to 'server'.
    QDRANT_MODE=${QDRANT_MODE:-server}

    echo -e "\n${C_BLUE}--------------------------------------------------${C_RESET}"
    echo -e "${C_BOLD}⚙️  Configuring Qdrant Vector DB: [${QDRANT_MODE}]${C_RESET}"

    if [ "$QDRANT_MODE" = "server" ]; then
        echo "   -> Enabling network settings for Server Mode..."
        # Use 'sed' (stream editor) to perform the configuration.
        # - The '-i' flag modifies the file in-place.
        # - The '-E' flag enables extended regular expressions.
        # - The regex 's/^(\s*)#\s*(host:.*)/\1\2/' finds a commented-out 'host:' line
        #   and replaces it with the same line, uncommented.
        # - Passing both filenames to a single sed command is efficient and POSIX-compliant.
        sed -i -E 's/^(\s*)#\s*(host:.*)/\1\2/' "$API_CONFIG" "$CLI_CONFIG"
        sed -i -E 's/^(\s*)#\s*(port:.*)/\1\2/' "$API_CONFIG" "$CLI_CONFIG"
        sed -i -E 's/^(\s*)(location:.*)/#\1\2/' "$API_CONFIG" "$CLI_CONFIG"
        echo -e "${C_GREEN}   ✅ Configured for Server Mode.${C_RESET}"
    else
        echo "   -> Enabling local file settings for Embedded Mode..."
        sed -i -E 's/^(\s*)(host:.*)/#\1\2/' "$API_CONFIG" "$CLI_CONFIG"
        sed -i -E 's/^(\s*)(port:.*)/#\1\2/' "$API_CONFIG" "$CLI_CONFIG"
        sed -i -E 's/^(\s*)#\s*(location:.*)/\1\2/' "$API_CONFIG" "$CLI_CONFIG"
        echo -e "${C_GREEN}   ✅ Configured for Embedded Mode.${C_RESET}"
    fi
}

# --- Function: Configure PostgreSQL Databases ---
# Conditionally sets PostgreSQL connection URIs if credentials are provided.
# It supports reading secrets from Docker Secrets first, then falls back to env vars.
configure_database() {
    CONFIG_FILE="configs/api_config.yaml"

    # Robustness Check: This function cannot run without the main config file.
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${C_YELLOW}🟡 Warning: API Config file '$CONFIG_FILE' not found. Skipping all Database configuration.${C_RESET}"
        return
    fi

    echo -e "\n${C_BLUE}--------------------------------------------------${C_RESET}"
    echo -e "${C_BOLD}⚙️  Configuring Database Connections${C_RESET}"

    # --- Read Passwords from Docker Secrets first, for enhanced security ---
    if [ -f /run/secrets/postgres_password ]; then
      POSTGRES_PASSWORD=$(cat /run/secrets/postgres_password)
    fi
    if [ -f /run/secrets/postgres_user_password ]; then
      POSTGRES_USER_PASSWORD=$(cat /run/secrets/postgres_user_password)
    fi

    # --- Configure Main PostgreSQL Database (for Search Results) ---
    # This block only runs if a complete set of credentials is provided.
    if [ -n "$POSTGRES_DB" ] && [ -n "$POSTGRES_USER" ] && [ -n "$POSTGRES_PASSWORD" ]; then
        echo "   -> Main PostgreSQL credentials found. Setting 'database_uri'..."
        # Construct the URI. The ':-' syntax provides a default value if the env var is not set.
        # e.g., ${POSTGRES_HOST:-db} uses 'db' if $POSTGRES_HOST is empty.
        RESULTS_DB_URI="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST:-db}:${POSTGRES_PORT:-5432}/${POSTGRES_DB}"

        # This 'sed' command robustly replaces the line, whether it is commented out or not.
        # This makes the script idempotent (safe to re-run). See Appendix A.5.
        sed -i -E "s|^(\s*#\s*)?database_uri:.*|database_uri: ${RESULTS_DB_URI}|" "$CONFIG_FILE"
        echo -e "${C_GREEN}   ✅ Main database URI configured for PostgreSQL.${C_RESET}"
    else
        echo -e "${C_YELLOW}   -> Main PostgreSQL credentials not set. Application will use its default (e.g., SQLite).${C_RESET}"
    fi

    # --- Configure Second PostgreSQL Database (for Users) ---
    # This block is independent of the first, allowing for flexible configuration.
    if [ -n "$POSTGRES_USER_DB" ] && [ -n "$POSTGRES_USER_USER" ] && [ -n "$POSTGRES_USER_PASSWORD" ]; then
        echo "   -> User PostgreSQL credentials found. Setting 'user_database_uri'..."
        USER_DB_URI="postgresql://${POSTGRES_USER_USER}:${POSTGRES_USER_PASSWORD}@${POSTGRES_USER_HOST:-db}:${POSTGRES_USER_PORT:-5432}/${POSTGRES_USER_DB}"
        sed -i -E "s|^(\s*#\s*)?user_database_uri:.*|user_database_uri: ${USER_DB_URI}|" "$CONFIG_FILE"
        echo -e "${C_GREEN}   ✅ User database URI configured for PostgreSQL.${C_RESET}"
    else
        echo -e "${C_YELLOW}   -> User PostgreSQL credentials not set. Skipping configuration.${C_RESET}"
    fi
}

# ==============================================================================
#                             MAIN EXECUTION
# ==============================================================================

echo -e "\n${C_BOLD}CONTAINER ENTRYPOINT SCRIPT STARTED${C_RESET}"

# 1. Run all configuration functions to prepare the environment.
configure_qdrant
configure_database

# 2. Determine the application mode from the first command-line argument ($1).
#    If no argument is given, default to 'flask-api'.
MODE=${1:-flask-api}

echo -e "\n${C_BLUE}==================================================${C_RESET}"
echo -e "${C_BOLD}🚀 LAUNCHING APPLICATION: [${MODE}]${C_RESET}"
echo -e "${C_BLUE}==================================================${C_RESET}\n"

# CRITICAL: Use 'exec' to replace this script's process with the application's.
#
# WHY THIS IS ESSENTIAL: In Docker, the container's main process runs as PID 1.
# This process is special: it's responsible for receiving signals from the
# Docker daemon (like SIGTERM when you run `docker stop`).
#
# If you don't use 'exec', this shell script remains PID 1. When `docker stop`
# sends a signal, it goes to the script, which may not forward it to the
# child application process (gunicorn, python, etc.). The result is that the
# application doesn't shut down gracefully, and Docker has to forcibly kill it
# after a timeout.
#
# `exec` solves this by making the application the container's PID 1, allowing
# it to receive signals directly from Docker and perform a clean shutdown.

case "$MODE" in
    flask-api)
        echo "--> Starting Flask API on port 5001 with Gunicorn..."
        exec gunicorn --bind 0.0.0.0:5001 --workers 4 --timeout 120 --preload "api_server.run_api_server:create_app()"
        ;;
    fastapi-api)
        echo "--> Starting FastAPI on port 8000 with Gunicorn/Uvicorn..."
        exec gunicorn -k uvicorn.workers.UvicornWorker --workers 4 --bind 0.0.0.0:8000 "fastapi_app.main:app"
        ;;
    gui)
        echo "--> Starting Streamlit GUI for Flask on port 8501..."
        exec streamlit run gui_app/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
        ;;
    fastapi-gui)
        echo "--> Starting Streamlit GUI for FastAPI on port 8502..."
        exec streamlit run fastapi_gui/main.py --server.port 8502 --server.address 0.0.0.0 --server.headless true
        ;;
    cli)
        echo "--> Executing Core CLI (find_sim_images.py)..."
        shift # Removes 'cli' from the argument list.
        # '$@' passes all remaining arguments to the script, correctly handling spaces.
        exec python find_sim_images.py "$@"
        ;;
    manage)
        echo "--> Executing Management CLI (e.g., migrations, user creation)..."
        shift
        exec python manage.py "$@"
        ;;
    backup)
        echo "--> Executing Backup Tool (backup_tool.py)..."
        shift
        exec python backup_tool.py "$@"
        ;;
    bash | sh)
        echo "--> Entering interactive shell for debugging..."
        # Use /bin/sh for maximum portability across base images.
        exec /bin/sh
        ;;
    *)
        # Provide a helpful error message if an unknown command is given.
        # '>&2' redirects this output to Standard Error, which is the correct
        # stream for error messages.
        echo -e "${C_RED}❌ ERROR: Unknown command '$MODE'${C_RESET}" >&2
        echo "       Available commands: flask-api, fastapi-api, gui, fastapi-gui, cli, manage, backup, bash, sh" >&2
        exit 1 # Exit with a non-zero status code to indicate failure.
        ;;
esac
```

#### 📜 Detailed Breakdown: `docker-entrypoint.sh`

*   **`#!/bin/sh`**: Specifies the script should be run with the standard Bourne shell, ensuring portability across different Linux base images.
*   **`set -e`**: A critical safety command. It tells the script to exit immediately if any command fails (returns a non-zero exit code). This prevents the container from starting in a partially configured or broken state.
*   **`configure_qdrant()` / `configure_database()`**: The script is organized into functions for clarity and reusability. Each function is responsible for configuring one part of the system.
*   **Dynamic Configuration with `sed`**: The functions use `sed` (stream editor), a powerful command-line tool for text manipulation. They read environment variables (e.g., `$QDRANT_MODE`) and use `sed` to find and replace lines in the application's `.yaml` configuration files. This is what allows a single, generic Docker image to be configured at runtime for many different deployment scenarios.
*   **Idempotency**: The `sed` commands are written carefully to be "idempotent". This means they can be run multiple times without causing problems. For a full explanation, see Appendix A.5.
*   **Docker Secrets Support**: The `configure_database` function first checks if a secret file exists at `/run/secrets/postgres_password`. If it does, it reads the password from there. This allows the system to seamlessly use the more secure Docker Secrets mechanism without changing the rest of the logic.
*   **`MODE=${1:-flask-api}`**: This line reads the first argument passed to the script (which comes from the `command:` in `docker-compose.yaml`). If no argument is provided, it defaults to `flask-api`.
*   **`case "$MODE" in ... esac`**: This is a `case` statement (similar to a `switch` statement in other languages). It inspects the `$MODE` variable and executes the appropriate block of code to launch the correct application process.
*   **`exec ...`**: This is arguably the most important command in the script. `exec` **replaces** the current process (the shell script) with the new process (e.g., `gunicorn`). This is vital because it makes the application server the main process (PID 1) inside the container, allowing it to correctly receive shutdown signals from Docker (like when you run `docker stop`). Without `exec`, the container might not shut down gracefully.
*   **`shift` and `"$@"`**: In the `cli` and `manage` modes, `shift` removes the first argument (`cli` or `manage`) from the list of arguments. `"$@"` then passes all *remaining* arguments to the Python script, preserving spaces and quotes correctly. This allows you to run commands like `docker-compose run cli_runner manage --create-user "Jeevan Kumar"`.

### 1.6. The Database Magician: `init-multiple-databases.sh` Explained

The official `postgres` Docker image has a fantastic feature: any `.sh`, `.sql`, or `.sql.gz` files placed in the `/docker-entrypoint-initdb.d` directory will be executed automatically the **first time** the container starts with an empty data directory. This script leverages that feature to create multiple databases from a single container instance.

```bash
#!/bin/bash
# ==============================================================================
#       PostgreSQL Multi-Database Initialization Script
# ==============================================================================
# This script is executed automatically by the official Postgres image on its
# first run. It creates not only the primary database (via POSTGRES_DB) but
# also a secondary database for user authentication, if defined.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Function to create a database ---
# This function checks if a database name is provided. If so, it uses the psql
# tool to connect to the default 'postgres' database and execute a 'CREATE DATABASE'
# command. The `\gexec` command in psql is a powerful way to execute dynamically
# generated SQL.
create_database() {
  local db_name=$1
  if [ -n "$db_name" ]; then
    echo "  -> Creating database '$db_name'..."
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "postgres" <<-EOSQL
      SELECT 'CREATE DATABASE ' || quote_ident('$db_name')
      WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$db_name')\gexec
EOSQL
    echo "  ✓ Database '$db_name' created or already exists."
  fi
}

# --- Main Execution ---
echo "▶️  Running multi-database initialization script..."

# 1. The main database specified by POSTGRES_DB is created automatically
#    by the entrypoint of the official postgres image before this script runs.
#    We just add a log message for clarity.
if [ -n "$POSTGRES_DB" ]; then
  echo "  ✓ Primary database '$POSTGRES_DB' handled by default mechanism."
fi

# 2. We call our function to create the second database for users.
#    The name of this database is passed in via the POSTGRES_USER_DB_NAME
#    environment variable, which we set in docker-compose.yaml.
create_database "$POSTGRES_USER_DB_NAME"

echo "✅ Multi-database initialization complete."
```

#### 📜 Detailed Breakdown: `init-multiple-databases.sh`

*   **Execution Context**: This script is mounted into the `/docker-entrypoint-initdb.d/` directory of the `postgres` container. The official postgres entrypoint script will automatically execute it after it has finished its own basic setup (like creating the main user and database specified by `POSTGRES_USER` and `POSTGRES_DB`).
*   **`psql -v ON_ERROR_STOP=1 ...`**: This invokes the PostgreSQL command-line client, `psql`, which is available inside the container. It connects as the superuser (`$POSTGRES_USER`) to the maintenance database (`postgres`).
*   **`<<-EOSQL ... EOSQL`**: This is a "here document". It feeds the text between the two `EOSQL` markers as standard input to the `psql` command.
*   **`SELECT ... WHERE NOT EXISTS ... \gexec`**: This is a clever and robust SQL command.
    *   It constructs a `CREATE DATABASE` string dynamically.
    *   The `WHERE NOT EXISTS` clause checks if a database with that name already exists. This makes the script idempotent—if you somehow run it again, it won't fail trying to create a database that's already there.
    *   `\gexec` is a special `psql` meta-command that executes the query, then treats the *result* of that query as a new SQL statement to execute. If the `WHERE` clause finds the database, the query returns no rows, and nothing is executed. If it doesn't find the database, the query returns the `CREATE DATABASE...` string, which `\gexec` then runs.

### 1.7. The Master Control Panel: `.env` File Explained

This file is the single point of configuration for any deployment. It allows you to define your desired architecture, set security keys, and provide database credentials without ever modifying the core `docker-compose.yaml` files. It is the heart of the system's flexibility. A `.env.template` file should be provided in the project, which users copy to `.env` to fill out.

**The `.env` file must NOT be committed to version control (e.g., Git) as it contains sensitive credentials.** The `.gitignore` file should always contain an entry for `.env`.

```ini
# ==============================================================================
#            Master Configuration for the JK Image Similarity System
# ==============================================================================
# Copy this file to .env and fill in your values.
# It acts as the master control panel. Edit the values here to change the
# behavior and architecture of your deployment.
# ==============================================================================


# ==============================================================================
#                      --- 1. APPLICATION & SECURITY ---
# ==============================================================================
# These variables control core application behavior and security settings.

# --- Cryptographic Secret Key ---
# Used by web frameworks for securely signing session cookies.
# 🔒 CRITICAL: For production, this MUST be replaced with a long, unpredictable,
# cryptographically secure random string.
SECRET_KEY="replace_this_with_a_real_secret_key"

# --- API Keys ---
# 🔒 CRITICAL: For production, this should be replaced with newly generated keys.
ADMIN_API_USER_KEY="replace_this_with_a_real_api_key"

# ==============================================================================
#                      --- 2. DATABASE ARCHITECTURE SWITCHES ---
# ==============================================================================

# --- MASTER QDRANT SWITCH (See Appendix A.13 for details) ---
# This variable tells your APPLICATION code how to interact with Qdrant.
# - "server":   Connect to a Qdrant server instance (internal or external).
# - "embedded": Run Qdrant as a library within the application container.
QDRANT_MODE=server

# --- PostgreSQL Configuration (for Search Results Database) ---
# To use PostgreSQL for search results, fill in these variables.
# To use SQLite instead, leave these THREE variables blank or comment them out.
POSTGRES_DB=image_similarity_results_db
POSTGRES_USER=jeevan
POSTGRES_PASSWORD=a_strong_password_here

# --- PostgreSQL Configuration (for User Authentication Database) ---
# To use PostgreSQL for user data, fill in these variables.
# To use the application's default for users, leave these THREE blank.
POSTGRES_USER_DB=image_similarity_users_db
POSTGRES_USER_USER=jeevan
POSTGRES_USER_PASSWORD=a_strong_password_here

# ==============================================================================
#                      --- 3. EXTERNAL HOST OVERRIDES (ADVANCED) ---
# ==============================================================================
# Use these variables to connect to EXISTING database servers running outside
# of this Docker Compose stack. If these are commented out, the application will
# attempt to connect to the internal services ('db', 'qdrant').
# See Appendix A.4 for details on `host.docker.internal`.

# --- External PostgreSQL Server Override ---
# To connect to an existing PostgreSQL server, uncomment BOTH lines and set
# its IP/hostname and PORT.
# POSTGRES_HOST_EXTERNAL=192.168.1.100
# POSTGRES_PORT_EXTERNAL=5432

# --- External Qdrant Server Override ---
# To connect to an existing Qdrant server, uncomment BOTH lines and set
# its IP/hostname and PORT.
# QDRANT_HOST_EXTERNAL=host.docker.internal
# QDRANT_PORT_EXTERNAL=6333

# --- External Flask API Server Override ---
# To connect a GUI to an existing Flask API, uncomment BOTH lines.
# FLASK_API_HOST_EXTERNAL=host.docker.internal
# FLASK_API_PORT_EXTERNAL=5001

# --- External FastAPI Server Override ---
# To connect a GUI to an existing FastAPI, uncomment BOTH lines.
# FASTAPI_API_HOST_EXTERNAL=host.docker.internal
# FASTAPI_API_PORT_EXTERNAL=8000
```

***

## Part 2: The Developer's Handbook: Building and Packaging

This part of the guide is for the software development team responsible for maintaining the application code, building the final Docker images, and creating the distributable package for clients.

### 2.1. Prerequisite: Setting Up Your Development Environment

1.  **Docker Engine & Docker Compose**: The core runtime and orchestration tool.
2.  **A Text Editor or IDE**: Such as Visual Studio Code, which has excellent Docker integration.
3.  **Git**: For version control.
4.  **A Shell Environment**: Such as Bash, Zsh, or PowerShell.

### 2.2. Automating Secure Configuration (`.env`)

To prevent the accidental use of default credentials in development or production, it is a best practice to provide a script that generates a secure `.env` file from the template.

➡️ **Action:** Create a file named `generate_env.sh` with the following content:

```bash
#!/bin/bash
# ==============================================================================
#           Secure .env File Generator
# ==============================================================================
# This script creates a .env file from the .env.template and populates it
# with new, cryptographically secure random values for secrets.
# ==============================================================================

set -e
ENV_FILE=".env"
TEMPLATE_FILE=".env.template"

if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "Error: $TEMPLATE_FILE not found."
    exit 1
fi

if [ -f "$ENV_FILE" ]; then
    read -p "Warning: '$ENV_FILE' already exists. Overwrite? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

cp "$TEMPLATE_FILE" "$ENV_FILE"

# Generate a new SECRET_KEY
# This command must be compatible with both Linux and macOS sed
if [[ "$(uname)" == "Darwin" ]]; then # macOS
    SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
    sed -i '' "s|SECRET_KEY=.*|SECRET_KEY=${SECRET_KEY}|" "$ENV_FILE"
    ADMIN_KEY=$(python3 -c 'import secrets; print("jk_pub_" + secrets.token_hex(32))')
    sed -i '' "s|ADMIN_API_USER_KEY=.*|ADMIN_API_USER_KEY=${ADMIN_KEY}|" "$ENV_FILE"
else # Linux
    SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
    sed -i "s|SECRET_KEY=.*|SECRET_KEY=${SECRET_KEY}|" "$ENV_FILE"
    ADMIN_KEY=$(python3 -c 'import secrets; print("jk_pub_" + secrets.token_hex(32))')
    sed -i "s|ADMIN_API_USER_KEY=.*|ADMIN_API_USER_KEY=${ADMIN_KEY}|" "$ENV_FILE"
fi

echo "✅ Successfully created a secure '$ENV_FILE' file."
echo "   Please review it and adjust any non-secret settings (like database passwords)."
```

Make it executable with `chmod +x generate_env.sh` and run it with `./generate_env.sh`.

### 2.3. Workflow A: The Live-Reload Development Loop

For active, day-to-day development, you need a fast feedback loop. The goal is to make a change in your Python code and see the result immediately without rebuilding the entire Docker image. Our development override files are designed for exactly this "hot-reloading" workflow.

➡️ **Action:** To start your full development environment, run the following command from the root of the project directory:

```bash
# For Debian/Ubuntu-based development
# This command merges the base and dev compose files, activates all relevant
# profiles, and builds the special development image.
docker-compose -f docker-compose.yaml -f docker-compose.dev.yaml --profile postgres --profile qdrant --profile gui --profile fastapi_gui up --build -d

# For RHEL-based development
docker-compose -f docker-compose.rhel.yml -f docker-compose.dev.rhel.yml --profile postgres --profile qdrant --profile gui --profile fastapi_gui up --build -d
```

**What this command does:**

*   It merges the configurations, with `docker-compose.dev.yaml` overriding settings from the base file.
*   The `build:` directive in the dev file takes precedence, building your code locally.
*   The `volumes:` directive in the dev file mounts your local source code directory (`.`) directly into the container.
*   The `command:` directive in the dev file starts the Python web servers in `--debug` or `--reload` mode.

Now, you can edit any `.py` file on your local machine. The running server inside the container will detect the file change and automatically restart, reflecting your changes instantly.

#### Using the Interactive Development Shell

For exploratory work, debugging, or running multiple commands, you can start an interactive shell inside the container. This is the equivalent of activating a Python virtual environment.

➡️ **Action:** To get an interactive shell, run:

```bash
# This starts a temporary container and drops you into a bash prompt
docker-compose run --rm cli_runner bash
```

From this shell, you can now run any command (e.g., `python manage.py --help`, `ls -l`, `pip freeze`) and see the live-synced files from your host machine. Type `exit` to close the session and automatically remove the container.

### 2.4. Workflow B: "Baking" the Final Production Package

This workflow is performed **only when development of a new version is complete** and you are ready to create a secure, immutable, and portable package for the client. This is where you take your finished code and "bake" it into a final production image.

#### Step 1: Versioning

First, update the version number in the `image:` tag within your `docker-compose.yaml` and `docker-compose.rhel.yml` files. Adhere to semantic versioning (e.g., `v1.1.0`).
Example: `image: jk-image-similarity-app:v1.1.0`

#### Step 2: Build the Production Image

From the project root, run the `docker build` command. Use the `-t` flag to tag the image with its name and new version number.

```bash
# Build the Debian-based production image
docker build -f Dockerfile -t jk-image-similarity-app:v1.1.0 .

# Build the RHEL-based production image
docker build -f Dockerfile.rhel -t jk-image-similarity-app:rhel-v1.1.0 .
```

#### Step 3: Save Images into Air-Gap Bundles

For deployment in secure environments without internet access, we must save our application image and all its third-party dependencies (Postgres, Qdrant) into portable `.tar` archives. We will create two separate bundles: one for Debian-based systems and one for RHEL.

➡️ **Action:** Run the following commands to create the two bundles:

```bash
# Create the Debian bundle
echo "Creating Debian/Ubuntu package..."
docker save \
jk-image-similarity-app:v1.1.0 \
qdrant/qdrant:v1.9.1-avx-disabled \
postgres:15-alpine \
-o jk-image-similarity-system-debian-v1.1.0.tar

# Create the RHEL bundle
echo "Creating RHEL package..."
docker save \
jk-image-similarity-app:rhel-v1.1.0 \
qdrant/qdrant:v1.9.1-avx-disabled \
postgres:15-alpine \
-o jk-image-similarity-system-rhel-v1.1.0.tar
```

#### Step 4: Assemble the Client Package

Create a distribution directory (e.g., `dist/`) and copy all the necessary files into it. The client should receive a clean, self-contained package.

```
dist/
├── jk-image-similarity-system-debian-v1.1.0.tar  (Or the RHEL version)
├── docker-compose.yaml
├── docker-compose.rhel.yml
├── .env.template
├── generate_env.sh                               (Optional but helpful for clients)
├── README.md                                     (This master guide document)
└── postgres-init/
    └── init-multiple-databases.sh
```

Finally, zip or tar this `dist` directory. This archive is the final product you deliver to the client.

***

## Part 3: The Client's Deployment Guide: Step-by-Step Installation

This guide is for the end-user or system administrator responsible for deploying the application. It assumes you have received the distribution package and have Docker and Docker Compose installed on your target machine.

### 3.1. Prerequisites & Package Contents

1.  A host machine with **Docker** and **Docker Compose** installed.
2.  The application package (e.g., `dist.zip`).

Unzip the package to begin. You will have a directory structure containing the files listed in Step 4 of the developer's handbook above.

### 3.2. Step 1: Load the Application into Docker

This command loads all the necessary images from the `.tar` archive into your local Docker registry. This only needs to be done once per machine, or when updating to a new version.

➡️ **Action:** In your terminal, navigate to the unzipped directory and run:

```bash
# Use the filename provided in your package
docker load -i jk-image-similarity-system-debian-v1.1.0.tar
```

You can verify the images are loaded by running `docker images`. You should see `jk-image-similarity-app`, `postgres`, and `qdrant` in the list.

### 3.3. Step 2: Configure Your Deployment

The `.env` file is your control panel. This is the only file you should need to edit.

➡️ **Action:**

1.  **Create your configuration:** `cp .env.template .env`
2.  **Open `.env` in a text editor.**
3.  **Set Security Keys:** **CRITICAL:** Replace the placeholder `SECRET_KEY` and `ADMIN_API_USER_KEY` with new, secure random values.
4.  **Configure Architecture:** Set `QDRANT_MODE` and fill in `POSTGRES_` variables according to your desired setup.

#### Quick Start Configurations

To simplify setup, here are pre-filled examples for common scenarios.

*   **For "All-in-One" Mode (Recommended for Production):**
    *   Runs PostgreSQL and Qdrant as separate, high-performance services.
    ```ini
    # .env for All-in-One Mode
    SECRET_KEY="your_secure_random_string_here"
    ADMIN_API_USER_KEY="your_secure_api_key_here"
    QDRANT_MODE=server
    POSTGRES_DB=image_similarity_results_db
    POSTGRES_USER=jeevan
    POSTGRES_PASSWORD=a_strong_password
    POSTGRES_USER_DB=image_similarity_users_db
    POSTGRES_USER_USER=jeevan
    POSTGRES_USER_PASSWORD=a_strong_password
    ```

*   **For "Embedded" Mode (Lightweight / Simple Testing):**
    *   Runs Qdrant as a library and uses SQLite. No separate database containers needed.
    ```ini
    # .env for Embedded Mode
    SECRET_KEY="your_secure_random_string_here"
    ADMIN_API_USER_KEY="your_secure_api_key_here"
    QDRANT_MODE=embedded
    # Leave all POSTGRES variables blank or commented out
    # POSTGRES_DB=
    # POSTGRES_USER=
    # POSTGRES_PASSWORD=
    ```

### 3.4. Step 3: Choose Your Architecture & Start the System

Your startup command now depends on which services you want to run *inside* Docker. The `-d` flag runs the system in the background (detached mode).

➡️ **Action:** In the same directory, choose the command that matches your **Host Operating System** and desired **Architecture**.

---

#### For Standard Debian/Ubuntu Systems:

*   **Scenario A: All-in-One (Internal PostgreSQL & Internal Qdrant Server)**
    *   This is the full-stack, batteries-included deployment.
    *   Your `.env` should have `QDRANT_MODE=server` and PG credentials filled out.
    ```bash
    docker-compose --profile postgres --profile qdrant --profile gui --profile fastapi_gui up -d
    ```

*   **Scenario B: Embedded Mode (SQLite & Embedded Qdrant)**
    *   This is the simplest, lightweight deployment with no separate database containers.
    *   Your `.env` should have `QDRANT_MODE=embedded` and PG credentials blank.
    ```bash
    docker-compose --profile gui --profile fastapi_gui up -d
    ```

---

#### For Red Hat Enterprise Linux (RHEL) Systems:

*   **Scenario A: All-in-One (Internal PostgreSQL & Internal Qdrant Server)**
    *   This is the full-stack, batteries-included deployment.
    *   Your `.env` should have `QDRANT_MODE=server` and PG credentials filled out.
    ```bash
    docker-compose -f docker-compose.rhel.yml --profile postgres --profile qdrant --profile gui --profile fastapi_gui up -d
    ```

*   **Scenario B: Embedded Mode (SQLite & Embedded Qdrant)**
    *   This is the simplest, lightweight deployment with no separate database containers.
    *   Your `.env` should have `QDRANT_MODE=embedded` and PG credentials blank.
    ```bash
    docker-compose -f docker-compose.rhel.yml --profile gui --profile fastapi_gui up -d
    ```

---

*   **Scenario C: Hybrid Mode (External PostgreSQL & Internal Qdrant Server)**
    *   Use this when you want to connect to an existing corporate PostgreSQL database but run Qdrant locally.
    *   Set `POSTGRES_HOST_EXTERNAL` in `.env` and `QDRANT_MODE=server`.
    *   Run the same command as Scenario A, but **without** the `--profile postgres` flag:
    ```bash
    docker-compose --profile qdrant --profile gui --profile fastapi_gui up -d
    ```

*   **Scenario D: Fully External (Connect to existing PG and Qdrant servers)**
    *   Use this to integrate the application logic with fully managed external databases.
    *   Set `POSTGRES_HOST_EXTERNAL` and `QDRANT_HOST_EXTERNAL` in `.env`.
    *   **No database profiles are needed** as they are not being started by Compose.
    ```bash
    docker-compose --profile gui --profile fastapi_gui up -d
    ```

### 3.5. Step 4: Accessing the Services

Once the containers are running, you can access the various front-ends and APIs:

*   **Flask API Swagger UI:** `http://<your_host_ip>:5001/api/v1/docs`
*   **FastAPI Swagger UI:** `http://<your_host_ip>:8000/docs`
*   **Flask GUI (Streamlit):** `http://<your_host_ip>:8501`
*   **FastAPI GUI (Streamlit):** `http://<your_host_ip>:8502`
*   **Qdrant Web UI:** `http://<your_host_ip>:6333/dashboard` (Only available if you started with the `qdrant` profile)

### 3.6. Step 5: Verify the System is Running

To check the status of your running containers:

```bash
docker-compose ps
```

You should see all the services you started in a `running` or `Up` state. Check the `STATUS` column for health information (e.g., `(healthy)`). Any service marked as "unhealthy" or "Exited" requires investigation using the troubleshooting commands in Appendix A.7.

### 3.7. Step 6: Using the Command-Line Tools (CLI)

The `cli_runner` service is your dedicated tool for running one-off tasks like database management, backups, or running direct, non-API-based scripts. The key is to use `docker-compose run`.

➡️ **Action:** Choose the command that matches your task.

*   **For CLI tasks that use SQLite (no external DBs needed):**
    *   You do not need to start any services beforehand.
    ```bash
    # This command starts a temporary container, runs the script, and then disappears.
    # Note: Replace arguments with your actual script's needs.
    docker-compose run --rm cli_runner cli --action find_similar --image-path /home/appuser/app/downloads/my_image.jpg
    ```

*   **For CLI tasks that require PostgreSQL and/or Qdrant:**
    *   You must first ensure the database services are running (see Step 3.4).
    ```bash
    # Step 1: Ensure the databases are running.
    # docker-compose --profile postgres --profile qdrant up -d

    # Step 2: Run your command. Docker Compose will connect the temporary
    # cli_runner container to the running databases.
    docker-compose run --rm cli_runner manage --create-user --username=new_admin --role=admin
    ```

### 3.8. Step 7: Stopping the System

To safely stop and remove all running services, networks, and containers for this project, use the `down` command. **You do not need to specify profiles for `down`**; it intelligently stops all containers associated with the project.

```bash
docker-compose down
```

To stop the services without removing them (so they start faster next time), use `docker-compose stop`.

### 3.9. Step 8: Updating the Application (The Basic Workflow)

This section covers the basic update process. For a robust production workflow involving database changes, please see the critical **Part 4.1**.

1.  Receive the new application package (e.g., `jk-image-similarity-system-v1.2.0.zip`).
2.  Stop the current running application: `docker-compose down`.
3.  Load the new images from the new `.tar` file: `docker load -i jk-image-similarity-system-v1.2.0.tar`.
4.  Replace the old `docker-compose.yaml`, `.env.template`, etc., with the new ones from the package. Review your `.env` file to see if any new variables need to be set by comparing it with the new `.env.template`.
5.  Start the new version using the appropriate `up` command from Step 3.4.

### 3.10. Step 9: Post-Deployment Verification

After starting the system, you can quickly test if the core services are responding.

➡️ **Action:** Run these `curl` commands from your host machine's terminal.

```bash
# Test the Flask API health endpoint
curl http://localhost:5001/health

# Test the FastAPI health endpoint
curl http://localhost:8000/health
```

A successful response will typically be a simple JSON message like `{"status":"ok"}` and an HTTP 200 status code. If you get a "Connection refused" error, the service may still be starting or may have failed. Check the logs with `docker-compose logs <service_name>`.

***

## Part 4: Production Operations & Best Practices

Deploying the application is just the beginning. This section provides essential guidance for maintaining, updating, securing, and scaling the system in a live production environment.

### 4.1. 📈 Production-Grade Update Strategy: Database Migrations

Updating a live application requires a clear strategy, especially when the database schema changes.

**The Challenge:** A new application version (e.g., v1.2.0) might require a different database table structure than the previous version (v1.1.0). Simply launching the new code against the old database can cause crashes or data corruption.

**The Solution:** Use a dedicated database migration tool like **Alembic** (for SQLAlchemy) or **Flyway**. These tools version your database schema in code, just as Git versions your application code. This allows you to apply changes incrementally and roll them back if needed.

#### The Professional Migration Workflow

This workflow ensures that the database schema is updated to be compatible with the new code *before* the application starts serving traffic, minimizing downtime and risk.

*   **Step 1: 🚨 Backup Your Data!**
    Before any major change, always create a full backup of your database volumes. See **Section 4.2** for detailed instructions. This is your safety net.

*   **Step 2: Start Only the Database Service**
    Bring up the PostgreSQL container so the migration tool can connect to it.
    ```bash
    docker-compose --profile postgres up -d
    ```

*   **Step 3: Run the Migration Command**
    Use the `cli_runner` service to execute your migration script. This starts a temporary container using the **new** application image and connects it to the **existing, running** database to apply the schema changes.
    ```bash
    # This example assumes your application's management script handles migrations.
    # Replace 'manage --migrate' with your actual migration command.
    docker-compose run --rm cli_runner manage --migrate
    ```

*   **Step 4: Shut Down and Restart the Full System**
    Once the migration is successful, you can bring down the database-only instance and launch the complete, updated application stack.
    ```bash
    docker-compose down
    docker-compose --profile postgres --profile qdrant --profile gui --profile fastapi_gui up -d
    ```

### 4.2. 🗄️ Critical Operations: Backup and Recovery Strategy

Data is your most valuable asset. A robust, tested backup and recovery strategy is non-negotiable for any production environment.

#### Strategy 1: Full Volume Backup (Disaster Recovery)

This method creates a complete, byte-for-byte backup of a volume's contents into a compressed archive. It is the most reliable way to perform a full system restore.

➡️ **Action: Backing Up Named Volumes**
This command runs a temporary `busybox` container, mounts the volume you want to back up (`postgres_data`), mounts your current host directory as `/backup`, and creates a timestamped, compressed archive of the volume's contents.

```bash
# Backup the PostgreSQL data volume
docker run --rm -v postgres_data:/data -v $(pwd):/backup busybox tar czf /backup/postgres_data_backup_$(date +%F).tar.gz -C /data .

# Backup the Qdrant data volume
docker run --rm -v qdrant_data:/data -v $(pwd):/backup busybox tar czf /backup/qdrant_data_backup_$(date +%F).tar.gz -C /data .
```

➡️ **Action: Restoring from a Volume Backup**
This command performs the reverse operation, unpacking an archive into a (typically new or empty) named volume.

```bash
# 1. Ensure all services are stopped to prevent data corruption
docker-compose down

# 2. (Optional) If restoring to a fresh system, create the empty volume
# docker volume create postgres_data

# 3. Restore the PostgreSQL data from a backup file
docker run --rm -v postgres_data:/data -v $(pwd):/backup busybox tar xzf /backup/postgres_data_backup_YYYY-MM-DD.tar.gz -C /data

# 4. Start the services again with the restored data
docker-compose up -d
```

#### Strategy 2: Logical Backup (Data Portability)

This uses an application-level tool (e.g., `backup_tool.py` or `pg_dump`) to export data in a structured format like SQL, JSON, or CSV. This is useful for migrating data between different database versions or types, but is generally slower for full disaster recovery.

```bash
# Example of running a logical backup tool via the cli_runner
docker-compose run --rm cli_runner backup --output /backup/logical_db_export.json

# Example using the native pg_dump tool
docker-compose exec db pg_dump -U ${POSTGRES_USER} -d ${POSTGRES_DB} > db_dump.sql
```

### 4.3. 🔐 Security Hardening & Advanced Secrets Management

Storing secrets in `.env` files is acceptable for development but poses a risk in production. For high-security or air-gapped environments, **Docker Secrets** is the recommended solution.

**How It Works:** Docker Secrets stores sensitive data in a secure, encrypted raft log (in Swarm mode) or on the host filesystem with tight permissions. It mounts secrets into containers as in-memory files at `/run/secrets/<secret_name>`, which is much more secure than environment variables that can be inspected.

#### Workflow for Using Docker Secrets:

*   **Step 1: Create a Docker Secret on the Host**
    ```bash
    # The '-' reads the secret from stdin, preventing it from appearing in shell history
    echo "your-super-secret-password" | docker secret create postgres_password -
    ```
*   **Step 2: Update `docker-compose.yaml` to Use the Secret**
    The official `postgres` image has built-in support for reading a password from a file.
    ```yaml
    services:
      db:
        # ... other settings
        environment:
          # Remove POSTGRES_PASSWORD from here
          POSTGRES_USER: ${POSTGRES_USER}
          POSTGRES_DB: ${POSTGRES_DB}
          # Instead, point to the file where the secret will be mounted
          POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
        secrets:
          - postgres_password # Grant this service access to the secret
    
    # Declare the secret as externally managed
    secrets:
      postgres_password:
        external: true
    ```
*   **Step 3: Modify Your Application to Use Secrets**
    For your own applications, you must modify the entrypoint script to prefer secrets over environment variables. Our `docker-entrypoint.sh` in Section 1.5 already includes this logic:
    ```bash
    # From docker-entrypoint.sh
    if [ -f /run/secrets/postgres_password ]; then
      POSTGRES_PASSWORD=$(cat /run/secrets/postgres_password)
    fi
    ```

### 4.4. 📊 System Observability: Monitoring and Logging

You can't manage what you can't see. Observability is key to reliability and performance tuning.

#### Monitoring (Metrics)

*   **What it is:** Collecting time-series data about your system's performance (CPU, memory, request latency, error rates).
*   **Basic Tool:** `docker stats` gives a live, real-time view of container resource usage.
*   **Production Standard:** Deploy a dedicated monitoring stack like **Prometheus & Grafana**.
    *   **Prometheus:** Scrapes metrics from your services (which can be configured to expose them on a `/metrics` endpoint).
    *   **Grafana:** Provides powerful, beautiful dashboards to visualize these metrics over time, set up alerts, and gain deep insights.

#### Logging (Events)

*   **What it is:** Collecting, aggregating, and analyzing the event logs produced by all your services.
*   **Basic Tool:** `docker-compose logs -f <service_name>` is fine for debugging a single service.
*   **Production Standard:** Configure Docker to send logs to a **centralized logging aggregator** like the **ELK Stack** (Elasticsearch, Logstash, Kibana), **Grafana Loki**, or **Fluentd**. This allows you to search, analyze, and alert on logs from all services in one unified interface.
*   **⭐ Best Practice: Structured Logging**
    Modify your application to output logs in a structured format like **JSON**. This makes them machine-readable and infinitely easier to parse, index, and analyze in a logging platform.

    **Plain Text Log (Hard to Parse):**
    `INFO: 2023-10-27 10:30:05 - User 'admin' logged in from 192.168.1.100`

    **Structured JSON Log (Easy to Parse):**
    `{"level": "INFO", "timestamp": "2023-10-27T10:30:05Z", "message": "User login successful", "user": "admin", "source_ip": "192.168.1.100"}`

### 4.5. 🚀 Preparing for Growth: Scaling the System

Docker Compose is excellent for single-host deployments. When you need high availability or need to handle more traffic than a single machine can manage, you must move to an orchestration platform.

#### Scaling Strategies

*   **Vertical Scaling:** Upgrade the host machine to have more CPU, RAM, or faster storage. This is the simplest approach but has physical limits.
*   **Horizontal Scaling (Stateless Services):** Services like the APIs and GUIs are stateless and can be scaled horizontally. You simply run more instances (replicas) of them.
    *   **Intermediate Step:** On a single powerful host, you can use `docker-compose up --scale flask_api=3` to run three API containers. You would then need a load balancer like Nginx or Traefik to distribute traffic between them.
*   **Orchestration Platforms (Multi-Host):**
    *   **Docker Swarm:** Docker's native orchestration engine. It's the simplest transition from Compose, allowing you to manage a cluster of hosts as a single unit. You can deploy your stack with `docker stack deploy`.
    *   **Kubernetes (K8s):** The undisputed industry standard for large-scale container orchestration. It offers unparalleled power for automated scaling, self-healing, advanced deployment strategies (like blue-green and canary), and managing complex applications.
        *   **Transitioning:** The `kompose` tool (`kompose convert`) can automatically convert your `docker-compose.yaml` into a set of Kubernetes resource definitions, providing an excellent starting point for a K8s migration.

***

## Appendix: Deeper Dives & Foundational Knowledge

This appendix is designed for users who want to move beyond the "copy-paste" and achieve a deep, fundamental understanding of the technologies and design patterns used in this project. Mastering these concepts will enable you to troubleshoot complex issues, customize the system for unique requirements, and apply these best practices to your own projects.

### A.1. 🐳 Docker Fundamentals (A Primer for Beginners)

*   **Image:** A read-only blueprint for creating a container. It contains the application code, a runtime (like Python), libraries, and environment variables. Images are built from a `Dockerfile`.
*   **Container:** A runnable, isolated instance of an image. It is a lightweight process that runs on the host's kernel but has its own filesystem, network, and process space. You can run many containers from the same image.
*   **Volume:** A mechanism for persisting data generated by and used by Docker containers. Volumes are managed by Docker and exist outside the container's lifecycle, so your data is safe even if the container is removed and recreated.
*   **Network:** A private communication channel that allows containers to talk to each other. Docker Compose automatically creates a network for your project, enabling service discovery using service names (e.g., the `flask_api` container can reach the `db` container by its name).
*   **Dockerfile:** A text file that contains the step-by-step instructions for building a Docker image.
*   **Docker Compose:** A tool for defining and running multi-container Docker applications using a single YAML file (`docker-compose.yaml`). It simplifies the management of complex applications.

### A.2. Understanding the Dynamic Host Configuration (`${VAR:-default}`)

The flexibility of connecting to internal or external services is powered by a shell parameter expansion feature that Docker Compose supports in the `environment` section: `${VARIABLE:-default}`. This simple syntax is incredibly powerful.

**How It Works:**
When Docker Compose processes the `docker-compose.yaml` file, it substitutes environment variables. Let's look at this line from our file:

```yaml
environment:
  - POSTGRES_HOST=${POSTGRES_HOST_EXTERNAL:-db}
```

Compose follows this logic:

1.  **Check for `POSTGRES_HOST_EXTERNAL`:** It looks for a variable named `POSTGRES_HOST_EXTERNAL` in the shell environment where you are running `docker-compose`, and also inside the `.env` file.
2.  **If Found:** If the variable is defined and has a non-empty value (e.g., `POSTGRES_HOST_EXTERNAL=192.168.1.100` in your `.env` file), Compose uses that value. The container will receive an environment variable `POSTGRES_HOST=192.168.1.100`.
3.  **If Not Found (or Empty):** If the variable is not defined or is empty, Compose uses the `default` value provided after the `:-`. In this case, it's `db`. The container will receive `POSTGRES_HOST=db`.

**Why This Is the Gold Standard:**
This pattern allows us to have a single, clean, and stable `docker-compose.yaml` file that does not need to be edited for different environments. All the architectural "switching" is offloaded to the `.env` file, which is designed to be configurable. This separates the static definition of the application stack (the "what") from the dynamic configuration of a specific deployment (the "where"). This is a core principle of the [Twelve-Factor App methodology](https://12factor.net/config), a set of best practices for building modern, scalable applications.

### A.3. CPU Architecture Explained (AVX2 vs. AVX-Disabled)

The `qdrant/qdrant` image is a high-performance piece of software. To achieve its incredible speed, it is, by default, compiled to use modern CPU instruction sets like **AVX2 (Advanced Vector Extensions 2)**. These are special instructions built into modern CPUs that can perform mathematical operations on large arrays of numbers simultaneously (a concept called SIMD - Single Instruction, Multiple Data). This is perfect for the vector math that powers similarity search.

**The Problem:**
Not all CPUs support AVX2, especially those found in older servers, some virtual machines, or certain budget cloud instances. If you attempt to run the default Qdrant image on a CPU without AVX2 support, the container may appear to start, but the core Qdrant process will crash internally when it tries to execute an instruction that doesn't exist. This often results in a "silent failure" where the container is running but its healthcheck fails, marking it as `(unhealthy)`.

**The Solution:**
The Qdrant team provides alternative images tagged with `-avx-disabled`. These images are compiled without the AVX2-specific optimizations. They are slightly slower for massive workloads but have the massive advantage of being **universally compatible** with any x86-64 CPU.

**Our Choice:**
For this project, we have standardized on the `qdrant/qdrant:tag-avx-disabled` image in our `docker-compose.yaml` files. This prioritizes **robustness and out-of-the-box compatibility** over maximum performance. It ensures the system will work for the widest possible audience without requiring them to first diagnose their CPU architecture.

**How to Check Your CPU and Upgrade:**
If you are deploying to a production machine that you know has a modern CPU, you can squeeze out extra performance by switching back to the standard image.

1.  **Check for AVX2 support on your Linux host:**
    ```bash
    lscpu | grep -i avx2
    ```
    If this command produces output, your CPU supports AVX2.
2.  **Edit `docker-compose.yaml`:**
    Change the line under the `qdrant` service from:
    `image: qdrant/qdrant:v1.9.1-avx-disabled`
    To:
    `image: qdrant/qdrant:v1.9.1`

### A.4. Connecting to External Services (`host.docker.internal`)

When you need a Docker container to communicate with a service running on the **host machine itself** (like another Docker container managed outside your compose file, a native database install, or a service running on your laptop for testing), you cannot use `localhost` or `127.0.0.1`.

**Why `localhost` Doesn't Work:**
From inside a container, `localhost` refers to the container's own isolated network namespace, not the host's. If an application inside a container tries to connect to `localhost:5432`, it's trying to connect to port 5432 *on itself*, not on the host machine.

**The Docker-Provided Solution:**
Docker provides a special, magical DNS name for this exact purpose: `host.docker.internal`.

When a container makes a network request to `host.docker.internal`, the Docker daemon intercepts this and resolves it to the internal IP address of the host machine. This allows the container to "reach out" of its isolated network and connect to services listening on the host's network.

**Practical Example:**
Imagine you are a developer running a PostgreSQL database directly on your laptop (not in a container) for testing. It's listening on `localhost:5432`. To have your application container connect to it, you would set this in your `.env` file:

```ini
# .env
POSTGRES_HOST_EXTERNAL=host.docker.internal
```

The application inside the container will now successfully connect to the PostgreSQL instance running on your laptop. This is the recommended way to configure the `_EXTERNAL` host variables when the "external" service is, in fact, running on the same machine as the Docker daemon.

### A.5. Idempotency Explained: Why Our Scripts Are Robust

Idempotency is a computer science term that means an operation can be applied multiple times without changing the result beyond the initial application. In the context of our `docker-entrypoint.sh` script, this is a critical feature that makes it robust and reliable.

**Consider this line:**

```bash
sed -i -E "s|^(\s*#\s*)?database_uri:.*|database_uri: ${RESULTS_DB_URI}|" "$CONFIG_FILE"
```

Let's break down the regular expression `^(\s*#\s*)?database_uri:.*`:

*   `^`: Matches the beginning of the line.
*   `(\s*#\s*)?`: This is the key part.
    *   `\s*`: Matches zero or more whitespace characters (spaces, tabs).
    *   `#`: Matches the literal `#` character.
    *   `\s*`: Matches zero or more whitespace characters again.
    *   `(...)`: Groups this whole pattern together.
    *   `?`: Makes the entire group **optional**. It can match the group **zero or one time**.
*   `database_uri:`: Matches the literal text.

**The Effect:**
This regex will successfully match a line that looks like `database_uri: ...` (uncommented) AND a line that looks like `#  database_uri: ...` (commented).

**Why This Matters:**
Imagine you start, stop, and restart a container.

*   **First Run:** The `database_uri` line is commented out. The script runs, matches the commented line, and replaces it with the new, uncommented URI. The configuration is now correct.
*   **Second Run:** The `database_uri` line is now uncommented. The script runs again. Because our regex is idempotent, it correctly matches the *existing, uncommented* line and replaces it with the exact same URI. No errors are thrown, no duplicate lines are added. The configuration remains correct.

This makes the entrypoint script "stateless" and safe to run under any condition, preventing configuration drift and unexpected errors on container restarts.

### A.6. Docker Command Reference: `up`, `down`, `run`, `exec`

Understanding the subtle differences between these core Docker Compose commands is essential for effective management of the application.

*   **`docker-compose up`**
    *   **What it does:** The main command. It creates and starts containers defined in your `docker-compose.yaml`. It checks if containers from a previous run exist. If their configuration or underlying image has changed, it will stop and recreate them. If nothing has changed, it will simply start the existing containers.
    *   **When to use it:** For starting your long-running services (APIs, GUIs, databases).
    *   **Key Flags:**
        *   `-d`: **Detached Mode.** Runs the containers in the background and returns you to your terminal prompt. This is standard for production and most development. Without it, logs from all services stream to your terminal, and closing the terminal will stop the containers.
        *   `--build`: Forces Docker to build the image from the `Dockerfile` before starting the services, even if an image with the same tag already exists. Essential after you've made code changes.
        *   `--force-recreate`: Forces Compose to stop and recreate all containers, even if nothing appears to have changed. Useful for clearing out a bad state.

*   **`docker-compose down`**
    *   **What it does:** The complete opposite of `up`. It stops and **removes** all containers, networks, and (optionally) volumes associated with the project. It is the best way to perform a "full reset" of the running components.
    *   **Important Note:** By default, `down` **does not** remove named volumes. This is a safety feature to prevent you from accidentally deleting your database data.
    *   **When to use it:** When you are finished working and want to clean up all resources, or when you need to start from a completely clean slate.
    *   **Key Flags:**
        *   `-v` or `--volumes`: Tells `down` to also remove the named volumes defined in the `volumes:` section of your compose file. **USE WITH EXTREME CAUTION! This will permanently delete your data.**

*   **`docker-compose run`**
    *   **What it does:** Starts a **new, one-off container** to run a specific command and then exit. It is designed for short-lived tasks. It does *not* use the port mappings from the compose file, to avoid conflicts with long-running services.
    *   **When to use it:** For all CLI tasks: running database migrations, using management scripts, creating backups, or running a direct, non-API script. This is the command you use with our `cli_runner` service.
    *   **Key Flags:**
        *   `--rm`: **Highly Recommended.** Automatically removes the container after it finishes its task. This prevents your system from filling up with hundreds of stopped, used-once containers.
        *   `-e`: Set an environment variable for just this one run.
        *   `-w`: Set the working directory for the command.

*   **`docker-compose exec`**
    *   **What it does:** Executes a command **inside an already-running container**. This is your window into a live service.
    *   **When to use it:** Primarily for debugging. It is incredibly powerful for inspecting the live state of a running service.
    *   **Example Usage:**
        *   Get an interactive shell inside the `flask_api` container: `docker-compose exec flask_api bash`
        *   List files in a directory inside the container: `docker-compose exec flask_api ls -l /home/appuser/app/instance`
        *   Run a diagnostic command inside the Postgres container: `docker-compose exec db psql -U jeevan -d image_similarity_results_db -c "SELECT * FROM users;"`

### A.7. Essential Troubleshooting Commands

When things go wrong, these commands are your best friends.

*   **View Logs for a Specific Service:**
    If a service is unhealthy or not starting, its logs are the first place to look. The `-f` flag "follows" the log output in real-time.
    ```bash
    # See the logs for the qdrant container
    docker-compose logs -f qdrant
    
    # See logs for all services defined in the active profiles
    docker-compose --profile postgres --profile qdrant logs -f
    ```

*   **See All Containers (including stopped/crashed ones):**
    If a container starts and then immediately stops (a common problem), `docker ps` won't show it. Use `ps -a` to see all containers, their status (e.g., `Exited (1)`), and how long ago they stopped. The exit code is a crucial clue. An exit code of `0` means it finished successfully. Any non-zero code (like `1` or `137`) indicates an error.
    ```bash
    docker ps -a
    ```

*   **Enter a Running Container for Debugging:**
    This is an incredibly powerful tool. It gives you an interactive `/bin/sh` or `bash` shell inside the container's isolated environment, letting you look around as if you were SSH'd into it.
    ```bash
    # Get a shell inside the flask_api container
    docker-compose exec flask_api sh

    # From inside, you can:
    # - Check network connectivity: ping db
    # - Verify file existence: ls -l /home/appuser/app/configs
    # - Check environment variables: env
    # - See running processes: ps aux
    ```

*   **Check Port Mappings and Conflicts:**
    If you get a "port is already allocated" error, it means another process on your host machine is already using a port you're trying to map. Use these commands on the **host machine** to investigate:
    ```bash
    # For Linux/macOS
    sudo lsof -i :<port_number>
    # Example: Find what's using port 6333
    sudo lsof -i :6333

    # For Windows
    netstat -ano | findstr :<port_number>
    ```

*   **Inspect a Container's Configuration:**
    The `inspect` command dumps a massive JSON object containing every possible detail about a container, including its full configuration, network settings, volume mounts, and more. It's verbose, but contains all the answers.
    ```bash
    docker inspect <container_name_or_id>
    # Example:
    docker inspect flask_api_prod_service
    ```

### A.8. Managing Data and Volumes

It's crucial to understand how your data is being stored. This project uses **Named Volumes**, which is the Docker-recommended way to persist data generated by and used by Docker containers.

*   **What are they?** When you declare `postgres_data:` in the `volumes:` section at the bottom of the `docker-compose.yaml` file, you are telling Docker to create and manage a storage bucket for you. Docker creates this on the host machine in a special, protected directory (e.g., `/var/lib/docker/volumes/`). You don't need to know the exact location; you just refer to it by its name (`postgres_data`).

*   **Why are they better than Bind Mounts for data?**
    *   **Portability:** They are independent of the host's file structure. You can move your project folder, and the data remains safe. A bind mount (`./my-data:/data`) would break if you moved the project.
    *   **Permissions:** Docker manages permissions, avoiding complex host/container user ID mapping issues that are common with bind mounts, especially on Linux.
    *   **Management:** You can easily manage them with `docker volume` commands.
    *   **Performance:** On macOS and Windows, named volumes often have significantly better I/O performance than bind mounts from the host filesystem.
    *   **Safety:** It's much harder to accidentally delete or modify a named volume from the host machine than it is to alter a bind-mounted directory.

*   **Useful Volume Commands:**
    *   `docker volume ls`: List all named volumes on your system.
    *   `docker volume inspect <volume_name>`: See details about a volume, including its physical location on the host (`Mountpoint`).
    *   `docker volume rm <volume_name>`: **Permanently delete a volume and all its data.** Use with extreme caution! This is the command you would use to completely wipe the database and start fresh.
    *   `docker volume prune`: A helpful cleanup command that removes all "dangling" volumes (volumes not currently attached to any existing container).

Your application data (`postgres_data`, `qdrant_data`, `downloads_data`, `instance_data`, `logs_data`) is safe even when you run `docker-compose down`, which removes the containers. The data will be re-attached the next time you run `docker-compose up`.

### A.9. Deep Dive into Docker Networking

When you run `docker-compose up`, Docker Compose does something amazing behind the scenes: it creates a **custom bridge network** for your project.

*   **What is a custom bridge network?** It's a private, virtual network where all the containers in your project can reside. Think of it as a virtual ethernet switch that only your services are plugged into.

*   **Automatic Service Discovery:** The most powerful feature of this network is its built-in DNS server. Each container can find any other container on the same network simply by using its **service name**.
    *   When the `gui` container wants to talk to the `flask_api`, its code just makes a request to `http://flask_api:5001`.
    *   Docker's internal DNS resolves the hostname `flask_api` to the private IP address of the `flask_api` container on the custom network.
    *   This works regardless of which host the containers are on (in a multi-host Swarm setup) and survives container restarts, where the internal IP address might change. This is the foundation of modern microservice communication.

*   **Isolation:** By default, containers on this custom network are isolated from the host and from other Docker projects. Only the ports you explicitly publish in the `ports:` section (e.g., `"5001:5001"`) are accessible from the outside world. This provides a strong layer of security.

*   **Inspecting the Network:**
    *   `docker network ls`: See all Docker networks on your host. You'll see one named something like `myproject_default`.
    *   `docker network inspect <network_name>`: This will show you detailed information about the network, including a list of all containers connected to it and their internal IP addresses.

### A.10. Security Hardening Best Practices

This architecture already incorporates many security best practices, but it's important to understand them and know how to maintain them.

1.  **Run as Non-Root User:** We've covered this, but it cannot be overstated. All our application containers run as the unprivileged `appuser` (or UID `1001` on RHEL). This is the single most effective security measure for a container.

2.  **Use Minimal Base Images (`-slim`):** Our `Dockerfile` starts from `python:3.10-slim-bookworm`. These `-slim` images have had many non-essential tools and libraries removed. The principle is simple: if a tool doesn't exist in the container, an attacker cannot use it. This reduces the attack surface.

3.  **Multi-Stage Builds:** By using a separate "builder" stage, we ensure that no build tools (like compilers, `git`, or `curl`) are present in the final production image. An attacker who gains access to the container will find a very barren environment with few tools to leverage for further attacks.

4.  **Read-Only Mounts (`:ro`):** We mount the `configs` directory and the `postgres-init` script as read-only (`:ro`). This prevents the application, even if compromised, from maliciously modifying its own configuration files or the database initialization logic.

5.  **Secrets Management:** In a true high-security production environment, you should avoid putting secrets like passwords or API keys directly in the `.env` file. Use a secrets management tool like **Docker Secrets** (as detailed in Part 4.3), or external tools provided by your environment.
    *   **Cloud Providers (AWS, GCP, Azure):** Use their native secrets management services (AWS Secrets Manager, Google Secret Manager, Azure Key Vault).
    *   **HashiCorp Vault:** A popular open-source tool for managing secrets.
    The code would be modified to read secrets from a file path (e.g., `/run/secrets/postgres_password`) instead of an environment variable.

6.  **Regularly Scan Your Images:** Use tools like **Trivy**, **Snyk**, or **Docker Scout** to scan your final Docker image for known vulnerabilities (CVEs) in its operating system packages and language dependencies. This should be a mandatory step in any CI/CD pipeline.
    ```bash
    # Example using Trivy
    trivy image jk-image-similarity-app:v1.1.0
    ```

### A.11. Adapting for Automated CI/CD Pipelines

While this guide focuses on a manual, air-gapped workflow, the underlying principles and artifacts (`Dockerfile`, `docker-compose.yaml`) are the perfect foundation for a fully automated, internet-based **Continuous Integration/Continuous Deployment (CI/CD)** pipeline. This section explains how to adapt the workflow for tools like GitHub Actions, GitLab CI, or Jenkins.

The core difference is replacing the manual "sneakernet" transfer with an automated, network-based workflow involving three key components: a **CI Server**, a **Private Docker Registry**, and a **CD Orchestrator**.

#### 1. The CI Server (The Automated Builder)

The CI server is the heart of the automation. It automatically detects code changes (e.g., a `git push` to the `main` branch) and runs a predefined script. This script would perform the exact same build step you do manually.

**Conceptual CI Pipeline (`.github/workflows/ci.yml`):**

```yaml
name: Build and Push Production Image

on:
  push:
    branches: [ main ]

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
    - name: Check out code
      uses: actions/checkout@v3

    - name: Log in to Private Docker Registry
      uses: docker/login-action@v2
      with:
        registry: private-registry.example.com
        username: ${{ secrets.REGISTRY_USERNAME }}
        password: ${{ secrets.REGISTRY_PASSWORD }}

    - name: Build and tag the image
      run: |
        VERSION=$(# Logic to determine version, e.g., from a file or git tag)
        docker build -f Dockerfile -t private-registry.example.com/jk-image-similarity-app:$VERSION .

    - name: Push the image to the registry
      run: |
        VERSION=$(# Same version logic)
        docker push private-registry.example.com/jk-image-similarity-app:$VERSION
```

This automated process replaces **Part 2: The Developer's Handbook**.

#### 2. The Private Docker Registry (The Secure Warehouse)

Instead of using `docker save` to create a `.tar` file, the CI server uses `docker push` to upload the versioned image to a **Private Docker Registry**. This is a secure, centralized server for storing and distributing your Docker images.

*   **Popular Examples:** Amazon ECR, Google Artifact Registry, Azure Container Registry, Docker Hub (Private Repos), GitLab Container Registry.

This registry acts as the single source of truth for your production images. It replaces the `.tar` bundle from the air-gapped workflow.

#### 3. The CD Orchestrator (The Automated Deployer)

The final piece is the Continuous Deployment system. This is a tool running in your production environment that gets notified when a new image is pushed to the registry. It then automatically pulls the new image and updates the running application.

*   **For Simple Deployments:** You could have a script on your production server that periodically runs `docker-compose pull` (to fetch the newest images referenced in your `docker-compose.yaml`) and then `docker-compose up -d` to restart the services with the new images.
*   **For Advanced Deployments (Kubernetes):** In large-scale systems, **Kubernetes** is used. Kubernetes would manage the deployment, and you would trigger an update by changing the image tag in your Kubernetes deployment configuration file. Kubernetes would then perform a safe, zero-downtime **rolling update**, gradually replacing old containers with new ones.

**Summary of the Workflow Change:**

| Air-Gapped Workflow                  | Automated CI/CD Workflow                           |
| :----------------------------------- | :------------------------------------------------- |
| 1. **Manual Build:** `docker build` on dev machine. | 1. **Automated Build:** `docker build` on CI server.   |
| 2. **Manual Package:** `docker save` to a `.tar` file. | 2. **Automated Push:** `docker push` to a private registry. |
| 3. **Physical Transfer:** Move `.tar` file to secure zone. | 3. **Network Transfer:** Image is available over the network. |
| 4. **Manual Load:** `docker load` on production host. | 4. **Automated Pull:** CD orchestrator runs `docker pull`.   |
| 5. **Manual Deploy:** `docker-compose up` on production host. | 5. **Automated Deploy:** CD orchestrator updates running containers. |

By understanding these parallel concepts, you can see how the robust, versioned Docker images created by the process in this guide are the universal key to both highly secure manual deployments and highly efficient automated ones.

### A.12. 🖥️ Cross-Platform Considerations (Windows/macOS)

While this guide is primarily Linux-focused, the entire system is designed to run perfectly on Windows and macOS via **Docker Desktop**.

*   **How it works:** Docker Desktop runs a lightweight Linux virtual machine in the background to host the Docker daemon. This means your Linux containers are running in a genuine Linux environment, ensuring maximum compatibility.
*   **Key Differences:**
    *   **`host.docker.internal`:** This special DNS name (covered in A.4) works out-of-the-box on Docker Desktop, making it easy to connect containers to services running on your Windows/macOS host. On Linux, it requires extra configuration on some older Docker versions.
    *   **Volume Mounts:** Bind-mounting your source code (as done in the `dev` override) can have slower I/O performance on Docker Desktop compared to native Linux. This is because there is a virtualization layer to cross for every file read/write. For I/O-heavy tasks, using named volumes is often faster even in development.
    *   **SELinux:** The `:z` flag in the `docker-compose.rhel.yml` file is specific to SELinux on RHEL-based systems and is ignored (harmlessly) on other platforms like Debian, Ubuntu, Windows, and macOS.

### A.13. ⚖️ Clarifying Qdrant Modes: Server vs. Embedded

The choice of `QDRANT_MODE` in your `.env` file has significant implications for performance, scalability, and operational simplicity. Understanding these trade-offs is crucial for choosing the right architecture for your needs.

| Feature                 | Server Mode (`server`)                                                                                    | Embedded Mode (`embedded`)                                                                             |
| :---------------------- | :-------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------- |
| **Architecture**        | Runs as a separate, dedicated container (`qdrant` service). Your app communicates with it over the network. | Runs as a Python library inside your application container. No network communication needed.               |
| **Performance**         | **Higher.** Optimized for concurrent requests and heavy loads. Uses gRPC for efficient communication.      | **Lower.** Suitable for single-user or low-concurrency workloads. Less optimized for parallel operations. |
| **Scalability**         | **High.** Can be scaled independently of the application. Can be deployed on a separate, powerful machine.   | **Low.** Scales with the application. If you run 3 API containers, you get 3 separate, independent databases. |
| **Resource Usage**      | Higher initial memory footprint due to running a separate server process.                                   | Lower initial memory footprint, as it's just a library.                                                |
| **Simplicity**          | Requires an additional container to manage but centralizes the vector data, which is simpler at scale.      | Simpler initial setup with fewer moving parts (`--profile qdrant` is not needed).                       |
| **Data Persistence**    | Data is stored in a single, dedicated named volume (`qdrant_data`), making it easy to back up and manage.  | Data is stored within the application's `instance_data` volume, potentially mixed with other files.      |
| **Best For**            | **Production, multi-user applications, high-traffic APIs, and any scenario requiring high performance.**      | **Development, single-user desktop apps, lightweight demos, and simple scripting tasks.**               |