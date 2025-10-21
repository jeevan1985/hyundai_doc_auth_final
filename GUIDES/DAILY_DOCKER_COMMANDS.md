> Refactor Notice (2025-10): Data directory path normalized from `/appdat` to `/appdata`. Update any custom volume mounts, scripts, or environment variables accordingly. Only lowercase `appdat` forms were renamed; unrelated terms like `data` or `database` remain unchanged.
>
> Privacy naming note: configuration uses `search_task.privacy_mode` to control persistence of query crops. In code, this maps to `persist_query_crops = (not privacy_mode)`. Treat them as equivalent signals in documentation and operations.

# 👑 The Ultimate Operator's Guide: Image Similarity System

**Version:** 7.0.0
**Environments:** Mamba/Debian & RHEL

![Docker Banner](https://user-images.githubusercontent.com/10269477/189831613-346769f3-847e-4053-9025-4513a89e0c67.png)

Welcome! This is the definitive, all-in-one guide for the daily operation, management, and deployment of the Image Similarity System. It is meticulously designed for both developers who need to iterate on code and operators who manage the system in production, covering both Mamba/Debian and RHEL-based environments.

---

## 📑 Table of Contents
- [👑 The Ultimate Operator's Guide: Image Similarity System](#-the-ultimate-operators-guide-image-similarity-system)
  - [📑 Table of Contents](#-table-of-contents)
  - [✅ Core Concepts: The Foundation of Understanding](#-core-concepts-the-foundation-of-understanding)
    - [The Two Core Environments: Production vs. Development](#the-two-core-environments-production-vs-development)
    - [The File Ecosystem: Understanding Your `docker-compose` Files](#the-file-ecosystem-understanding-your-docker-compose-files)
    - [Service Profiles: Running Only What You Need](#service-profiles-running-only-what-you-need)
    - [Networking & `0.0.0.0`: How Services Talk to the World](#networking--0000-how-services-talk-to-the-world)
  - [🚀 Quick Start: First-Time Setup in Any Environment](#-quick-start-first-time-setup-in-any-environment)
    - [🖥️ For Production Users (Running Pre-Built Images)](#️-for-production-users-running-pre-built-images)
    - [🧪 For Developers (Live Code Reloading)](#-for-developers-live-code-reloading)
  - [1. 🏭 Building Application Images: A Deep Dive into Philosophies and Practice](#1--building-application-images-a-deep-dive-into-philosophies-and-practice)
    - [1.1. The Two Philosophies: `docker compose build` vs. `docker build`](#11-the-two-philosophies-docker-compose-build-vs-docker-build)
    - [1.2. Method A: The Integrated Approach (`docker compose build`)](#12-method-a-the-integrated-approach-docker-compose-build)
    - [1.3. Method B: The Direct Approach (`docker build`) - Recommended for Production Artifacts](#13-method-b-the-direct-approach-docker-build---recommended-for-production-artifacts)
    - [1.4. Publishing an Image to a Registry](#14-publishing-an-image-to-a-registry)
  - [2. ⚙️ GPU Acceleration: Concepts, Setup & Verification](#2-️-gpu-acceleration-concepts-setup--verification)
    - [2.1. Prerequisites: The Host Machine Setup](#21-prerequisites-the-host-machine-setup)
    - [2.2. How GPU Support is Implemented in this Project](#22-how-gpu-support-is-implemented-in-this-project)
    - [2.3. Verification: Confirming GPU Access](#23-verification-confirming-gpu-access)
  - [3. 🚢 Preparing for Air-Gapped Deployment: Offline Operations](#3--preparing-for-air-gapped-deployment-offline-operations)
  - [4. ▶️ Daily Operations: Running the Application Stack](#4-️-daily-operations-running-the-application-stack)
    - [▶️ 4.1. Starting Services (Production Mode)](#️-41-starting-services-production-mode)
      - [Scenario 1: Running on a CPU-Only System](#scenario-1-running-on-a-cpu-only-system)
      - [Scenario 2: Running on a GPU-Enabled System](#scenario-2-running-on-a-gpu-enabled-system)
    - [🧪 4.2. Starting Services (Development Mode)](#-42-starting-services-development-mode)
      - [Scenario 1: Developing on a CPU-Only System](#scenario-1-developing-on-a-cpu-only-system)
      - [Scenario 2: Developing on a GPU-Enabled System](#scenario-2-developing-on-a-gpu-enabled-system)
  - [5. 🔄 Updating a Running Application: Seamless Transitions](#5--updating-a-running-application-seamless-transitions)
    - [Scenario A: Updating from a New Image in a Registry](#scenario-a-updating-from-a-new-image-in-a-registry)
    - [Scenario B: Updating with a Locally Rebuilt Image](#scenario-b-updating-with-a-locally-rebuilt-image)
  - [6. ⚙️ Using the Command-Line Interface (CLI): Powerful Utilities](#6-️-using-the-command-line-interface-cli-powerful-utilities)
    - [⚡ 6.1. Building the Search Index](#-61-building-the-search-index)
    - [🔎 6.2. Performing a Search via CLI](#-62-performing-a-search-via-cli)
    - [👥 6.3. Managing API Users](#-63-managing-api-users)
  - [7. 🔍 Monitoring & Debugging: Keeping an Eye on Your System](#7--monitoring--debugging-keeping-an-eye-on-your-system)
    - [📋 7.1. Listing Running Services](#-71-listing-running-services)
    - [📜 7.2. Viewing Logs](#-72-viewing-logs)
    - [💻 7.3. Getting an Interactive Shell](#-73-getting-an-interactive-shell)
    - [📊 7.4. Monitoring Resource Usage](#-74-monitoring-resource-usage)
  - [8. 🧹 System Cleanup: Maintaining a Tidy Docker Environment](#8--system-cleanup-maintaining-a-tidy-docker-environment)
    - [🛑 Stopping Application Services](#-stopping-application-services)
    - [💥 Full System Prune ‼️](#-full-system-prune-️)
  - [9. ⚠️ CRITICAL OPERATIONS: Data Management & Disaster Recovery](#9-️-critical-operations-data-management--disaster-recovery)
    - [📥 9.1. Adding Images to the System (Seeding Data)](#-91-adding-images-to-the-system-seeding-data)
    - [🛡️ 9.2. Backing Up the PostgreSQL Database](#️-92-backing-up-the-postgresql-database)
    - [♻️ 9.3. Restoring the PostgreSQL Database](#️-93-restoring-the-postgresql-database)
    - [🗑️ 9.4. Stopping Services and Deleting Data ‼️](#️-94-stopping-services-and-deleting-data-️)
  - [10. 🤔 Troubleshooting Common Issues: Solutions to Unexpected Problems](#10--troubleshooting-common-issues-solutions-to-unexpected-problems)
  - [🎉 Conclusion](#-conclusion)

---

## ✅ Core Concepts: The Foundation of Understanding

Before diving into commands, let's establish a solid understanding of the underlying Docker and Docker Compose concepts that govern this system.

### The Two Core Environments: Production vs. Development
Understanding the distinction between these two environments is paramount, as it dictates which Dockerfiles and `docker-compose` configurations you will use.

-   🌳 **Production Environment:**
    -   **Purpose:** Designed for stability, performance, and reliability. This is where your live application will serve users.
    -   **Image Source:** Always uses **pre-built, optimized Docker images** (either from a public registry like Docker Hub or your private enterprise registry).
    -   **Code within Container:** The application code is baked *inside* the Docker image. It is static and immutable during runtime. This ensures consistency and security.
    -   **Volumes:** Typically uses **named Docker volumes** for persistent data (like databases, uploaded files, logs). These volumes are managed by Docker and keep data safe even if containers are destroyed.
    -   **Debugging:** Less verbose logging; remote debugging might be configured but is not the primary mode of development.

-   🔬 **Development Environment:**
    -   **Purpose:** Optimized for rapid iteration, debugging, and code changes. This is where developers actively write and test new features.
    -   **Image Source:** Images are built **locally from your source code** (`--build` flag).
    -   **Code within Container:** Your local source code directory is **bind-mounted** into the container. This means any changes you save on your host machine are instantly reflected inside the running container, allowing for live code reloading without rebuilding the image.
    -   **Volumes:** Primarily uses **bind mounts** for code, and often for data directories (like `instance/` or `logs/`) so developers can easily inspect and modify generated files directly on their host machine. This can sometimes lead to permission issues if not handled carefully (addressed by `HOST_UID`/`HOST_GID` in Dockerfiles).
    -   **Debugging:** Application servers often run in debug mode, providing more verbose logs and sometimes interactive debuggers.

### The File Ecosystem: Understanding Your `docker-compose` Files

Your project employs a highly modular `docker-compose` strategy. This allows for flexible configuration and avoids duplication across different scenarios. Getting familiar with each file's role is crucial.

```
.
├── docker-compose.conda.yaml         <-- Main CPU/Debian PRODUCTION base file
├── docker-compose.gpu.conda.yaml     <-- Standalone GPU PRODUCTION base file (contains ALL config + GPU)
├── docker-compose.conda.dev.yaml     <-- Development OVERRIDE file (merges with a base file)
├── Dockerfile.conda                  <-- Dockerfile for Debian production builds
├── Dockerfile.conda.dev              <-- Dockerfile for Debian development builds
└── hyundai_document_authenticator/
    └── Docker_for_airgapped/
        ├── Dockerfiles_RHEL/
        │   └── docker-compose.rhel.yaml    <-- RHEL backup compose file
        ├── Dockerfiles_Ubuntu/
        │   └── docker-compose.yaml         <-- Ubuntu backup compose file
        └── Dockerfiles_Ubuntu_mamba/
            └── docker-compose.mamba.yaml   <-- Mamba backup compose file
```

-   **Base Compose File (`docker-compose.conda.yaml`):**
    -   This is your primary blueprint for running the application in a **CPU-only production setting**.
    -   It defines all the services, networks, volumes, environment variables, and health checks for a complete application stack.
    -   Crucially, services in this file are configured to use a **pre-built `image:`** from a registry. They **do not** contain active `build:` instructions (unless you explicitly uncomment them for a one-off local build, which is generally discouraged for clarity).

-   **GPU-Enabled Base File (`docker-compose.gpu.conda.yaml`):**
    -   This is a **standalone** file. It means it is a complete `docker-compose` definition on its own, just like `docker-compose.conda.yaml`.
    -   Its purpose is to define the **production-ready stack WITH GPU support**. It achieves this by including the `deploy.resources.reservations.devices` configuration.
    -   When running a GPU-enabled production stack, you will only use this file (`-f docker-compose.gpu.conda.yaml`), not the `docker-compose.conda.yaml` file, as it already contains everything.

-   **Development Override File (`docker-compose.conda.dev.yaml`):**
    -   This file is specifically designed to be an **overlay** (an "override"). It's typically much smaller than the base files.
    -   It contains *only* the settings necessary to transform a production configuration into a development configuration. This includes:
        -   Switching from `image:` to `build:` for application services.
        -   Adding bind mounts for live-reloading of source code.
        -   Changing container `command:`s to run services in debug/reload mode.
    -   **It cannot run by itself.** It must be merged with one of the base files (`docker-compose.conda.yaml` or `docker-compose.gpu.conda.yaml`).

-   **Air-Gapped Backup Files:**
    -   Located in `hyundai_document_authenticator/Docker_for_airgapped/` subdirectories.
    -   These are backup compose files for specific deployment scenarios (RHEL, Ubuntu, Mamba) in air-gapped environments.
    -   Each contains complete service definitions tailored for their respective base images and environments.

> **💡 The Merging Principle in Action:**
> When you run `docker compose -f <base_file> -f <override_file> ...`, Docker Compose reads the first file, then applies the second. If a service (e.g., `flask_api`) is defined in both files, any key present in the second file will **override** the corresponding key from the first. Keys not present in the second file will simply be inherited from the first. This powerful mechanism allows for flexible, non-destructive configuration changes.

### Service Profiles: Running Only What You Need
To optimize resource usage and tailor your deployment, the application is divided into `profiles`. You activate these optional service groups using the `--profile <name>` flag when running `docker compose up`.

-   **`postgres`**: Activates the PostgreSQL database service. Essential for all data persistence.
-   **`qdrant`**: Activates the Qdrant vector database service. Used for storing and searching vector embeddings.
-   **`flask_api`**: Activates the Flask API backend service.
-   **`fastapi_api`**: Activates the FastAPI backend service.
-   **`gui`**: Activates the Streamlit GUI frontend service.
    -   **Dependency Awareness:** If you activate the `gui` profile, Docker Compose is smart enough to know that `gui` depends on `flask_api` (as defined in your compose file). It will **automatically activate** `flask_api` and any of its required dependencies (like `postgres` or `qdrant` if they are active profiles). You don't have to explicitly list all dependencies.
-   **`fastapi_gui`**: Activates the FastAPI GUI frontend service.
    -   **Dependency Awareness:** Similar to `gui`, activating `fastapi_gui` will automatically activate `fastapi_api`.
-   **`cli`**: Activates the CLI helper service. This is a "dummy" service primarily used for running one-off commands (like database migrations, index building, or administrative scripts) without starting a full API server. It is typically used with `docker compose run --rm`.

### Networking & `0.0.0.0`: How Services Talk to the World
Docker creates an isolated network for your Compose project, allowing containers to communicate with each other using their service names (e.g., `flask_api` can reach `db` by using `db` as the hostname). But how do you access these services from *outside* the Docker host?

-   **Inside the Container (Binding to `0.0.0.0`):**
    -   In your Dockerfiles, applications like Flask or Uvicorn are configured to run and listen on `0.0.0.0`. This is a special IP address that means "listen for incoming connections on **all available network interfaces**."
    -   **Why it's necessary:** If the application inside the container only listened on `127.0.0.1` (localhost), it would only be accessible from *within that container itself*. By listening on `0.0.0.0`, it signals that it's ready to accept connections from *any* IP address on its internal Docker network.

-   **Outside the Container (Port Mapping with `ports:`):**
    -   The `ports:` section in your `docker-compose.yaml` (e.g., `- "5001:5001"`) is what creates the bridge between the Docker network and your host machine's network.
    -   Security: exposing database ports on the host increases attack surface. Prefer `docker exec` or internal network access in development. See canonical guidance: MASTER_GUIDE.md → [Development database exposure (dev-only)](./MASTER_GUIDE.md#development-database-exposure-dev-only)
    -   **Syntax:** `HOST_PORT:CONTAINER_PORT`
        -   The `HOST_PORT` (e.g., `5001` on the left) is the port on your actual physical machine that you will type into your web browser or client application.
        -   The `CONTAINER_PORT` (e.g., `5001` on the right) is the port that the application inside the container is listening on (which is typically `0.0.0.0:<port>`).
    -   **Access:** Once mapped, you can access the service from your host machine (or from other machines on your local network, if your firewall allows) using:
        -   `http://localhost:<HOST_PORT>` (on the Docker host itself)
        -   `http://<Your_Host_Machine_IP_Address>:<HOST_PORT>` (from another machine)
    -   **Flexibility:** You can change the `HOST_PORT` to avoid conflicts (e.g., `"8080:5001"` would map the container's port 5001 to your host's port 8080).

---

## 🚀 Quick Start: First-Time Setup in Any Environment

This section provides streamlined, step-by-step instructions to get the system up and running quickly for the very first time, tailored for different user needs and hardware.

### 🖥️ For Production Users (Running Pre-Built Images)
This workflow focuses on getting your application deployed using pre-optimized, stable Docker images pulled from a registry.

1.  **Configure Environment:**
    -   Create your `.env` file from the example. This file holds sensitive credentials and configuration.
    ```bash
    cp .env.example .env
    # Optionally, open .env in a text editor to review/customize settings.
    ```

2.  **Pull Required Images:**
    -   Download the Docker images that form your application stack. This ensures you have all components locally before starting.
    ```bash
    # Pull your custom application image from your registry
    docker pull your-registry/jk-image-similarity-app:v1.0.0  # For Conda-based images

    # Pull public dependency images (PostgreSQL and Qdrant)
    docker pull postgres:15-alpine
    docker pull qdrant/qdrant:v1.9.2
    ```
    > 💡 **Why Pull?** Pulling images beforehand can speed up `docker compose up` as it avoids pulling during the startup phase. It's also essential for air-gapped scenarios (see Section 3).

3.  **Start the Full Application Stack (Choose your hardware setup):**
    -   This command brings all specified services online, creates the necessary Docker networks and volumes, and runs the containers in the background (`-d`).

    -   **On a CPU-only system:**
        ```bash
        # Files used: docker-compose.conda.yaml (for Debian/Conda)
        docker compose -f docker-compose.conda.yaml up -d --profile gui --profile fastapi_gui --profile postgres --profile qdrant
        # 🐧 For RHEL (using backup compose file):
        # docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.rhel.yaml up -d --profile gui --profile fastapi_gui --profile postgres --profile qdrant
        ```

    -   **On a GPU-enabled system:**
        ```bash
        # File used: docker-compose.gpu.conda.yaml (this file already includes GPU setup and all service definitions)
        docker compose -f docker-compose.gpu.conda.yaml up -d --profile gui --profile fastapi_gui --profile postgres --profile qdrant
        ```
> **Congratulations!** Your system is now running in production mode. Your next step is to **Section 9.1** to add your image data and build the search index.

### 🧪 For Developers (Live Code Reloading)
This workflow sets up a local development environment where your code changes are immediately reflected in the running containers.

1.  **Configure Environment:**
    ```bash
    cp .env.example .env
    ```

2.  **Build & Start the Full Development Stack (Choose your hardware setup):**
    -   This single command will:
        -   Build your application image(s) locally using the `Dockerfile.mamba.dev` (or `Dockerfile.rhel.dev`).
        -   Start all the specified services.
        -   Enable live-reloading of your code through bind mounts.
        -   Run containers in the background (`-d`).

    -   **On a CPU-only system:**
        ```bash
        # Files used: docker-compose.conda.yaml (base) + docker-compose.conda.dev.yaml (override)
        docker compose -f docker-compose.conda.yaml -f docker-compose.conda.dev.yaml up -d --build --profile gui --profile fastapi_gui --profile postgres --profile qdrant
        # 🐧 For RHEL (using backup files):
        # docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.rhel.yaml -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.dev.rhel.yaml up -d --build --profile gui --profile fastapi_gui --profile postgres --profile qdrant
        ```

    -   **On a GPU-enabled system:**
        ```bash
        # Files used: docker-compose.gpu.conda.yaml (base) + docker-compose.conda.dev.yaml (override)
        docker compose -f docker-compose.gpu.conda.yaml -f docker-compose.conda.dev.yaml up -d --build --profile gui --profile fastapi_gui --profile postgres --profile qdrant
        ```
> **Congratulations!** Your development environment is ready. Any changes to your local source code will now auto-reload in the running containers. Your next step is to **Section 9.1** to add your image data and build the search index.

---

## 1. 🏭 Building Application Images: A Deep Dive into Philosophies and Practice

Creating the Docker image is a fundamental step. There are two primary philosophies for how to approach this with Docker Compose, each with its own advantages and ideal use cases.

### 1.1. The Two Philosophies: `docker compose build` vs. `docker build`

#### The "Integrated" Approach: `docker compose build`
-   **Philosophy:** "My image is an integral part of my application stack defined within Docker Compose. I will use the `docker compose` tool to manage all aspects of the stack, including building the images."
-   **How it works:** When you include a `build:` key in your `docker-compose.yaml` service definition, `docker compose build` (or `docker compose up --build`) will follow those instructions to create the image. The image automatically gets tagged with the service name and often the project name.
-   **Pros:**
    -   **Cohesion:** Keeps image build instructions directly tied to the service definitions that use them.
    -   **Simpler Tagging:** Docker Compose often handles default tagging automatically based on service and project names, which can be convenient.
    -   **Integrated Workflow:** Fits seamlessly into `docker compose up --build` for single-command setup.
-   **Cons:**
    -   **Clarity/Ambiguity:** If your `docker-compose.yaml` is primarily designed to *run* pre-built `image:`s, using `docker compose build` on it directly will do nothing (as we experienced). It often requires separate override files (like `docker-compose.dev.yaml`) to activate the `build:` instruction, which can add complexity.
    -   **Explicit Tagging:** While Compose tags automatically, for production releases, you often want a very specific, manual tag (e.g., `v1.0.0`) which `docker compose build` doesn't provide as easily without extra `image:` key definitions or manual tagging afterwards.
-   **When to use:** Primarily for **development workflows** where `docker compose up --build` is common, or if your `docker-compose.yaml` files are *always* configured for building from source.

#### The "Direct" Approach: `docker build`
-   **Philosophy:** "My first task is to create a standalone Docker image artifact from a specific Dockerfile. I will use the direct, low-level `docker build` command for that. Afterwards, my second task is to run that artifact using `docker compose`."
-   **How it works:** You directly tell the Docker daemon to build an image from a specified `Dockerfile` and apply a specific tag. This command operates independently of your `docker-compose.yaml` files.
-   **Pros:**
    -   **Clear Separation of Concerns:** Building is distinct from running. Your `docker-compose.yaml` files can remain purely focused on orchestrating *running* containers from `image:` tags.
    -   **No File Edits:** You don't need to comment/uncomment `build:` or `image:` lines in your main compose files, nor do you need extra build-specific override files.
    -   **Explicit Tagging:** Gives you full control over the exact name and version tag of the resulting image. This is crucial for consistent releases.
    -   **Independent:** Can be run outside of a `docker-compose` context, useful for CI/CD pipelines.
-   **Cons:**
    -   **Manual Sync:** You must ensure that the image tag you provide in your `docker build` command precisely matches the `image:` tag used in your `docker-compose.yaml` files.
    -   **Less Integrated:** Not part of a single `docker compose up` command.
-   **When to use:** Highly recommended for creating **production-ready image artifacts**, preparing images for **air-gapped environments**, and for **CI/CD pipelines** where explicit control over image naming and layers is paramount.

### 1.2. Method A: The Integrated Approach (`docker compose build`)
This method is commonly used for developers leveraging the `up --build` command in their development cycle.

```bash
# This command is for building the development image specifically.
# It uses the combination of the base compose file and the development override file
# to activate the 'build:' instruction.

docker compose -f docker-compose.conda.yaml -f docker-compose.conda.dev.yaml build
# 🐧 For RHEL (using backup files):
# docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.rhel.yaml -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.dev.rhel.yaml build
```
> **What it does:** This command tells Docker Compose to merge the configuration, find all services that now have a `build:` instruction (due to the `docker-compose.dev.yaml` override), and build their corresponding images.

### 1.3. Method B: The Direct Approach (`docker build`) - Recommended for Production Artifacts
This is the preferred method for creating final, shippable production images.

```bash
# This command explicitly builds your production-ready Conda image.

docker build \
  -f Dockerfile.conda \
  -t your-registry/jk-image-similarity-app:v1.0.0 \
  .
```
**Breakdown of the `docker build` command:**
*   **`docker build`**: The core command that initiates the image creation process.
*   **`-f Dockerfile.conda`**: This crucial flag specifies the exact path to the Dockerfile that Docker should use for the build instructions. If omitted, Docker looks for `Dockerfile` in the current context.
*   **`-t your-registry/jk-image-similarity-app:v1.0.0`**: The `-t` (or `--tag`) flag is used to name and tag the resulting image.
    *   `your-registry/jk-image-similarity-app`: This is the image name, typically prefixed with your Docker Hub username or private registry URL.
    *   `:v1.0.0`: This is the tag, indicating the version. This tag should **exactly match** the `image:` value in your `docker-compose.yaml` files for production.
*   **`.` (The dot at the end)**: This specifies the **build context**. It tells the Docker daemon to send **all files and subdirectories** from the *current directory* where you are running the command to the Docker daemon. This entire package of files is then available for `COPY` commands within your `Dockerfile`. **Without this dot, your `COPY` commands inside the Dockerfile would fail.**

### 1.4. Publishing an Image to a Registry
Once your image is built and tagged locally, you can push it to a Docker registry (like Docker Hub or a private registry) to share it or deploy it to other machines.

```bash
# 1. Login to your registry (if needed). You'll be prompted for username and password.
docker login your-registry.com

# 2. Push the tagged image. This uploads the image layers to the remote registry.
docker push your-registry/jk-image-similarity-app:v1.0.0
```

---

## 2. ⚙️ GPU Acceleration: Concepts, Setup & Verification

Leveraging GPU acceleration can provide significant performance benefits for computationally intensive tasks like image similarity calculations. This section explains how GPU support is configured and verified in your Docker setup.

### 2.1. Prerequisites: The Host Machine Setup
Before Docker containers can utilize your GPU, your host machine must be correctly configured.
1.  **Compatible NVIDIA GPU:** Ensure your system has an NVIDIA graphics card that supports CUDA.
2.  **NVIDIA Drivers:** The latest proprietary NVIDIA drivers for your Linux distribution must be installed and functioning correctly. You should be able to open a terminal on your host and run `nvidia-smi` to see your GPU details and utilization. If `nvidia-smi` doesn't work, your drivers are likely not installed or configured properly.
3.  **NVIDIA Container Toolkit:** This is a crucial software package from NVIDIA that integrates with Docker. It allows the Docker runtime to recognize and grant containers access to the host's NVIDIA GPUs and CUDA libraries.
    -   **Installation:** Follow the official installation guide for your specific operating system: [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). This typically involves adding NVIDIA's repository, installing the `nvidia-container-toolkit` package, and restarting the Docker daemon.

### 2.2. How GPU Support is Implemented in this Project
Your project utilizes a specific `docker-compose` file to enable GPU support: `docker-compose.gpu.conda.yaml`. This file explicitly tells Docker to provide GPU resources to the relevant services.

The key to enabling GPU access is the `deploy` configuration within a service definition in your Compose file:

```yaml
x-app-base: &app-base
  # ... (other existing configurations like image, restart, volumes, environment) ...

  # ============================================================================
  # ⭐ GPU ACCELERATION CONFIGURATION ⭐
  # This block tells Docker Compose to grant this service access to the host's NVIDIA GPU(s).
  # This configuration is present in your 'docker-compose.gpu.conda.yaml' file.
  # ============================================================================
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all              # 'all' means pass all available GPUs. Can be '1' for a single GPU, or a specific GPU UUID.
            capabilities: [gpu]     # Requesting general GPU capabilities (e.g., CUDA, NVML)
  # ============================================================================
```
**Explanation of the `deploy` block:**
*   **`deploy`**: A top-level service key in Docker Compose, primarily used for deployment and runtime configurations.
*   **`resources`**: Under `deploy`, this sub-key allows you to specify resource constraints or reservations for the service.
*   **`reservations`**: Ensures that these resources are reserved for the container. If the resources are not available on the host (e.g., no GPU, or NVIDIA Container Toolkit not installed), the service will fail to start.
*   **`devices`**: A list of devices to expose to the container.
    *   **`- driver: nvidia`**: This critical line instructs Docker to use the NVIDIA Container Runtime (provided by the NVIDIA Container Toolkit) when starting this container. This is what enables the magic of GPU passthrough.
    *   **`count: all`**: Specifies that all available NVIDIA GPUs on the host machine should be accessible to the container. You could also specify `count: 1` to expose only one GPU, or use specific GPU UUIDs for fine-grained control.
    *   **`capabilities: [gpu]`**: This tells the NVIDIA Container Runtime to expose the necessary NVIDIA-specific capabilities to the container, such as access to CUDA libraries and `nvidia-smi`.

### 2.3. Verification: Confirming GPU Access

After you have started your application using a GPU-enabled `docker compose up` command (refer to Section 4.1 or 4.2 for the correct commands), you can verify that the container successfully sees and can utilize the GPU.

1.  **Get an interactive shell inside a running application container** that is configured for GPU use (e.g., `flask_api` or `fastapi_api`):
    ```bash
    # For a production GPU setup:
    docker compose -f docker-compose.gpu.conda.yaml exec flask_api bash

    # For a development GPU setup:
    docker compose -f docker-compose.gpu.conda.yaml -f docker-compose.conda.dev.yaml exec flask_api bash
    ```

2.  **Once inside the container, run the NVIDIA System Management Interface (`nvidia-smi`):**
    ```bash
    appuser@container_id:/home/appuser/app$ nvidia-smi
    ```
    -   **Expected Output:** If the GPU passthrough is successful, you will see a table displaying your NVIDIA GPU's details (driver version, CUDA version, temperature, memory usage, and a list of processes currently running on the GPU, including potentially your application's Python process).
    -   **Troubleshooting (If `nvidia-smi` fails):** If you receive a "command not found" error, or `nvidia-smi` runs but shows no GPUs or errors, it indicates that the GPU was not correctly exposed to the container. Revisit the "Prerequisites" in Section 2.1 to ensure the NVIDIA Container Toolkit is fully installed and your Docker daemon is configured to use it.

---

## 3. 🚢 Preparing for Air-Gapped Deployment: Offline Operations

For environments with restricted or no internet access (air-gapped systems), you cannot pull Docker images directly. You must pre-package all necessary images on a connected system and transfer them.

1.  **Build Your Production Application Image:**
    -   On your internet-connected machine, build your custom application image using the direct `docker build` method. This ensures the image exists locally.
    ```bash
    docker build -f Dockerfile.conda -t your-registry/jk-image-similarity-app:v1.0.0 .
    # 🐧 For RHEL (using backup Dockerfile):
    # docker build -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/Dockerfile.rhel -t your-registry/jk-image-similarity-app:rhel-v1.0.0 .
    ```

2.  **Pull All Public Dependency Images:**
    -   Even if you've run them before, explicitly pull the exact versions of public images (like PostgreSQL and Qdrant) that are specified in your `docker-compose.yaml` files.
    ```bash
    docker pull postgres:15-alpine
    docker pull qdrant/qdrant:v1.9.2
    ```

3.  **Save All Images into a Single Archive:**
    -   The `docker save` command bundles one or more specified images into a single `.tar` archive. This is the portable file you will transfer.
    ```bash
    docker save -o image-similarity-stack.tar \
      your-registry/jk-image-similarity-app:v1.0.0 \
      postgres:15-alpine \
      qdrant/qdrant:v1.9.2
    ```
    > 💡 **Pro Tip:** Replace `your-registry/jk-image-similarity-app:v1.0.0` with the exact tag of your application image if it's different. Ensure you list *all* images required by your `docker-compose.yaml` (including the GPU version if that's what you plan to deploy).

4.  **Transfer the Files:**
    -   Copy two things to your air-gapped system:
        1.  The image archive file: `image-similarity-stack.tar`
        2.  The relevant `docker-compose.yaml` file(s) for your deployment (`docker-compose.conda.yaml` or `docker-compose.gpu.conda.yaml`).

5.  **Load Images on the Air-Gapped System:**
    -   On the air-gapped machine, use the `docker load` command to import all the images from your archive into the local Docker daemon.
    ```bash
    docker load -i image-similarity-stack.tar
    ```
    -   **Verification:** After loading, you can run `docker images` to confirm that all necessary images are now available locally.
    > You can now proceed to **Section 4.1** and run the application using `docker compose up`. Since the images already exist locally, Docker will not attempt to pull them.

---

## 4. ▶️ Daily Operations: Running the Application Stack

This section provides the core commands for starting and stopping your application stack in various configurations (production/development, CPU/GPU).

### ▶️ 4.1. Starting Services (Production Mode)

#### Scenario 1: Running on a CPU-Only System
> **💡 File(s) Used:** `docker-compose.conda.yaml` (for Debian/Conda)
>
> This scenario is for deploying on servers or machines that **do not have a GPU** or where you prefer to run on CPU only.

```bash
# This command starts the GUI, FastAPI GUI, and their backend APIs,
# along with the internal PostgreSQL and Qdrant databases.
# All services will run using their respective images defined in the compose file.

docker compose -f docker-compose.conda.yaml up -d --profile gui --profile fastapi_gui --profile postgres --profile qdrant
# 🐧 For RHEL-based systems (using backup compose file):
# docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.rhel.yaml up -d --profile gui --profile fastapi_gui --profile postgres --profile qdrant
```
**Explanation:**
*   `-f docker-compose.conda.yaml`: Specifies the primary configuration file for a CPU-only Conda stack.
*   `up`: Creates and starts containers, networks, and volumes as defined in the compose file.
*   `-d`: Detached mode. Runs containers in the background and returns control to your terminal.
*   `--profile gui --profile fastapi_gui --profile postgres --profile qdrant`: Activates all the specified service profiles, ensuring the full application stack is brought online.

#### Scenario 2: Running on a GPU-Enabled System
> **💡 File(s) Used:** `docker-compose.gpu.conda.yaml`
>
> This scenario is for deploying on machines that **have a properly configured NVIDIA GPU** and the NVIDIA Container Toolkit installed (as per Section 2.1).

```bash
# This command starts the full application stack with GPU acceleration enabled
# for the application services.

docker compose -f docker-compose.gpu.conda.yaml up -d --profile gui --profile fastapi_gui --profile postgres --profile qdrant
```
**Explanation:**
*   `-f docker-compose.gpu.conda.yaml`: Directly uses the GPU-enabled compose file. This file contains *all* the necessary service definitions, including the `deploy` blocks for GPU access. You do not need to combine it with `docker-compose.mamba.yaml` for this purpose.

### 🧪 4.2. Starting Services (Development Mode)

#### Scenario 1: Developing on a CPU-Only System
> **💡 File(s) Used:** `docker-compose.conda.yaml` (base) + `docker-compose.conda.dev.yaml` (override)
>
> This is for developers working on machines without GPUs or when GPU acceleration is not required for the development task.

```bash
# This command merges the base CPU configuration with the development overrides.
# It builds the application image(s) locally (--build) and starts them with
# live-reloading enabled.

docker compose -f docker-compose.conda.yaml -f docker-compose.conda.dev.yaml up -d --build --profile gui --profile fastapi_gui --profile postgres --profile qdrant
# 🐧 For RHEL-based systems (using backup files):
# docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.rhel.yaml -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.dev.rhel.yaml up -d --build --profile gui --profile fastapi_gui --profile postgres --profile qdrant
```
**Explanation:**
*   `-f docker-compose.conda.yaml -f docker-compose.conda.dev.yaml`: This is the crucial part for merging. The development overrides (`docker-compose.conda.dev.yaml`) are applied on top of the CPU base configuration (`docker-compose.conda.yaml`). This activates local builds, bind mounts, and development commands.
*   `--build`: Forces Docker Compose to build the application image(s) from your `Dockerfile.conda.dev` (as specified in `docker-compose.conda.dev.yaml`). This is essential for reflecting your latest code changes.

#### Scenario 2: Developing on a GPU-Enabled System
> **💡 File(s) Used:** `docker-compose.gpu.conda.yaml` (base) + `docker-compose.dev.yaml` (override)
>
> This is for developers who want to leverage GPU acceleration during their development cycle (e.g., for faster indexing or testing GPU-specific features).

```bash
# This command merges the GPU-enabled base configuration with the development overrides.
# It provides a live-reloading environment with full GPU access.

docker compose -f docker-compose.gpu.conda.yaml -f docker-compose.conda.dev.yaml up -d --build --profile gui --profile fastapi_gui --profile postgres --profile qdrant
```
**Explanation:**
*   `-f docker-compose.gpu.conda.yaml -f docker-compose.conda.dev.yaml`: Here, the `docker-compose.gpu.conda.yaml` provides the initial stack definition *with GPU capabilities*. The `docker-compose.conda.dev.yaml` then layers on top, overriding the image source to a local build and enabling live-reloading, while *preserving* the `deploy` key for GPU access.

---

## 5. 🔄 Updating a Running Application: Seamless Transitions

Updating your application to a new version, whether from a registry or a local rebuild, is a common operation. Docker Compose makes this process smooth, often without requiring full downtime for the entire stack.

### Scenario A: Updating from a New Image in a Registry
Use this when a new version of your application (e.g., `your-registry/jk-image-similarity-app:v1.0.1`) has been pushed to your Docker registry.

```bash
# Step 1: Pull the latest version of the application image.
# This downloads the new image to your local Docker cache.
docker pull your-registry/jk-image-similarity-app:v1.0.1

# Step 2: Update the 'image:' tag in your docker-compose.conda.yaml file.
# Change 'jk-image-similarity-app:v1.0.0' to 'jk-image-similarity-app:v1.0.1'
# (or whatever your new tag is) for all relevant application services (flask_api, fastapi_api, gui, fastapi_gui, cli_runner).

# Step 3: Recreate the services.
# Docker Compose is intelligent. It will detect that the image has changed for
# the application services and will gracefully stop and restart *only* those
# containers. Database containers (like postgres and qdrant) will remain running
# unless their image or configuration also changed.

# (Use the appropriate compose file for your hardware setup, as used in Section 4.1)
docker compose -f docker-compose.conda.yaml up -d --profile gui --profile fastapi_gui --profile postgres --profile qdrant
```

### Scenario B: Updating with a Locally Rebuilt Image
Use this after you've made code changes, built a new local image, and want to deploy it to your running production stack.

```bash
# Step 1: Rebuild your production image.
# This creates a fresh, updated image on your local Docker daemon.
# (Refer to Section 1.3 for the detailed 'docker build' command)
docker build -f Dockerfile.conda -t your-registry/jk-image-similarity-app:v1.0.1 .

# Step 2: Ensure the 'image:' tag in your docker-compose.conda.yaml file
# matches the new tag you used in the 'docker build' command.
# (e.g., 'jk-image-similarity-app:v1.0.1')

# Step 3: Recreate the services with the newly built image.
# Similar to Scenario A, Compose will gracefully stop and restart only the
# application containers that use the updated image.

# (Use the appropriate compose file for your hardware setup, as used in Section 4.1)
docker compose -f docker-compose.conda.yaml up -d --profile gui --profile fastapi_gui --profile postgres --profile qdrant
```

---

## 6. ⚙️ Using the Command-Line Interface (CLI): Powerful Utilities

The `cli_runner` service is designed as a lightweight, disposable container to execute one-off tasks. It joins the app network, loads .env, and shares volumes, making it ideal for manual runs of the core pipeline.

Key concept: Use the entrypoint "cli" mode to run doc_image_verifier.py subcommands. The container is removed automatically with `--rm`.

### ⚡ 6.1. Building the Image Index from TIFs
> Run this after seeding new TIFs to (re)build the image index from extracted photos.

- Production (CPU/root Conda):
```bash
docker compose -f docker-compose.conda.yaml run --rm cli_runner \
  cli hyundai_document_authenticator/doc_image_verifier.py build-image-index \
  --folder ./hyundai_document_authenticator/data_real
```
- Production (GPU/root Conda):
```bash
docker compose -f docker-compose.gpu.conda.yaml run --rm cli_runner \
  cli hyundai_document_authenticator/doc_image_verifier.py build-image-index \
  --folder ./hyundai_document_authenticator/data_real
```
- Development (root Conda):
```bash
docker compose -f docker-compose.conda.yaml -f docker-compose.conda.dev.yaml run --rm cli_runner \
  cli hyundai_document_authenticator/doc_image_verifier.py build-image-index \
  --folder ./hyundai_document_authenticator/data_real
```
- Ubuntu backup:
```bash
docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_Ubuntu/docker-compose.yaml run --rm cli_runner \
  cli hyundai_document_authenticator/doc_image_verifier.py build-image-index \
  --folder ./hyundai_document_authenticator/data_real
```
- RHEL backup:
```bash
docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.rhel.yaml run --rm cli_runner \
  cli hyundai_document_authenticator/doc_image_verifier.py build-image-index \
  --folder ./hyundai_document_authenticator/data_real
```
- Mamba backup:
```bash
docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_Ubuntu_mamba/docker-compose.mamba.yaml run --rm cli_runner \
  cli hyundai_document_authenticator/doc_image_verifier.py build-image-index \
  --folder ./hyundai_document_authenticator/data_real
```

### 🔎 6.2. Performing a Batch Search from TIFs
> Performs TIF batch search (photo extraction + image similarity + aggregation).

- Production (CPU/root Conda):
```bash
docker compose -f docker-compose.conda.yaml run --rm cli_runner \
  cli hyundai_document_authenticator/doc_image_verifier.py search-doc \
  --folder ./hyundai_document_authenticator/data_real \
  --top-doc 5 --top-k 5 --aggregation-strategy max
```
- Production (GPU/root Conda):
```bash
docker compose -f docker-compose.gpu.conda.yaml run --rm cli_runner \
  cli hyundai_document_authenticator/doc_image_verifier.py search-doc \
  --folder ./hyundai_document_authenticator/data_real \
  --top-doc 5 --top-k 5 --aggregation-strategy max
```
- Development (root Conda):
```bash
docker compose -f docker-compose.conda.yaml -f docker-compose.conda.dev.yaml run --rm cli_runner \
  cli hyundai_document_authenticator/doc_image_verifier.py search-doc \
  --folder ./hyundai_document_authenticator/data_real \
  --top-doc 5 --top-k 5 --aggregation-strategy max
```
- Ubuntu backup:
```bash
docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_Ubuntu/docker-compose.yaml run --rm cli_runner \
  cli hyundai_document_authenticator/doc_image_verifier.py search-doc \
  --folder ./hyundai_document_authenticator/data_real \
  --top-doc 5 --top-k 5 --aggregation-strategy max
```
- RHEL backup:
```bash
docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_RHEL/docker-compose.rhel.yaml run --rm cli_runner \
  cli hyundai_document_authenticator/doc_image_verifier.py search-doc \
  --folder ./hyundai_document_authenticator/data_real \
  --top-doc 5 --top-k 5 --aggregation-strategy max
```
- Mamba backup:
```bash
docker compose -f hyundai_document_authenticator/Docker_for_airgapped/Dockerfiles_Ubuntu_mamba/docker-compose.mamba.yaml run --rm cli_runner \
  cli hyundai_document_authenticator/doc_image_verifier.py search-doc \
  --folder ./hyundai_document_authenticator/data_real \
  --top-doc 5 --top-k 5 --aggregation-strategy max
```

### 👥 6.3. API Services
The system provides API functionality through the mock API server for TIFF document retrieval.

**Available API Endpoint:**
- **`/images`** (GET/POST): Retrieves TIFF documents based on filename parameter
  - Accepts parameters: `filename` (required), `save_to_folder` (optional)
  - Returns JSON with base64-encoded TIFF data and metadata

**Starting the Mock API Server:**
```bash
# The mock API server can be started independently if needed
# (Note: This is typically handled by the main application stack)
docker compose -f docker-compose.conda.yaml run --rm cli_runner \
  python hyundai_document_authenticator/external/key_input/mock_api_server.py \
  --key-table "path/to/key_table.xlsx" \
  --search-root "path/to/tiff/files" \
  --tail-len 5 --glob-suffix "_*.tif" --any-depth
```

**Notes:**
- The API focuses on document retrieval functionality
- Ensure your .env contains correct DB/Qdrant settings for the chosen stack and paths exist inside the container

## 7. 🔍 Monitoring & Debugging: Keeping an Eye on Your System

---

## 7. 🔍 Monitoring & Debugging: Keeping an Eye on Your System

Effective monitoring and debugging are essential for maintaining the health and performance of your Dockerized application.

### 📋 7.1. Listing Running Services
Quickly see which containers are currently running for your project.
```bash
# This lists containers defined in the specified compose file(s) that are currently running.
# Use the appropriate compose file(s) for your running environment (e.g., production CPU, production GPU, development CPU, development GPU)
docker compose -f docker-compose.conda.yaml ps

# To see ALL containers (running and stopped) on your Docker daemon, regardless of project:
docker ps -a
```

### 📜 7.2. Viewing Logs
Logs are your first line of defense when troubleshooting.
```bash
# View logs from all running services in your project (streamed live, press Ctrl+C to exit)
docker compose -f docker-compose.conda.yaml logs -f

# View logs for a specific service (e.g., flask_api)
# The '-f' (follow) flag streams new logs as they appear.
docker compose -f docker-compose.conda.yaml logs -f flask_api

# View a specific number of recent log lines (e.g., last 100 lines for flask_api)
docker compose -f docker-compose.conda.yaml logs --tail 100 flask_api
```

### 💻 7.3. Getting an Interactive Shell
This is invaluable for exploring a container's filesystem, inspecting configuration files, or running commands directly within the container's environment.

-   **Into a running Production Service:**
    ```bash
    # This command executes a new 'bash' process inside the already running 'flask_api' container.
    # The '-it' flags provide an interactive terminal.
    # (Use the appropriate compose file for your running environment)
    docker compose -f docker-compose.conda.yaml exec flask_api bash
    ```
-   **Into a Development Container (New Temporary Container):**
    ```bash
    # This command starts a *new*, temporary 'cli_runner' container.
    # It mounts your local source code and drops you into a bash shell.
    # This is often preferred in development to avoid interfering with a running debug server.
    # The '--rm' flag ensures the temporary container is cleaned up when you exit.
    # (Use the appropriate combination of compose files for your hardware)
    docker compose -f docker-compose.conda.yaml -f docker-compose.conda.dev.yaml run --rm cli_runner bash
    ```

### 📊 7.4. Monitoring Resource Usage
Get real-time insights into your containers' resource consumption (CPU, memory, network I/O, disk I/O).
```bash
docker stats
# To stop continuous streaming, add --no-stream:
# docker stats --no-stream
```

---

## 8. 🧹 System Cleanup: Maintaining a Tidy Docker Environment

Regular cleanup prevents disk space issues and avoids conflicts with old containers or images.

### 🛑 Stopping Application Services
This command gracefully stops the containers defined in your specified compose file(s) but **leaves all associated data volumes intact.** This means your database data, uploaded files, and logs will persist.

-   **Production Environment:**
    ```bash
    # Use the appropriate compose file for your running environment
    docker compose -f docker-compose.conda.yaml down
    ```
-   **Development Environment:**
    ```bash
    # Use the appropriate combination of compose files
    docker compose -f docker-compose.conda.yaml -f docker-compose.conda.dev.yaml down
    ```

### 💥 Full System Prune ‼️
> ⚠️ **CAUTION: This is a powerful and destructive command.** It affects your **entire Docker daemon**, not just this project. It removes ALL:
> -   Stopped containers
> -   Unused networks
> -   Dangling (untagged) images
> -   Build cache
> -   **All unused volumes** (if `--volumes` is included, which is often desirable but highly destructive for non-project volumes).

Use this command with extreme care and only when you are certain you want to clear out all unused Docker resources from your system.

```bash
docker system prune -a --volumes
```
*   `-a`: Remove all unused images, not just dangling ones.
*   `--volumes`: Also remove all unused volumes.

---

## 9. ⚠️ CRITICAL OPERATIONS: Data Management & Disaster Recovery

These commands deal with your application's persistent data. Use them with caution and ensure you understand their implications.

### 📥 9.1. Adding Images to the System (Seeding Data)
This method safely copies your local image library into the persistent Docker volume used by the application, making the images available for processing and indexing.

```bash
# Step 1: Ensure the application's core services are running (e.g., postgres, qdrant)
#         You can use the appropriate 'docker compose up -d' command from Section 4.1 or 4.2.

# Step 2: Copy your local image folder into the 'instance_data' volume.
#         The 'cli_runner' service (even if not explicitly running) provides the context
#         to access the shared volume.
#         Replace `/path/to/your/local/images` with the actual path to your images on your host.

# (Use the appropriate compose file for your running environment)
docker compose -f docker-compose.conda.yaml cp /path/to/your/local/images/. cli_runner:/home/appuser/app/instance/database_images/
```
**Explanation:**
*   `docker compose cp`: This command is similar to `docker cp` but allows you to specify a service name from your compose file, which simplifies path resolution.
*   `cli_runner`: The service name from your `docker-compose.yaml` that shares the `instance_data` volume.
*   `:/home/appuser/app/instance/database_images/`: The destination path *inside the container*. This path corresponds to the mount point of your `instance_data` volume.

### 🛡️ 9.2. Backing Up the PostgreSQL Database (Best Practice)
Regular database backups are crucial for disaster recovery. This command creates a compressed backup of your entire PostgreSQL database and saves it to your current host directory.

```bash
# 1. Dynamically find the running PostgreSQL container's name.
#    This makes the command robust even if the container name changes slightly.
CONTAINER_NAME=$(docker ps --filter "name=postgres_db" --format "{{.Names}}")

# 2. Define the backup filename with a timestamp for easy organization.
BACKUP_FILE="db_backup_$(date +%Y%m%d_%H%M%S).gz"

# 3. Execute 'pg_dumpall' inside the container, pipe its output to gzip, and save to a local file.
#    'pg_dumpall -c -U postgres' creates a SQL dump of all databases, including roles and tablespaces.
#    'gzip' compresses the output.
docker exec -t ${CONTAINER_NAME} pg_dumpall -c -U postgres | gzip > ${BACKUP_FILE}

echo "✅ Backup created at ${BACKUP_FILE}"
```

### ♻️ 9.3. Restoring the PostgreSQL Database
This command restores a previously created database backup from your local machine into the running PostgreSQL container.

> ⚠️ **CAUTION:** Restoring a database will **overwrite** the current data in the database. Ensure you have a fresh backup of the current state before proceeding.

```bash
# 1. Dynamically find the running PostgreSQL container's name.
CONTAINER_NAME=$(docker ps --filter "name=postgres_db" --format "{{.Names}}")

# 2. Specify the actual backup file name to restore from.
#    ⭐⭐ IMPORTANT: REPLACE 'db_backup_YYYYMMDD_HHMMSS.gz' with your exact backup filename ⭐⭐
BACKUP_FILE="db_backup_YYYYMMDD_HHMMSS.gz"

# 3. Decompress the backup file and pipe it into the 'psql' command inside the container.
#    'gunzip -c ${BACKUP_FILE}' decompresses the file and outputs to standard output.
#    'docker exec -i ${CONTAINER_NAME} psql -U postgres' takes the piped input and
#    executes it as SQL commands within the PostgreSQL container.
gunzip -c ${BACKUP_FILE} | docker exec -i ${CONTAINER_NAME} psql -U postgres

echo "✅ Restore from ${BACKUP_FILE} complete."
```

### 🗑️ 9.4. Stopping Services and Deleting Data ‼️
> ⚠️ **DANGER: This command is irreversible.** It not only stops the containers but also **permanently deletes all associated data volumes** (e.g., `postgres_data`, `qdrant_data`, `downloads_data`, `instance_data`, `logs_data`). All your application data will be lost. Use with extreme caution.

```bash
# Use the appropriate compose file(s) for your running environment
# The '-v' (or '--volumes') flag is the key here; it explicitly removes volumes.
docker compose -f docker-compose.conda.yaml down -v
# Example for development environment:
# docker compose -f docker-compose.conda.yaml -f docker-compose.conda.dev.yaml down -v
```

---

## 10. 🤔 Troubleshooting Common Issues: Solutions to Unexpected Problems

### Issue: `permission denied` when running `docker` commands.
-   **Cause:** Your current user account does not have the necessary permissions to interact with the Docker daemon. This is a common security measure on Linux.
-   **Solution:**
    -   **Temporary:** Prefix all your `docker` or `docker compose` commands with `sudo` (as shown in some examples in this guide).
    -   **Permanent:** Add your user to the `docker` group. This is the recommended long-term solution.
        ```bash
        sudo usermod -aG docker $USER
        # After running this, you must log out of your session and log back in (or restart your machine)
        # for the group changes to take effect.
        ```

### Issue: Container fails to start, or exits immediately with an error code (e.g., `Exit Code 1`, `Exit Code 137`).
-   **Cause:** This usually indicates a configuration error, a missing dependency, or an application-level crash within the container. `Exit Code 137` specifically often points to an Out-of-Memory (OOM) error where the kernel killed the container process.
-   **Solution:**
    1.  **Check Logs:** The most important step. Logs will provide specific error messages from your application or from the container's startup process.
        ```bash
        # Replace <service_name> with the name of the failing service (e.g., flask_api, db)
        docker compose -f docker-compose.conda.yaml logs <service_name>
        ```
    2.  **Increase Resources:** If you suspect memory issues (Exit Code 137), increase the memory allocated to your Docker Desktop (if on Windows/macOS) or to the Docker daemon/system if on Linux (e.g., through systemd cgroup limits if not using Docker Desktop).
    3.  **Inspect Container:** For more details on the container's state and configuration:
        ```bash
        docker inspect <container_id_or_name>
        ```

### Issue: Error like `Bind for 0.0.0.0:5001 failed: port is already allocated`.
-   **Cause:** Another program or container on your host machine is already using one of the ports that your Docker Compose services are trying to map (e.g., port 5001).
-   **Solution:**
    1.  **Identify the culprit:**
        -   **Linux:** `sudo netstat -tulpn | grep 5001`
        -   **macOS:** `sudo lsof -i :5001`
        -   **Windows:** `netstat -ano | findstr :5001` (then use `tasklist /fi "PID eq <PID_found>"`)
    2.  **Stop the conflicting process** or container.
    3.  **Change the port mapping in your `docker-compose.yaml` file.** For example, if port `5001` is taken, change `- "5001:5001"` to `- "5002:5001"` in the `ports` section of the conflicting service. This maps the container's internal port `5001` to your host's port `5002`.
       -  For dev DB services, reconsider exposing to host; see MASTER_GUIDE.md → [Development database exposure (dev-only)](./MASTER_GUIDE.md#development-database-exposure-dev-only).

### Issue: `docker compose` command not found.
-   **Cause:** This usually means you have an older version of Docker Compose (v1.x) installed, which uses a hyphenated command, or the Docker Compose V2 plugin is not correctly installed.
-   **Solution:**
    -   **Try the legacy command:** Use `docker-compose` (with a hyphen) instead of `docker compose`.
    -   **Upgrade Docker Compose:** It's highly recommended to upgrade to Docker Compose V2, which integrates directly into the `docker` CLI (`docker compose`). Follow the official Docker documentation for installation/upgrade instructions for your OS.

### Issue: GPU not detected inside the container.
-   **Cause:** The NVIDIA Container Toolkit is not correctly installed/configured on the host, or the `deploy` key for GPU resources is missing/incorrect in your `docker-compose.yaml` file, or the Docker daemon was not restarted after installing the toolkit.
-   **Solution:**
    1.  **Verify Host Setup:** Run `nvidia-smi` on your host. If it fails, fix your NVIDIA driver/toolkit installation (refer to Section 2.1).
    2.  **Verify Compose File:** Double-check that you are using `docker-compose.gpu.conda.yaml` (or have added the `deploy` block to your `x-app-base` in `docker-compose.conda.yaml`) and that the `deploy` block is correctly formatted (refer to Section 2.2).
    3.  **Restart Docker Daemon:** After installing the NVIDIA Container Toolkit, you must restart the Docker daemon for the changes to take effect.
        ```bash
        sudo systemctl restart docker
        # Or if using Docker Desktop, restart it from the GUI.
        ```
    4.  **Rerun Verification:** Attempt the `nvidia-smi` verification inside the container again (Section 2.3).

---

## 🎉 Conclusion

This guide has provided a comprehensive and in-depth understanding of how to build, run, manage, and troubleshoot the Image Similarity System using Docker and Docker Compose. By mastering these commands and concepts, you are now equipped to confidently operate the system in both development and production environments, leveraging powerful features like GPU acceleration and maintaining data integrity.
