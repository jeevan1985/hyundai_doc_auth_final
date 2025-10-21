# ==============================================================================
#                 Production Dockerfile (Debian-based)
# ==============================================================================
# This Dockerfile builds a secure, efficient, and production-ready image for the
# application using a multi-stage build process on a Debian base.


# ==============================================================================
# --- Stage 1: The "Builder" Stage                                           ---
# ==============================================================================
# Purpose: This stage installs all dependencies, including build-time tools
# that we don't need in our final, lean production image.
# We use a slim Debian "bookworm" image for a good balance of size and stability.
FROM python:3.10-slim-bookworm AS builder

# --- Environment Variables ---
# Set common Python environment variables for optimized container behavior.
# PYTHONUNBUFFERED: Prevents Python from buffering stdout/stderr, ensuring logs appear instantly.
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files, which are not needed in a container.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Set common pip environment variables to optimize installs.
# PIP_NO_CACHE_DIR: Disables the pip cache, saving space in the image.
# PIP_DISABLE_PIP_VERSION_CHECK: Supresses warnings about outdated pip versions during builds.
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on

# --- System Dependencies ---
# Install build-essential for compiling C-extensions, AND netcat for the entrypoint.
# We install netcat here in the builder to avoid bloating the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# --- Python Dependencies ---
# Set the working directory for this stage.
WORKDIR /app

# Copy only the requirements file first. This leverages Docker's layer caching.
# If requirements.txt doesn't change, Docker will reuse the cached layer from a
# previous build, making subsequent builds much faster.
COPY requirements.txt .

# Install all Python dependencies specified in requirements.txt.
# We upgrade pip first as a best practice.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# ==============================================================================
# --- Stage 2: The "Final" Production Stage                                  ---
# ==============================================================================
# Purpose: This stage creates the final, minimal, and secure image. It copies
# only the necessary artifacts from the "builder" stage.
FROM python:3.10-slim-bookworm AS final

# --- START: FIX FOR PRODUCTION DOCKERFILE ---
# Install the required system library for OpenCV/Pillow at runtime.
# This is a runtime dependency and must be in the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# --- END: FIX ---

# --- Environment Variables ---
# Set the same Python environment variables for the final image.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Set the PYTHONPATH. This tells Python to also look for modules in our app's
# root directory, which helps with clean, absolute imports in your project.
ENV PYTHONPATH=/home/appuser/app

# --- User and Group Setup (Security Best Practice) ---
# Create a dedicated, non-root user and group to run the application.
# Running as a non-root user is a critical security measure to limit the
# potential impact of a container compromise.
# `-r` creates a system user, `--no-log-init` is a modern performance optimization.
RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup appuser

# --- Copy Artifacts from Builder Stage ---
# Copy the installed Python packages from the 'builder' stage. These files will
# initially be owned by root, which is standard and secure.
# These are copied as root.
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
# Copy any command-line executables installed by pip (like 'gunicorn', 'flask', etc.).
COPY --from=builder /usr/local/bin /usr/local/bin

# --- OPTIMIZATION ---
# Copy ONLY the 'nc' binary from the builder. This keeps the final image
# leaner and more secure than installing the entire 'netcat-openbsd' package here.
# The `nc.openbsd` is the actual executable name on Debian.
COPY --from=builder /bin/nc.openbsd /bin/nc

# --- Application Setup ---
# Create and set the working directory in the user's home directory.
# This follows Linux conventions and creates a clear ownership boundary.
WORKDIR /home/appuser/app

# --- The Optimal Ownership and Permission Pattern ---
# Use the '--chown' flag to set the correct owner and group AS the files are
# being copied. This is fast and eliminates the need for a slow 'chown -R' command.
COPY --chown=appuser:appgroup . .

# Set the executable bit on the entrypoint script. This is run as root, so it
# is guaranteed to succeed.
RUN chmod +x ./docker-entrypoint.sh

# --- Final User Switch (Security Best Practice) ---
# Now that all files are in place with correct ownership and permissions,
# we switch to the non-root user for the final execution of the container.
USER appuser

# --- Port Exposure ---
# Document which ports the application services listen on.
# This does not actually publish the ports; it's for documentation and can be
# used by some automation tools. Publishing is done with `-p` in `docker run`.
# Flask API
EXPOSE 5001
# FastAPI API
EXPOSE 8000
# Flask GUI
EXPOSE 8501
# FastAPI GUI
EXPOSE 8502

# --- Execution ---
# Set the entrypoint to our custom script. We use the full path for clarity.
# This script will be the first thing that runs when the container starts.
ENTRYPOINT ["/home/appuser/app/docker-entrypoint.sh"]

# Set the default command. This value will be passed as an argument to the
# ENTRYPOINT script if no other command is provided in `docker run` or `docker-compose`.
CMD ["scheduler"]