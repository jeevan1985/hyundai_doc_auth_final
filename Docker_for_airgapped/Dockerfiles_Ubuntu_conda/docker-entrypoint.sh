#!/bin/sh
# ==============================================================================
#      BULLETPROOF, DOCUMENTED, & COMPLIANT DOCKER ENTRYPOINT SCRIPT
# ==============================================================================
#
# This script is the single entrypoint for the application container. It is
# designed to be secure, robust, and resilient for production environments. It
# has been heavily commented to be understandable by users of all skill levels.
#
# It is responsible for four key tasks:
#
#   1. RESILIENCE (WAITING FOR DEPENDENCIES): It intelligently pauses execution
#      to wait for critical services (like a database) to be online before
#      the application starts. This prevents common startup crashes.
#
#   2. DYNAMIC CONFIGURATION (using SED): It uses the standard 'sed' utility
#      to modify configuration files based on environment variables. This
#      avoids dependencies on third-party tools like 'yq'.
#
#   3. IMMUTABILITY (COPY-ON-WRITE): It handles read-only configuration by
#      copying the source configs to a writable location inside the container
#      before modifying them. This is a critical security best practice.
#
#   4. PROCESS LAUNCHING (EXEC): It determines which application component to
#      run and uses 'exec' to correctly launch it as the main container
#      process, ensuring proper shutdown signals are received.
#
# It is assumed that the dependency 'nc' (netcat) is provided by the Docker image.
#
# ==============================================================================


# ==============================================================================
#                      SECTION 1: SCRIPT CONFIGURATION
# ==============================================================================

# --- Fail-Fast Behavior ---
# These settings are critical for creating reliable scripts.
#
# 'set -e': This command ensures that the script will exit immediately if any
#           command fails (returns a non-zero exit code). Without this, the
#           script might continue in a broken state.
#
# 'set -u': This command treats unset or unbound variables as an error and
#           causes the script to exit immediately. This is a great safety net
#           to catch typos in environment variable names. For example, if you
#           typed 'POSTGRES_PASWORD' instead of 'POSTGRES_PASSWORD', this
#           setting would cause the script to fail instantly.
set -eu

# --- Path Constants ---
# We define paths as variables for clarity, easy maintenance, and to avoid
# repeating magic strings throughout the script.
#
# SOURCE_CONFIG_DIR:   The path to the original, read-only configuration
#                      directory mounted into the container.
SOURCE_CONFIG_DIR="configs"
# WRITABLE_CONFIG_DIR: A temporary, writable directory inside the container's
#                      filesystem where we will place the modified configs.
WRITABLE_CONFIG_DIR="/tmp/runtime_configs"

# --- Terminal Color Constants ---
# These variables hold ANSI escape codes for adding color to the log output.
# This dramatically improves the readability of Docker logs.
C_BLUE='\033[94m'
C_GREEN='\033[92m'
C_YELLOW='\033[93m'
C_RED='\033[91m'
C_BOLD='\033[1m'
C_RESET='\033[0m' # This special code resets all text formatting.


# ==============================================================================
#                      SECTION 2: HELPER FUNCTIONS
# ==============================================================================
# This section contains functions that handle the dynamic configuration logic.
# Encapsulating logic in functions makes the main execution section clean and readable.

# --- Function: wait_for_service ---
# This function improves the resilience of the application stack. It pauses the
# script until a specific network service (like a database) is ready to accept
# connections, preventing the main application from crashing on startup.
#
# Usage: wait_for_service "Service Name" "hostname" "port"
wait_for_service() {
    local service_name="$1"
    local host="$2"
    local port="$3"

    # --- NOTE: This message is now slightly different to indicate the service is already known to be active. ---
    echo -e "${C_YELLOW}🟡 $service_name is active. Waiting for it to become fully ready at $host:$port...${C_RESET}"

    # This is the wait loop. It uses 'nc' (netcat) to check the port.
    #
    # Command Breakdown:
    #   'while ! ...': The '!' inverts the result. The loop will continue as long
    #                  as the 'nc' command FAILS (i.e., the port is not open).
    #   'nc':          Netcat, a utility for network connections.
    #   '-z':          The "zero-I/O" or "scanning" mode. 'nc' will just check if
    #                  the port is open without sending any data, and then exit.
    while ! nc -z "$host" "$port"; do
        # If the check fails, print a message and wait for 2 seconds before trying again.
        echo "   ... still waiting"
        sleep 2
    done

    echo -e "${C_GREEN}✅ $service_name is up and ready!${C_RESET}"
}

# --- Function: prepare_writable_configs ---
# This function implements the "copy-on-write" security best practice. It copies
# configuration files from a potentially read-only source (mounted from the
# host) to a temporary, writable destination inside the container. All subsequent
# modifications are made to these copies.
prepare_writable_configs() {
    # Step 1: Check if the source directory actually exists.
    #
    # Command Breakdown:
    #   'if [ ... ]': The standard shell conditional test.
    #   '!':          The "not" operator. It inverts the result of the test.
    #   '-d':         A test operator that returns true if the path exists
    #                 and is a Directory.
    if [ ! -d "$SOURCE_CONFIG_DIR" ]; then
        # '>&2' redirects this error message to Standard Error (stderr), which
        # is the correct place for errors.
        echo -e "${C_RED}❌ FATAL: Source config directory '$SOURCE_CONFIG_DIR' not found. Is it mounted correctly?${C_RESET}" >&2
        exit 1
    fi

    # Step 2: Create the writable directory inside the container.
    #
    # Command Breakdown:
    #   'mkdir -p': The '-p' (parent) flag is important; it creates parent
    #               directories as needed and does not throw an error if the
    #               directory already exists. This makes the script re-runnable.
    echo "   -> Preparing writable config directory at '$WRITABLE_CONFIG_DIR'..."
    mkdir -p "$WRITABLE_CONFIG_DIR"

    # Step 3: Copy all files from the source to the destination.
    # The '*' is a glob that matches all files in the source directory.
    cp "$SOURCE_CONFIG_DIR"/* "$WRITABLE_CONFIG_DIR"/
    echo -e "${C_GREEN}   ✓ Writable configs are ready.${C_RESET}"
}

# --- Function: configure_qdrant ---
# This function uses the standard 'sed' utility to modify the Qdrant configuration.
# While powerful, using 'sed' on structured files like YAML is less robust than
# a dedicated parser. This approach is used to avoid third-party dependencies.
configure_qdrant() {
    # 'local' declares variables within the scope of this function only. It's
    # good practice to prevent accidentally modifying global variables.
    local API_CONFIG_FILE="$WRITABLE_CONFIG_DIR/api_config.yaml"
    local CLI_CONFIG_FILE="$WRITABLE_CONFIG_DIR/image_similarity_config.yaml"

    # Robustness Check: Ensure the files exist before trying to modify them.
    # '-f' is a test operator that returns true if the path exists and is a regular File.
    if [ ! -f "$API_CONFIG_FILE" ] || [ ! -f "$CLI_CONFIG_FILE" ]; then
        echo -e "${C_YELLOW}🟡 Warning: Config files not found in writable directory. Skipping Qdrant configuration.${C_RESET}"
        return
    fi

    # Read QDRANT_MODE from the environment.
    # '${VAR:-default}': This syntax provides a default value. If QDRANT_MODE is
    # unset or empty, it will default to 'embedded'.
    local QDRANT_MODE=${QDRANT_MODE:-embedded}
    # --- ADDED: Define host and port variables for clarity and reuse. ---
    local QDRANT_HOST=${QDRANT_HOST:-qdrant}
    local QDRANT_PORT=${QDRANT_PORT:-6333}

    echo -e "\n${C_BOLD}⚙️  Configuring Qdrant Mode: [${QDRANT_MODE}]${C_RESET}"

    # Only wait for the Qdrant service if we are in server mode.
    # This prevents blocking CLI tools that use embedded mode.
    if [ "$QDRANT_MODE" = "server" ]; then
        # ==============================================================================
        # <<< FIX 1: ADD INTELLIGENT SERVICE CHECK BEFORE WAITING >>>
        # ==============================================================================
        # This new 'if' block checks if the service is actually running before we
        # commit to waiting for it. This is critical for working with Docker Compose
        # profiles, where a service like 'qdrant' might not be started at all.
        echo "   -> Server mode enabled. Checking if Qdrant service is active..."

        # Command Breakdown: 'nc -z -w 1 "$QDRANT_HOST" "$QDRANT_PORT" >/dev/null 2>&1'
        #   'nc -z':       Checks if the port is open (just like in wait_for_service).
        #   '-w 1':        This is the key. It sets a timeout of 1 second. If 'nc' cannot
        #                  connect within 1 second, it gives up and fails. This prevents
        #                  the script from hanging indefinitely.
        #   '>/dev/null':  Redirects standard output (like success messages) to nowhere,
        #                  so we don't see it in the logs.
        #   '2>&1':        Redirects standard error (like "connection refused") to the
        #                  same place as standard output (nowhere). This keeps the logs clean.
        # The 'if' statement checks the exit code of this command. It's a quick, silent "ping".
        if nc -z -w 1 "$QDRANT_HOST" "$QDRANT_PORT" >/dev/null 2>&1; then
            # If the quick check succeeds, it means the service is running. Now we can safely
            # enter the longer, more patient wait loop to ensure it's fully initialized.
            wait_for_service "Qdrant" "$QDRANT_HOST" "$QDRANT_PORT"
        else
            # If the quick check fails, we assume the service was not started intentionally.
            # We print a warning and continue, allowing profile-based runs to work.
            echo -e "${C_YELLOW}   -> Qdrant service at '$QDRANT_HOST' is not running. Skipping wait. The application may fail if it requires Qdrant.${C_RESET}"
        fi

        echo "   -> Applying Qdrant server settings via YAML merge utility..."
        # --- Explanation of the 'sed' command ---
        #   'sed -i -E "s/.../.../" file1 file2'
        #   -i: In-place edit. Modifies the files directly.
        #   -E: Use Extended Regular Expressions for cleaner syntax.
        #   "s/find/replace/": The substitution command.
        #
        # --- The Regular Expression: 's/^(\s*)#\s*(host:.*)/\1\2/' ---
        #   ^           - Matches the beginning of the line.
        #   (\s*)       - Group 1: Captures any leading whitespace (spaces, tabs)
        #                 to preserve indentation. '\s*' means zero or more whitespace chars.
        #   #\s*        - Matches the '#' comment character, followed by any whitespace.
        #   (host:.*)   - Group 2: Captures the literal text 'host:' and the rest of the line.
        #   \1\2        - The replacement: Puts back the captured indentation (Group 1)
        #                 followed by the captured setting (Group 2). This effectively
        #                 deletes the '#' comment character between them.

        python -m hyundai_document_authenticator.extra_tools.config_env_merge qdrant \
            --mode server --host "$QDRANT_HOST" --port "$QDRANT_PORT" \
            --file "$API_CONFIG_FILE" --file "$CLI_CONFIG_FILE"
        echo -e "${C_GREEN}   ✓ Qdrant configured for Server Mode (YAML merge).${C_RESET}"
    else
        echo "   -> Applying Qdrant embedded settings via YAML merge utility..."
        python -m hyundai_document_authenticator.extra_tools.config_env_merge qdrant \
            --mode embedded --host "$QDRANT_HOST" --port "$QDRANT_PORT" \
            --file "$API_CONFIG_FILE" --file "$CLI_CONFIG_FILE"
        echo -e "${C_GREEN}   ✓ Qdrant configured for Embedded Mode (YAML merge).${C_RESET}"
    fi
}

# --- Function: configure_database ---
# This function modifies the copied API config to use PostgreSQL if credentials
# are provided in the environment. It will wait for PostgreSQL to be ready
# ONLY IF it's going to be used and is detected as running.
configure_database() {
    local API_CONFIG_FILE="$WRITABLE_CONFIG_DIR/api_config.yaml"

    # First, check if the config file we need to modify even exists.
    if [ ! -f "$API_CONFIG_FILE" ]; then
        echo -e "${C_YELLOW}🟡 Warning: API Config not found. Skipping DB configuration.${C_RESET}"
        return
    fi

    # This is the "switch" that enables PostgreSQL mode.
    #
    # Command Breakdown:
    #   '-z': A test operator that returns true if the variable string is Zero-length (empty).
    #   '${VAR:-}': A special syntax to prevent 'set -u' from erroring if the variable is
    #               completely unset. It provides an empty string as a default for the test.
    if [ -z "${POSTGRES_USER:-}" ]; then
        echo -e "${C_YELLOW}🟡 PostgreSQL credentials not set. Application will use default DB (SQLite).${C_RESET}"
        return
    fi

    # If we get here, it means credentials WERE provided.
    echo -e "\n${C_BOLD}⚙️  Configuring Database for PostgreSQL${C_RESET}"
    
    # --- ADDED: Define host and port variables for clarity and reuse. ---
    local POSTGRES_HOST=${POSTGRES_HOST:-db}
    local POSTGRES_PORT=${POSTGRES_PORT:-5432}

    # ==============================================================================
    # <<< FIX 2: APPLY THE SAME INTELLIGENT CHECK FOR THE DATABASE >>>
    # ==============================================================================
    # This applies the same non-blocking check to the database. This is essential
    # for `cli_runner` commands that might have POSTGRES_USER set in the .env file
    # but are being run without the 'postgres' profile active.
    echo "   -> PostgreSQL credentials found. Checking if database service is active..."
    if nc -z -w 1 "$POSTGRES_HOST" "$POSTGRES_PORT" >/dev/null 2>&1; then
        # If the quick check succeeds, now we can safely wait for the service to be healthy.
        wait_for_service "PostgreSQL" "$POSTGRES_HOST" "$POSTGRES_PORT"
    else
        # If the quick check fails, we print a warning and continue.
        echo -e "${C_YELLOW}   -> PostgreSQL service at '$POSTGRES_HOST' is not running. Skipping wait. The application may fail if it requires the database.${C_RESET}"
    fi

    echo "   -> Applying database URIs via YAML merge utility..."

    python -m hyundai_document_authenticator.extra_tools.config_env_merge database \
        --file "$API_CONFIG_FILE"

    echo -e "${C_GREEN}   ✓ Database URIs configured for PostgreSQL (YAML merge).${C_RESET}"
}


# ==============================================================================
#                      SECTION 3: MAIN EXECUTION
# ==============================================================================
# This is the main workflow of the entrypoint script.

echo -e "${C_BLUE}==================================================${C_RESET}"
echo -e "${C_BOLD}🚀 DOCKER ENTRYPOINT SCRIPT STARTING...${C_RESET}"
echo -e "${C_BLUE}==================================================${C_RESET}"

# Step 1: Prepare the writable configuration files. This must be the first step.
prepare_writable_configs

# Step 2: Run all configuration functions to modify the files in the writable directory.
# The waiting logic is handled inside these functions, so it's conditional.
configure_qdrant
configure_database

# Step 3: Determine the application mode from the first command-line argument.
# '$1' refers to the first argument passed to the script.
# If '$1' is empty (e.g., from 'docker-compose up'), default to 'scheduler'.
MODE=${1:-scheduler}

echo -e "\n${C_BLUE}==================================================${C_RESET}"
echo -e "${C_BOLD}🚀 LAUNCHING APPLICATION IN MODE: [${MODE}]${C_RESET}"
echo -e "${C_BLUE}==================================================${C_RESET}\n"

# Step 4: Launch the correct application process using a 'case' statement.
# This is a clean way to handle multiple possible commands.

# CRITICAL NOTE ON 'exec':
# 'exec' replaces this shell script's process with the final application's process.
# This makes the application PID 1 inside the container, which is essential for
# correctly receiving signals like SIGTERM (from 'docker stop') and SIGINT (Ctrl+C).
# Without 'exec', signals would go to this script, which would exit but might
# leave the actual application running as an orphaned process that Docker can't manage.
case "$MODE" in
    flask-api)
        echo "--> Starting Flask API on port 5001 with Gunicorn..."
        exec gunicorn --bind 0.0.0.0:5001 --workers 4 --timeout 120 --preload "api_server.run_api_server:app"
        ;;

    fastapi-api)
        echo "--> Starting FastAPI on port 8000 with Gunicorn/Uvicorn..."
        exec gunicorn -k uvicorn.workers.UvicornWorker --workers 4 --bind 0.0.0.0:8000 "fastapi_app.main:app"
        ;;

    gui)
        echo "--> Starting Result GUI (Flask + Jinja) on port 8501 with Gunicorn..."
        exec gunicorn --bind 0.0.0.0:8501 --workers 2 --timeout 120 --factory "hyundai_document_authenticator.external.result_gui.app:create_app"
        ;;

    fastapi-gui)
        echo "--> Starting FastAPI GUI on port 8502 with Gunicorn/Uvicorn..."
        exec gunicorn -k uvicorn.workers.UvicornWorker --workers 2 --bind 0.0.0.0:8502 "fastapi_gui.main:app"
        ;;

    cli)
        echo "--> Executing Python script with modified configuration..."
        # --- Flexible Script Execution ---
        # The original script was hardcoded to run 'find_sim_images.py'.
        # This has been modified to allow running any script passed as an argument.
        shift # Discard $1 ('cli').
        
        # The next argument ($1) is now expected to be the Python script to run.
        SCRIPT_NAME=$1
        shift # Discard the script name from the argument list.
        
        # '$@' now contains only the arguments for the Python script itself.
        # The script is executed with a dynamic config path and its own arguments.
        exec python "$SCRIPT_NAME" --config-path "$WRITABLE_CONFIG_DIR/image_similarity_config.yaml" "$@"
        ;;

    manage)
        echo "--> Executing Python module for management..."
        # --- Flexible Module Execution ---
        # Allows running any management module, not just the hardcoded default.
        shift # Discard $1 ('manage').

        # The next argument ($1) is the name of the Python module to execute.
        MODULE_NAME=$1
        shift # Discard the module name from the argument list.

        # Execute the specified module, passing the config path and any other arguments.
        exec python -m "$MODULE_NAME" --config-path "$WRITABLE_CONFIG_DIR/api_config.yaml" "$@"
        ;;

    backup)
        echo "--> Executing Python script for backup..."
        # --- Flexible Script Execution ---
        # Allows running any backup script, not just the hardcoded default.
        shift # Discard $1 ('backup').

        # The next argument ($1) is the name of the Python script to execute.
        SCRIPT_NAME=$1
        shift # Discard the script name from the argument list.

        # Execute the specified script, passing the config path and any other arguments.
        exec python "$SCRIPT_NAME" --config-path "$WRITABLE_CONFIG_DIR/api_config.yaml" "$@"
        ;;

    scheduler)
        echo "--> Starting Scheduler Service..."
        # Runs the scheduler orchestrator that executes doc_image_verifier.py at configured intervals.
        exec python scheduler_service.py
        ;;

    # The '|' acts as an "OR", matching either 'bash' or 'sh'.
    bash | sh)
        echo -e "${C_YELLOW}--> Entering interactive shell for debugging...${C_RESET}"
        echo "--------------------------------------------------------"
        echo "Your original, read-only configs are in: $SOURCE_CONFIG_DIR"
        echo "Your modified, runtime configs are in:   $WRITABLE_CONFIG_DIR"
        echo "Example: 'cat $WRITABLE_CONFIG_DIR/api_config.yaml' to see modified file."
        echo "--------------------------------------------------------"
        exec /bin/sh
        ;;

    *)
        # This is the default "catch-all" case for any unknown command.
        # It redirects output to stderr (>&2) and exits with an error code.
        echo -e "${C_RED}❌ Error: Unknown command '$MODE'${C_RESET}" >&2
        echo "" >&2
        echo "Usage: docker-compose run <service> [MODE]" >&2
        echo "Available modes: flask-api, fastapi-api, gui, fastapi-gui, cli, manage, backup, scheduler, bash" >&2
        exit 1
        ;;
esac