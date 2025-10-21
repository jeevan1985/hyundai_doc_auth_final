#!/bin/sh
# ==============================================================================
#                 SIMPLE & ROBUST DOCKER ENTRYPOINT SCRIPT
# ==============================================================================
#
# --- PURPOSE ---
# This script is the single entrypoint for the application container. It is
# responsible for two primary tasks:
#
#   1. DYNAMIC CONFIGURATION: It inspects environment variables at runtime to
#      configure the application (e.g., switch between SQLite and PostgreSQL)
#      without needing to rebuild the Docker image.
#
#   2. PROCESS LAUNCHING: It determines which application component to run
#      (API, GUI, CLI) based on the command passed to the container and uses
#      'exec' to launch it correctly.
# ==============================================================================

# --- Script Configuration ---
# 'set -e' is a critical safety measure. It ensures that the script will exit
# immediately if any command fails. This prevents the container from starting
# in a broken or unpredictable state.
set -e

# --- Terminal Color Constants (for beautiful logs) ---
C_BLUE='\033[94m'
C_GREEN='\033[92m'
C_YELLOW='\033[93m'
C_RED='\033[91m'
C_BOLD='\033[1m'
C_RESET='\033[0m'

# ==============================================================================
#                         HELPER FUNCTIONS
# ==============================================================================

# --- Function: Configure Qdrant Mode ---
# Dynamically modifies YAML config files to switch between 'server' (network)
# and 'embedded' (local file) modes for the Qdrant vector database.
configure_qdrant() {
    # Define paths to the configuration files.
    API_CONFIG="configs/api_config.yaml"
    CLI_CONFIG="configs/image_similarity_config.yaml"

    # Robustness Check: Ensure the files exist before attempting to modify them.
    if [ ! -f "$API_CONFIG" ] || [ ! -f "$CLI_CONFIG" ]; then
        echo -e "${C_YELLOW}🟡 Warning: One or more config files not found. Skipping Qdrant configuration.${C_RESET}"
        return
    fi

    # Read the QDRANT_MODE from the environment. Default to 'embedded' if not set.
    # In our docker-compose setup, this is always overridden to 'server'.
    QDRANT_MODE=${QDRANT_MODE:-embedded}

    echo -e "\n${C_BLUE}--------------------------------------------------${C_RESET}"
    echo -e "${C_BOLD}⚙️  Configuring Qdrant Mode: [${QDRANT_MODE}]${C_RESET}"
    echo -e "${C_BLUE}--------------------------------------------------${C_RESET}"

    # Use a case statement for clarity.
    case "$QDRANT_MODE" in
        server)
            echo "   -> Enabling network settings for Server Mode..."
            # 'sed' is the "stream editor" command. These commands find and uncomment
            # the network settings (host/port) and comment out the local file setting.
            sed -i -E 's/^(\s*)#\s*(host:.*)/\1\2/' "$API_CONFIG" "$CLI_CONFIG"
            sed -i -E 's/^(\s*)#\s*(port:.*)/\1\2/' "$API_CONFIG" "$CLI_CONFIG"
            sed -i -E 's/^(\s*)(location:.*)/#\1\2/' "$API_CONFIG" "$CLI_CONFIG"
            echo -e "${C_GREEN}   ✓ Configured for Server Mode.${C_RESET}"
            ;;
        *) # Default to embedded mode
            echo "   -> Enabling local file settings for Embedded Mode..."
            sed -i -E 's/^(\s*)(host:.*)/#\1\2/' "$API_CONFIG" "$CLI_CONFIG"
            sed -i -E 's/^(\s*)(port:.*)/#\1\2/' "$API_CONFIG" "$CLI_CONFIG"
            sed -i -E 's/^(\s*)#\s*(location:.*)/\1\2/' "$API_CONFIG" "$CLI_CONFIG"
            echo -e "${C_GREEN}   ✓ Configured for Embedded Mode.${C_RESET}"
            ;;
    esac
}

# --- Function: Configure Database (CORRECTED) ---
# Sets the database connection string(s) to use PostgreSQL ONLY IF the required
# credentials are provided in the environment. Otherwise, assumes a fallback
# to the application's default (SQLite).
configure_database() {
    API_CONFIG="configs/api_config.yaml"

    # Robustness Check: Ensure the config file exists.
    if [ ! -f "$API_CONFIG" ]; then
        echo -e "${C_YELLOW}🟡 Warning: API Config file not found at $API_CONFIG. Skipping Database configuration.${C_RESET}"
        return
    fi

    # --- The Conditional "Switch" ---
    # This is the most important logic. We check for a credential that is ONLY
    # set in the .env file. If it's missing, we know the user intends to run in
    # SQLite mode, so we exit this function early.
    if [ -z "$POSTGRES_USER" ] || [ -z "$POSTGRES_PASSWORD" ]; then
        echo -e "${C_YELLOW}🟡 INFO: PostgreSQL credentials not set in environment. Application will use its default database (e.g., SQLite).${C_RESET}"
        return
    fi

    # If we reach this point, the credentials WERE found, so we proceed to configure for PostgreSQL.
    echo -e "\n${C_BLUE}--------------------------------------------------${C_RESET}"
    echo -e "${C_BOLD}⚙️  Configuring Database for PostgreSQL${C_RESET}"
    echo -e "${C_BLUE}--------------------------------------------------${C_RESET}"
    echo "   -> PostgreSQL credentials found. Setting database URIs in config file..."

    # Construct the URI for the Search Results DB.
    # The ':-5432' syntax provides a default value for the port if it's not set in the environment.
    RESULTS_DB_URI="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT:-5432}/${POSTGRES_DB}"

    # Construct the URI for the User DB, using its own set of variables.
    USER_DB_URI="postgresql://${POSTGRES_USER_USER}:${POSTGRES_USER_PASSWORD}@${POSTGRES_USER_HOST}:${POSTGRES_USER_PORT:-5432}/${POSTGRES_USER_DB}"

    # Use 'sed' to robustly find and replace the database_uri lines in the config.
    # This regex is powerful because it works whether the line is currently commented out or not.
    # NOTE: This assumes your config file has keys named 'database_uri' and 'user_database_uri'.
    # If your keys are different, you must update the text here.
    sed -i -E "s|^(\s*#\s*)?database_uri:.*|database_uri: ${RESULTS_DB_URI}|" "$API_CONFIG"
    sed -i -E "s|^(\s*#\s*)?user_database_uri:.*|user_database_uri: ${USER_DB_URI}|" "$API_CONFIG"

    echo -e "${C_GREEN}   ✓ Database URIs have been configured to use PostgreSQL.${C_RESET}"
}

# ==============================================================================
#                         MAIN EXECUTION
# ==============================================================================

# 1. Run all configuration functions. This prepares the config files before the app starts.
echo "--- Running Entrypoint Configurations ---"
configure_qdrant
configure_database
echo "--- Configurations Complete ---"

# 2. Determine the application mode from the first command-line argument ($1).
#    If no argument is provided, default to 'flask-api'.
MODE=${1:-flask-api}

echo -e "\n${C_BLUE}==================================================${C_RESET}"
echo -e "${C_BOLD}🚀 LAUNCHING APPLICATION: [${MODE}]${C_RESET}"
echo -e "${C_BLUE}==================================================${C_RESET}\n"

# The 'case' statement is a clean way to handle multiple possible commands.
#
# CRITICAL: 'exec' replaces this entrypoint script's process with the
# final application's process. This is essential for correct signal handling
# (like Ctrl+C or 'docker stop') and proper process management by Docker.
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
        echo "--> Starting Flask GUI on port 8501 with Gunicorn..."
        exec gunicorn --bind 0.0.0.0:8501 --workers 2 "gui_app.app:app"
        ;;
    fastapi-gui)
        echo "--> Starting FastAPI GUI on port 8502 with Gunicorn/Uvicorn..."
        exec gunicorn -k uvicorn.workers.UvicornWorker --workers 2 --bind 0.0.0.0:8502 "fastapi_gui.main:app"
        ;;
    cli)
        echo "--> Executing Core CLI (find_sim_images.py)..."
        shift # Removes 'cli' from the argument list.
        # '$@' correctly passes all remaining arguments (like --action) to the script.
        exec python find_sim_images.py "$@"
        ;;
    manage)
        echo "--> Executing User Management CLI (manage_api_users.py)..."
        shift
        # Use 'python -m' to run the script as a module, ensuring correct import paths.
        exec python -m api_server.manage_api_users "$@"
        ;;
    backup)
        echo "--> Executing Backup Tool (backup_tool.py)..."
        shift
        exec python backup_tool.py "$@"
        ;;
    bash | sh)
        echo "--> Entering interactive shell for debugging..."
        # Use /bin/sh for maximum portability, as it exists in almost all base images.
        exec /bin/sh
        ;;
    *)
        # This block runs if the command is not recognized, providing helpful usage info.
        echo -e "${C_RED}❌ Error: Unknown command '$MODE'${C_RESET}" >&2 # Send error to stderr
        echo "" >&2
        echo -e "${C_BOLD}Available commands:${C_RESET}" >&2
        echo "  flask-api, fastapi-api, gui, fastapi-gui, cli, manage, backup, bash" >&2
        exit 1 # Exit with a non-zero status code to indicate failure.
        ;;
esac