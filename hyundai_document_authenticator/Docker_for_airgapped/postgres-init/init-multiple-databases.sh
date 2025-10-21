#!/bin/bash
# ==============================================================================
#            ROBUST & IDEMPOTENT POSTGRESQL INITIALIZATION SCRIPT
# ==============================================================================
#
# --- PURPOSE ---
# This script is automatically executed by the official PostgreSQL Docker image
# on its very first launch. Its primary job is to perform custom setup tasks
# AFTER the main user and the first database have been created.
#
# --- CORE LOGIC ---
# 1. It checks for an environment variable that holds the name of a second database.
# 2. If the variable exists, it safely creates that second database.
# 3. If the variable is missing, it does nothing and exits cleanly.
# 4. It is IDEMPOTENT: it will not produce an error if it is run again when the
#    database already exists.
#
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
C_BOLD='\033[1m'
C_RESET='\033[0m'

# ==============================================================================
#            CONDITIONAL CREATION OF THE SECOND (USERS) DATABASE
# ==============================================================================

# --- Step 1: Check if the second database is requested ---
# We check if the environment variable 'POSTGRES_USER_DB_NAME' was passed to
# this container. The '-n' test checks if the variable string is not empty.
#
#   - If it exists (provided via docker-compose), the 'then' block will run.
#   - If it's missing or empty, the 'else' block will run, and the script
#     will finish without error.
if [ -n "$POSTGRES_USER_DB_NAME" ]; then

    echo -e "\n${C_BLUE}--------------------------------------------------${C_RESET}"
    echo -e "${C_BOLD}⚙️  Custom Database Initialization Started${C_RESET}"
    echo -e "${C_BLUE}--------------------------------------------------${C_RESET}"
    echo -e "   -> Found request to create a second database: ${C_BOLD}'$POSTGRES_USER_DB_NAME'${C_RESET}"

    # --- Step 2: Check if the database already exists (Idempotency) ---
    # We use psql to run a quiet query. If the database exists, the query will
    # return '1'. If not, it returns nothing.
    #   -t : Tuples only (no headers/footers)
    #   -A : Unaligned (no column spacing)
    #   -c "..." : The SQL command to execute
    DB_EXISTS=$(psql -tA --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -c "SELECT 1 FROM pg_database WHERE datname='$POSTGRES_USER_DB_NAME'")

    if [ "$DB_EXISTS" = "1" ]; then
        # If the database already exists, we log a message and do nothing more.
        echo -e "${C_YELLOW}   🟡 Database '$POSTGRES_USER_DB_NAME' already exists. Skipping creation.${C_RESET}"
    else
        # If the database does not exist, we proceed with creation.
        echo -e "   -> Database '$POSTGRES_USER_DB_NAME' not found. Creating..."

        # 'psql' is the PostgreSQL command-line tool available inside the container.
        # We use it to connect to the already-running server and execute SQL commands.
        #   -v ON_ERROR_STOP=1  : A psql variable that ensures the command fails if any SQL error occurs.
        #   --username "$POSTGRES_USER" : Connects as the main user (e.g., 'jeevan'). This variable
        #                                is available from the container's environment.
        #   --dbname "$POSTGRES_DB"   : Connects to the FIRST database that was already created.
        #                                We must connect to an existing DB to run commands.
        #   <<-EOSQL ... EOSQL  : A "heredoc" which passes a multi-line string of SQL
        #                         commands to psql.
        psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
            -- This SQL command creates the second database. The variable name comes
            -- from the environment variable we are checking.
            CREATE DATABASE $POSTGRES_USER_DB_NAME;

            -- This command gives our main user full control over the new database so it
            -- can create tables, read, and write data.
            GRANT ALL PRIVILEGES ON DATABASE $POSTGRES_USER_DB_NAME TO $POSTGRES_USER;
EOSQL
        echo -e "${C_GREEN}   ✓ Database '$POSTGRES_USER_DB_NAME' created and privileges granted successfully.${C_RESET}"
    fi
    echo -e "${C_BLUE}--------------------------------------------------${C_RESET}"
    echo -e "${C_BOLD}✅ Custom Database Initialization Complete${C_RESET}"
    echo -e "${C_BLUE}--------------------------------------------------${C_RESET}\n"

else
    # This block runs if the environment variable for the second DB was NOT found.
    # This is a normal, expected outcome when running in a single-database setup.
    echo -e "\n${C_YELLOW}🟡 INFO: Second database variable 'POSTGRES_USER_DB_NAME' is not set. No custom databases will be created.${C_RESET}\n"
fi