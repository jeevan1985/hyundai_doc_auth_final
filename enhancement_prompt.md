# Prompt for AI Code Enhancement Agent

**Your Persona and Role:**

You are a Principal Software Architect and Code Quality Auditor with extensive experience in building and maintaining large-scale, production-grade systems. Your primary mission is to perform a deep, holistic audit of an entire codebase to identify not just surface-level bugs, but also underlying architectural inconsistencies, potential security risks, and maintainability issues. You are meticulous, thorough, and your analysis is always constructive and actionable. Your goal is to improve the codebase by directly fixing the issues you identify.

---

**Objective:**

Your task is to read, understand, and fix the critical, architectural, and quality issues outlined in the Code Audit Report below. You must adhere to the recommendations provided in the report while also respecting the key design constraint that the system must function perfectly with only the FAISS vector database (`faiss_manager.py`). The Qdrant implementation (`qdrant_manager.py`) is an optional, "Pro" feature and the codebase must remain robust to its absence.

---

**Core Instructions & Constraints:**

1.  **Analyze the Report:** Carefully review each item in the Code Audit Report provided below.
2.  **Prioritize Fixes:** Address the issues in order of severity: `Critical`, `High`, `Medium`, `Low`, then `Style`.
3.  **Implement Recommendations:** Apply the fixes exactly as described in the "Recommendation" for each issue.
4.  **Respect Optional Qdrant:** This is the most important constraint. When modifying any code that interacts with `qdrant_manager.py` or its related configurations, you **must** ensure that the code continues to work correctly if `qdrant_manager.py` is deleted from the filesystem. All interactions with Qdrant components must be guarded by `try...except ImportError` blocks or similar mechanisms that allow the system to degrade gracefully.
5.  **Atomic Changes:** Apply each fix as a distinct, logical change. Do not bundle unrelated fixes together.
6.  **Verify Your Work:** After applying a fix, briefly state how you have verified it or why you believe it is correct. You do not need to run the code, but you should reason about its correctness.
7.  **Do Not Introduce New Issues:** Your changes should only resolve the identified problems. Do not introduce new features, logic, or dependencies.

---

# Code Audit Report

This report details findings from a comprehensive audit of the codebase, categorized by severity and type.

## 1. Critical Bugs & Runtime Errors

No current critical issues found. Verified against the codebase:
- augmentation_orchestrator.py already calls self.base_manager._ensure_active_index_initialized() for FAISS initialization in persistent augmentation paths.
- feature_extractor.py uses self._feature_dim_cache; the earlier typo has been corrected.

## 2. Architectural & Design Flaws

*   **File:** `D:\frm_git\hyundai_document_authenticator\hyundai_document_authenticator\core_engine\image_similarity_system\faiss_manager.py` and `D:\frm_git\hyundai_document_authenticator\hyundai_document_authenticator\core_engine\image_similarity_system\qdrant_manager.py`
*   **Line(s):** N/A (Architectural)
*   **Severity:** `High`
*   **Description:** The system is designed to treat Qdrant as an optional, "Pro" feature, while FAISS serves as the base. The codebase correctly handles the absence of `qdrant_manager.py` by using `try...except` blocks around imports, ensuring the base version runs without error. However, for the "Pro" version where both managers are present, there are significant architectural inconsistencies. `FaissIndexManager` and `QdrantManager` have different application-level partitioning logic, separate configuration keys (`total_indexes_per_file` vs. `max_points_per_collection`), and different default capacity values (250,000 vs. 500,000). This divergence increases maintenance complexity and can lead to surprising behavior depending on the provider chosen by a "Pro" client.
*   **Recommendation:** Unify the configuration and partitioning logic. Create a base `VectorDBManager` class that defines a consistent interface. Both `FaissIndexManager` and `QdrantManager` should implement this interface. Standardize the configuration with a single, provider-agnostic key like **`partition_capacity`**. This key would define the maximum number of vectors stored in a single FAISS index file or a single Qdrant collection before the system creates a new one. This provides a consistent experience for users and simplifies maintenance.

*   **File:** `D:\frm_git\hyundai_document_authenticator\docker-entrypoint.sh`
*   **Line(s):** 135-165
*   **Severity:** `Medium`
*   **Description:** The entrypoint script uses `sed` to dynamically modify YAML configuration files. While this avoids extra dependencies, it is brittle and error-prone. If the structure of the YAML files changes (e.g., different indentation, new parent keys), the `sed` commands are likely to fail silently, leading to incorrect configurations at runtime.
*   **Recommendation:** Replace `sed` with a more robust method for configuration. The best practice is to read all configuration from environment variables directly within the Python application. This makes the container image immutable and the configuration explicit. If file-based configuration is still necessary, use a tool like `yq` or a small Python script within the entrypoint to safely merge environment variables into the YAML structure.

*   **File:** `D:\frm_git\hyundai_document_authenticator\hyundai_document_authenticator\core_engine\image_similarity_system\workflow.py`
*   **Line(s):** 79-95
*   **Severity:** `Medium`
*   **Description:** The `compute_threshold_match_count` and `compute_fraud_probability` functions are defined as fallbacks within the `workflow.py` module if they cannot be imported from `risk_scoring`. This indicates that `risk_scoring` is an optional component, but this pattern violates the Single Responsibility Principle. The workflow module should orchestrate, not define, scoring logic. This also makes the code harder to test and reason about, as the scoring logic is conditionally defined in two different places.
*   **Recommendation:** Move the fallback functions to a separate, dedicated module (e.g., `risk_scoring_fallback.py`). The `workflow.py` module should then attempt to import from `risk_scoring` and, on failure, import from `risk_scoring_fallback`. This keeps the responsibilities clean and the logic centralized.

## 3. Code Quality & Maintainability ("Code Smells")

*   **File:** `D:\frm_git\hyundai_document_authenticator\hyundai_document_authenticator\core_engine\image_similarity_system\cli.py`
*   **Line(s):** 1-9
*   **Severity:** `High`
*   **Description:** The file `cli.py` is "dead code". It immediately raises a `RuntimeError`, indicating it is deprecated and no longer in use. This file adds clutter to the codebase and can confuse developers who might assume it's a functional entrypoint.
*   **Recommendation:** Delete the file `D:\frm_git\hyundai_document_authenticator\hyundai_document_authenticator\core_engine\image_similarity_system\cli.py`.

*   **File:** `D:\frm_git\hyundai_document_authenticator\hyundai_document_authenticator\core_engine\image_similarity_system\utils.py`
*   **Line(s):** 464
*   **Severity:** `Medium`
*   **Description:** The function `save_tif_search_results_to_postgresql` is marked as `[DEPRECATED]` in its docstring, indicating it should no longer be used. A codebase search confirms it is not called by any active workflow. Keeping deprecated, dead code increases maintenance overhead and can lead to accidental use in the future.
*   **Recommendation:** Remove the `save_tif_search_results_to_postgresql` function from `utils.py`.

*   **File:** `D:\frm_git\hyundai_document_authenticator\hyundai_document_authenticator\core_engine\image_similarity_system\config_loader.py`
*   **Line(s):** 311
*   **Severity:** `Low`
*   **Description:** The function `get_config_value_by_key_path` accepts a `default_value` in `**kwargs` for backward compatibility. This creates a redundant and slightly confusing API. The standard `default` argument should be used exclusively.
*   **Recommendation:** Deprecate and remove the `default_value` kwarg. Update any internal call sites to use the `default` positional argument instead.

*   **File:** `D:\frm_git\hyundai_document_authenticator\hyundai_document_authenticator\core_engine\image_similarity_system\datasets.py`
*   **Line(s):** 71 and 151
*   **Severity:** `Style`
*   **Description:** The code uses `except Exception as e: # pylint: disable=broad-except`. While the broad exception is caught, disabling the linter warning hides a "code smell". It's better to catch specific, expected exceptions (e.g., `IOError`, `cv2.error`) and let unexpected ones propagate.
*   **Recommendation:** Replace `except Exception` with more specific exception types where possible (e.g., `IOError`, `AttributeError`, `cv2.error`). If a broad exception is truly necessary for robustness, add a comment explaining why specific exceptions are not sufficient.

## 4. Security Vulnerabilities

*   **File:** `D:\frm_git\hyundai_document_authenticator\hyundai_document_authenticator\tool_database_tester.py`
*   **Line(s):** Multiple locations (CREATE DATABASE and DROP DATABASE statements)
*   **Severity:** `High`
*   **Description:** The script uses f-strings to construct SQL queries (e.g., `f'CREATE DATABASE "{dbname}"'`). This is a significant SQL injection risk if the input variables (`dbname`, `table`, etc.) are ever sourced from user input or a non-trusted source. While this specific tool is for developers, it sets a dangerous precedent.
*   **Recommendation:** Use parameterized queries exclusively. The `psycopg2` library supports this natively. For DDL statements where identifiers cannot be parameterized, use the `psycopg2.sql` module to safely compose identifiers.
    ```python
    # Example fix
    from psycopg2 import sql

    # For CREATE DATABASE
    cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))

    # For DROP TABLE
    ident = sql.Identifier(schema, table)
    cur.execute(sql.SQL("DROP TABLE IF EXISTS {};").format(ident))
    ```

*   **File:** `D:\frm_git\hyundai_document_authenticator\docker-compose.conda.dev.yaml`
*   **Line(s):** 59
*   **Severity:** `Medium`
*   **Description:** The development compose file exposes the PostgreSQL database port `5433:5432` to the host machine. While convenient for development, this exposes the database to the local network, increasing the attack surface.
*   **Recommendation:** For a more secure development setup, remove the `ports` mapping. Developers can connect to the database from within the Docker network using `docker exec` or by connecting other tools to the same Docker network. If host access is required, add a prominent warning in the documentation about the security implications.

*   **File:** `D:\frm_git\hyundai_document_authenticator\Dockerfile.conda.dev`
*   **Line(s):** 50
*   **Severity:** `Medium`
*   **Description:** The development Dockerfile uses `ARG HOST_UID=1000` and `ARG HOST_GID=1000` to set file permissions. This is a good practice to avoid permission issues with bind mounts. However, if a developer's UID/GID on their host machine is not 1000, they will still encounter permission errors.
*   **Recommendation:** Add a note in the development setup guide (`README.md` or a `CONTRIBUTING.md`) instructing developers on how to pass their local UID/GID during the build process:
    ```bash
    docker-compose build --build-arg UID=$(id -u) --build-arg GID=$(id -g)
    ```
    This makes the development setup more robust for multi-user environments.

## 5. Documentation & Configuration Mismatches

*   **File:** `D:\frm_git\hyundai_document_authenticator\hyundai_document_authenticator\configs\image_similarity_config_TEMPLATE.yaml`
*   **Severity:** `Low`
*   Status: Already consistent. The template correctly documents sharding as the default and sets total_indexes_per_file: 250000. No change required.

*   **File:** `D:\frm_git\hyundai_document_authenticator\docker-entrypoint.sh`
*   **Severity:** `Low`
*   Status: Already addressed. The 'cli' case is now flexible and accepts a script name argument, forwarding the writable config path. Example usage:
    docker compose -f docker-compose.conda.yaml run --rm cli_runner cli doc_image_verifier.py search-doc --folder ...

