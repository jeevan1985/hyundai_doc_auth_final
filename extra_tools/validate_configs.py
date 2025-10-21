"""Cross-module configuration validator for production readiness.

This tool validates YAML configurations across core and external modules for
syntactic correctness and basic semantic compatibility. It also checks whether
required settings are resolvable via YAML or environment variables and
highlights privacy/safety pitfalls.

Usage:
    python extra_tools/validate_configs.py

Exit codes:
    0 -> All checks passed (no errors). Warnings may be present.
    1 -> One or more errors detected.

Notes:
    - Uses only the standard library and PyYAML for broad compatibility.
    - Reads optional .env at repository root and merges with process env.
      This does not replicate each module's precedence rules; it is only
      a best-effort validator asserting the presence of required inputs.

Author:
    Production Engineering (validated for Windows/Linux)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import sys

try:
    import yaml
except Exception as exc:  # pragma: no cover
    print(f"ERROR: PyYAML is required but not available: {exc}")
    sys.exit(2)


@dataclass
class CheckResult:
    """Validation result container.

    Attributes:
        file (Path): Path to the validated file.
        errors (List[str]): Blocking issues that must be addressed.
        warnings (List[str]): Non-blocking advisories.
    """

    file: Path
    errors: List[str]
    warnings: List[str]


def _read_yaml(path: Path) -> Any:
    """Read a YAML file using safe_load.

    Args:
        path: YAML file path.

    Returns:
        Any: Parsed content (mapping/list/scalar or None).

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If YAML is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as exc:  # pragma: no cover
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc


def _parse_dotenv(env_path: Path) -> Dict[str, str]:
    """Parse a simple .env file into a mapping.

    Only supports KEY=VALUE lines, strips surrounding quotes.

    Args:
        env_path: Path to .env file.

    Returns:
        Dict[str, str]: Key/value pairs from .env.
    """
    if not env_path.exists():
        return {}
    out: Dict[str, str] = {}
    with env_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip()
            val = v.strip().strip("\"").strip("'")
            if key:
                out[key] = val
    return out


def _env_merged(repo_root: Path) -> Dict[str, str]:
    """Resolve environment by merging .env at repo root with process env.

    Args:
        repo_root: Repository root path used to locate .env.

    Returns:
        Dict[str, str]: Combined mapping with process env taking precedence.
    """
    env_file = repo_root / ".env"
    return {**_parse_dotenv(env_file), **os.environ}


# ------------------------------- Validators ---------------------------------

def validate_image_similarity(path: Path, env: Dict[str, str]) -> CheckResult:
    """Validate image similarity configuration.

    Validates high-level safety and compatibility rules:
      - Bruteforce provider not allowed with privacy_mode=true.
      - If save_results_to_postgresql=true, Postgres connection fields must be resolvable.
      - Qdrant provider checks for either embedded location or host/port presence.

    Args:
        path: Path to image_similarity_config*.yaml.
        env: Merged environment mapping.

    Returns:
        CheckResult: Accumulated warnings and errors.
    """
    errors: List[str] = []
    warnings: List[str] = []

    try:
        data = _read_yaml(path) or {}
        if not isinstance(data, dict):
            return CheckResult(path, ["Top-level YAML must be a mapping"], [])
    except Exception as exc:
        return CheckResult(path, [f"Read error: {exc}"], [])

    # Provider vs privacy constraints
    provider = (data.get("vector_database", {}).get("provider") or "faiss").lower()
    search_task = data.get("search_task", {}) or {}
    privacy_mode = bool(search_task.get("privacy_mode", True))

    if provider == "bruteforce" and privacy_mode:
        errors.append(
            "privacy_mode=true is incompatible with provider=bruteforce (requires persisted crops)."
        )

    # Qdrant checks
    qdr = data.get("vector_database", {}).get("qdrant", {}) or {}
    if provider == "qdrant":
        # Allow embedded location OR network host/port; warn if neither appears resolvable.
        loc = qdr.get("location")
        host = qdr.get("host") or env.get("QDRANT_HOST")
        port = qdr.get("port") or env.get("QDRANT_PORT")
        if not loc and (not host or not port):
            warnings.append(
                "Qdrant provider selected but neither embedded 'location' nor network 'host/port' appear configured (YAML or env)."
            )

    # Postgres checks when saving enabled
    save_to_pg = bool(search_task.get("save_results_to_postgresql", False))
    if save_to_pg:
        pg = data.get("results_postgresql", {}) or {}
        # Accept YAML placeholders; ensure resolvable either via YAML scalar or env
        def _resolve(k: str) -> Optional[str]:
            val = pg.get(k)
            if isinstance(val, str) and val.strip():
                return val
            # Try env mapping
            mapping = {
                "database_name": "POSTGRES_DB",
                "host": "POSTGRES_HOST",
                "port": "POSTGRES_PORT",
                "user": "POSTGRES_USER",
                "password": "POSTGRES_PASSWORD",
            }.get(k)
            if mapping and env.get(mapping):
                return env[mapping]
            return None

        for key in ("database_name", "host", "port", "user", "password"):
            if not _resolve(key):
                errors.append(
                    f"results_postgresql.{key} is not resolvable via YAML or env (POSTGRES_*)."
                )

    return CheckResult(path, errors, warnings)


def validate_result_gui(path: Path, env: Dict[str, str]) -> CheckResult:
    """Validate Result GUI configuration.

    Rules:
      - If use_csv=true, csv_path must be provided.
      - If use_csv=false, DB connection must be resolvable via YAML or env, and results_table present.

    Args:
        path: Path to external/result_gui/config*.yaml.
        env: Merged environment mapping.

    Returns:
        CheckResult: Accumulated warnings and errors.
    """
    errors: List[str] = []
    warnings: List[str] = []

    try:
        data = _read_yaml(path) or {}
        if not isinstance(data, dict):
            return CheckResult(path, ["Top-level YAML must be a mapping"], [])
    except Exception as exc:
        return CheckResult(path, [f"Read error: {exc}"], [])

    use_csv = bool(data.get("use_csv", False))
    if use_csv:
        if not (isinstance(data.get("csv_path"), str) and str(data.get("csv_path")).strip()):
            errors.append("use_csv=true requires 'csv_path'.")
    else:
        # DB mode
        prefer_env = bool(data.get("user_environment_db_config", True))
        # Gather candidates from YAML or env
        def _choice(yaml_key: str, env_key: str) -> Optional[str]:
            yval = data.get(yaml_key)
            if isinstance(yval, str) and yval.strip():
                return yval
            return env.get(env_key)

        db_host = _choice("db_host", "POSTGRES_HOST")
        db_port = _choice("db_port", "POSTGRES_PORT")
        db_user = _choice("db_user", "POSTGRES_USER")
        db_password = _choice("db_password", "POSTGRES_PASSWORD")
        db_name = _choice("db_name", "POSTGRES_DB")
        results_table = data.get("results_table") or env.get("POSTGRES_TABLE_NAME")

        for name, val in (
            ("db_host", db_host),
            ("db_port", db_port),
            ("db_user", db_user),
            ("db_password", db_password),
            ("db_name", db_name),
        ):
            if not val:
                errors.append(
                    f"DB mode: '{name}' not resolvable via YAML or env (POSTGRES_*)."
                )
        if not results_table:
            errors.append(
                "DB mode: 'results_table' not provided in YAML and POSTGRES_TABLE_NAME not set."
            )

    return CheckResult(path, errors, warnings)


def validate_key_input(path: Path, env: Dict[str, str]) -> CheckResult:
    """Validate key_input configuration for orchestrated ingestion.

    Rules:
      - mode=api requires api_endpoint and a filename parameter mapping.
      - Recommend Authorization header via ${IMAGE_SIM_API_KEY}; warn if literal placeholder present.
      - mode=local should provide at least one search_roots entry.

    Args:
        path: Path to external/key_input/key_input_config*.yaml.
        env: Merged environment mapping.

    Returns:
        CheckResult: Accumulated warnings and errors.
    """
    errors: List[str] = []
    warnings: List[str] = []

    try:
        data = _read_yaml(path) or {}
        if not isinstance(data, dict):
            return CheckResult(path, ["Top-level YAML must be a mapping"], [])
    except Exception as exc:
        return CheckResult(path, [f"Read error: {exc}"], [])

    ds = data.get("data_source", {}) or {}
    mode = (ds.get("mode") or "api").lower()
    if mode == "api":
        api = ds.get("api", {}) or {}
        ep = api.get("api_endpoint")
        if not (isinstance(ep, str) and ep.strip()):
            errors.append("data_source.api.api_endpoint is required when mode=api.")
        req = api.get("request_mapping", {}) or {}
        param_map = req.get("param_map", {}) or {}
        fname_param = req.get("api_filename_param")
        file_name_col = (data.get("key_input", {}) or {}).get("file_name_column", "파일명")
        if not fname_param and file_name_col not in param_map:
            errors.append(
                "API mode: provide api_filename_param or include file_name_column in request_mapping.param_map."
            )
        headers = api.get("headers", {}) or {}
        auth = headers.get("Authorization")
        if isinstance(auth, str) and "REPLACE_WITH_TOKEN" in auth and not env.get("IMAGE_SIM_API_KEY"):
            warnings.append(
                "Authorization header contains a placeholder and IMAGE_SIM_API_KEY is not set."
            )
    elif mode == "local":
        local = ds.get("local", {}) or {}
        roots = local.get("search_roots", []) or []
        if not roots:
            warnings.append("Local mode: 'search_roots' is empty; no files will be found.")

    return CheckResult(path, errors, warnings)


# ------------------------------- Entrypoint ----------------------------------

def main() -> int:
    """Run configuration validations across supported modules.

    Returns:
        int: Exit status (0 pass with possible warnings, 1 on any error).
    """
    repo_root = Path(__file__).resolve().parents[1]
    env = _env_merged(repo_root)

    files: List[Tuple[Path, str]] = [
        # Core image similarity configs
        (repo_root / "hyundai_document_authenticator" / "configs" / "image_similarity_config.yaml", "image_similarity"),
        (repo_root / "hyundai_document_authenticator" / "configs" / "image_similarity_config_TEMPLATE.yaml", "image_similarity"),
        # External modules
        (repo_root / "hyundai_document_authenticator" / "external" / "result_gui" / "config.yaml", "result_gui"),
        (repo_root / "hyundai_document_authenticator" / "external" / "result_gui" / "config_minimal.yaml", "result_gui"),
        (repo_root / "hyundai_document_authenticator" / "external" / "key_input" / "key_input_config.yaml", "key_input"),
        (repo_root / "hyundai_document_authenticator" / "external" / "key_input" / "key_input_config_minimal.yaml", "key_input"),
    ]

    any_error = False
    for fpath, ftype in files:
        if not fpath.exists():
            print(f"SKIP (missing): {fpath}")
            continue
        if ftype == "image_similarity":
            res = validate_image_similarity(fpath, env)
        elif ftype == "result_gui":
            res = validate_result_gui(fpath, env)
        elif ftype == "key_input":
            res = validate_key_input(fpath, env)
        else:
            print(f"Unknown type for {fpath}, skipping.")
            continue

        # Report per file
        if res.errors:
            any_error = True
            print(f"ERRORS: {res.file}")
            for e in res.errors:
                print(f"  - {e}")
        if res.warnings:
            print(f"WARNINGS: {res.file}")
            for w in res.warnings:
                print(f"  - {w}")
        if not res.errors and not res.warnings:
            print(f"OK: {res.file}")

    return 1 if any_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
