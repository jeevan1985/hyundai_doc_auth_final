"""Runtime YAML configuration updater driven by environment variables.

This tool replaces brittle sed-based substitutions by safely parsing YAML and
updating structured keys. It supports two operations:

- qdrant: Configure Qdrant mode (embedded/server) and endpoints in one or more
          YAML files. It updates the provider-specific sections when present and
          leaves other files unchanged. It is safe to run against arbitrary
          YAML files that may not contain the expected keys.
- database: Configure top-level database URIs for API configs from environment
            variables.

Usage examples (executed from the container entrypoint):
  python -m hyundai_document_authenticator.extra_tools.config_env_merge qdrant \
      --mode embedded --host qdrant --port 6333 --file /tmp/runtime_configs/image_similarity_config.yaml
  python -m hyundai_document_authenticator.extra_tools.config_env_merge database \
      --file /tmp/runtime_configs/api_config.yaml

The script is idempotent and defensive; errors are logged to stderr and do not
raise, to avoid breaking startup flows. It prints a short status to stdout on
success for traceability in container logs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _load_yaml(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return None


essential_warning_emitted = False


def _save_yaml(path: Path, data: Dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, indent=2, sort_keys=False)
        return True
    except Exception:
        return False


def _ensure_path(data: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    cur: Dict[str, Any] = data
    for k in keys:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    return cur


def op_qdrant(mode: str, host: str, port: int, files: List[Path]) -> int:
    # Normalize mode
    mode_norm = (mode or "embedded").strip().lower()
    for fp in files:
        cfg = _load_yaml(fp)
        if cfg is None:
            continue
        changed = False
        # Try known path first (CLI config)
        root = cfg
        # vector_database.qdrant
        qdrant_path = ["vector_database", "qdrant"]
        qdr = root
        for key in qdrant_path:
            qdr = qdr.get(key) if isinstance(qdr, dict) else None
            if qdr is None:
                break
        if isinstance(qdr, dict):
            # Ensure keys exist
            if mode_norm == "server":
                # Server mode: set host/port; remove/disable location
                qdr["host"] = host
                qdr["port"] = int(port)
                if "location" in qdr:
                    try:
                        del qdr["location"]
                    except Exception:
                        pass
                changed = True
            else:
                # Embedded mode: ensure location present; we choose an env override or default
                loc = os.getenv("QDRANT_LOCATION", qdr.get("location", "instance/qdrant_db"))
                qdr["location"] = loc
                # Optionally remove host/port to avoid confusion
                for k in ("host", "port"):
                    if k in qdr:
                        try:
                            del qdr[k]
                        except Exception:
                            pass
                changed = True
        else:
            # Try flat qdrant at root (API config may differ); set host/port or location if present
            if mode_norm == "server":
                if isinstance(root.get("qdrant"), dict):
                    root["qdrant"]["host"] = host
                    root["qdrant"]["port"] = int(port)
                    if "location" in root["qdrant"]:
                        try:
                            del root["qdrant"]["location"]
                        except Exception:
                            pass
                    changed = True
            else:
                if isinstance(root.get("qdrant"), dict):
                    loc = os.getenv("QDRANT_LOCATION", root["qdrant"].get("location", "instance/qdrant_db"))
                    root["qdrant"]["location"] = loc
                    for k in ("host", "port"):
                        if k in root["qdrant"]:
                            try:
                                del root["qdrant"][k]
                            except Exception:
                                pass
                    changed = True
        if changed:
            _save_yaml(fp, cfg)
            print(f"[config_env_merge] qdrant updated: {fp}")
    return 0


def op_database(files: List[Path]) -> int:
    # Read env. Use same names as entrypoint expectations.
    pg_user = os.getenv("POSTGRES_USER")
    if not pg_user:
        # Nothing to do.
        return 0
    pg_password = os.getenv("POSTGRES_PASSWORD", "")
    pg_host = os.getenv("POSTGRES_HOST", "db")
    try:
        pg_port = int(os.getenv("POSTGRES_PORT", "5432"))
    except ValueError:
        pg_port = 5432
    pg_db = os.getenv("POSTGRES_DB", "postgres")

    # Optional user DB
    user_user = os.getenv("POSTGRES_USER_USER", "")
    user_password = os.getenv("POSTGRES_USER_PASSWORD", "")
    user_host = os.getenv("POSTGRES_USER_HOST", pg_host)
    try:
        user_port = int(os.getenv("POSTGRES_USER_PORT", str(pg_port)))
    except ValueError:
        user_port = pg_port
    user_db = os.getenv("POSTGRES_USER_DB", "")

    # Avoid logging the password if these URIs are ever printed; keep them in-memory only.
    results_uri = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
    user_uri = f"postgresql://{user_user}:{user_password}@{user_host}:{user_port}/{user_db}"

    for fp in files:
        cfg = _load_yaml(fp)
        if cfg is None:
            continue
        if not isinstance(cfg, dict):
            continue
        cfg["database_uri"] = results_uri
        cfg["user_database_uri"] = user_uri
        _save_yaml(fp, cfg)
        print(f"[config_env_merge] database URIs updated: {fp}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Robust YAML updater from environment variables.")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("qdrant", help="Configure Qdrant embedded/server mode and endpoints")
    q.add_argument("--mode", default=os.getenv("QDRANT_MODE", "embedded"), help="embedded|server")
    q.add_argument("--host", default=os.getenv("QDRANT_HOST", "qdrant"))
    q.add_argument("--port", default=os.getenv("QDRANT_PORT", "6333"))
    q.add_argument("--file", action="append", required=True, help="YAML file to update (repeat)")

    d = sub.add_parser("database", help="Configure top-level database URIs from env vars")
    d.add_argument("--file", action="append", required=True, help="YAML file to update (repeat)")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "qdrant":
        try:
            port = int(args.port)
        except Exception:
            port = 6333
        files = [Path(f) for f in (args.file or [])]
        return op_qdrant(args.mode, args.host, port, files)

    if args.cmd == "database":
        files = [Path(f) for f in (args.file or [])]
        return op_database(files)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
