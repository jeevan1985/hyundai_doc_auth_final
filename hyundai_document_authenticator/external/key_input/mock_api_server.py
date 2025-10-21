#!/usr/bin/env python
"""Folder-backed mock API server for TIFF retrieval.

- Loads a key table (CSV/XLSX/JSON) with columns like 사업자등록번호, 수신일자, 파일명, 업종명
- Exposes an HTTP endpoint that accepts parameters such as comany_registration,
  reception_date, filename, comapany_name, and save_to_folder
- Reuses LocalFolderFetcher matching logic to find a TIF under configured search roots
- Returns JSON with the base64-encoded TIF

# What it does
# - Loads a key table (CSV/XLSX) containing columns like 사업자등록번호, 수신일자, 파일명, 업종명
# - Exposes an HTTP endpoint that accepts parameters:
#     comany_registration (maps to 사업자등록번호)
#     reception_date      (maps to 수신일자)
#     filename            (maps to 파일명)
#     comapany_name       (maps to 업종명)
# - Reuses the same filename matching logic as data_source.local (NameMapping + rglob) to find a TIF
#   under the provided search root folder (e.g., D:\\test_data).
# - Returns JSON with base64 of the matched TIF as {"image_b64": "...", "server_filename": "..."}.
#
# Usage
#   pip install flask pandas openpyxl pillow
#   python mock_api_server.py \
#     --key-table "D:/real_data_key/filtered_rows.xlsx" \
#     --search-root "D:/test_data" \
#     --tail-len 5 [--insert-token 001] --glob-suffix "_*.tif" --any-depth [--save-by-default] [--debug]
#
#   For debugging filename matching, add --debug to see verbose matching logs.
#
# ## one liner
#  python mock_api_server.py --key-table "D:/real_data_key/sample_data_keys.xlsx" --search-root "D:\real_data_key\images_test_key" --tail-len 5 --glob-suffix "_*.tif" --any-depth

CLI usage is preserved; internal logging is added for diagnostics.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, request, jsonify

# Module logger with centralized $APP_LOG_DIR/tools fallback
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_ch = logging.StreamHandler()
_ch.setFormatter(_fmt)
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(_ch)
try:
    env_log_dir = os.getenv("APP_LOG_DIR")
    if env_log_dir:
        tools_dir = Path(env_log_dir) / "tools"
        tools_dir.mkdir(parents=True, exist_ok=True)
        _fh = logging.FileHandler(str(tools_dir / "mock_api_server.log"), encoding="utf-8")
        _fh.setFormatter(_fmt)
        logger.addHandler(_fh)
except Exception:
    pass

# Ensure we can import orchestrator-side classes for matching logic
try:
    THIS_DIR = Path(__file__).resolve().parent
    PARENT_DIR = THIS_DIR.parent
    if str(PARENT_DIR) not in os.sys.path:
        os.sys.path.append(str(PARENT_DIR))
    from key_input_orchestrator import LocalFolderFetcher, LocalFetchConfig, NameMappingConfig
except Exception as e:  # pragma: no cover - environment variance
    logger.warning("Local orchestrator imports failed: %s", e)
    LocalFolderFetcher = None  # type: ignore
    LocalFetchConfig = None  # type: ignore
    NameMappingConfig = None  # type: ignore

try:
    import pandas as pd  # For reading Excel/CSV
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    from PIL import Image  # Not strictly required; only used if we need to transcode
except Exception:  # pragma: no cover
    Image = None  # type: ignore

app = Flask(__name__)


# ------------------------------
# Helpers
# ------------------------------

def load_key_table_rows(path: Path, file_name_column: str) -> List[Dict[str, Any]]:
    """Load key rows from CSV/XLSX/JSON into list of dicts."""
    if not path.exists():
        raise FileNotFoundError(f"Key table not found: {path}")
    ext = path.suffix.lower()
    if ext in (".csv", ".tsv"):
        if pd is None:
            # Minimal CSV reader fallback
            import csv
            rows: List[Dict[str, Any]] = []
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
            return rows
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    if ext in (".xlsx", ".xls"):
        if pd is None:
            raise RuntimeError("pandas is required to read Excel. Install pandas/openpyxl.")
        df = pd.read_excel(path)
        return df.to_dict(orient="records")
    # Try JSON array
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
    except Exception:
        pass
    raise ValueError(f"Unsupported key table format: {ext}")


def build_fetcher(
    search_roots: List[Path],
    recursive: bool,
    allowed_exts: Tuple[str, ...],
    resolve_without_extension: bool,
    tail_len: int,
    insert_token: Optional[str],
    glob_suffix: str,
    use_rglob_any_depth: bool,
    debug: bool = False,
) -> LocalFolderFetcher:
    """Construct a LocalFolderFetcher using NameMappingConfig for filename heuristics."""
    if LocalFolderFetcher is None:
        raise RuntimeError("LocalFolderFetcher could not be imported. Check repository structure.")
    lf_cfg = LocalFetchConfig(
        search_roots=search_roots,
        recursive=recursive,
        allowed_extensions=allowed_exts,
        resolve_without_extension=resolve_without_extension,
        case_insensitive_match=True,
        stop_on_first_match=True,
    )
    nm_cfg = NameMappingConfig(
        enabled=bool(insert_token),
        debug_log=debug,
        tail_len=tail_len,
        insert_token=str(insert_token) if insert_token else "",
        glob_suffix=glob_suffix,
        use_rglob_any_depth=use_rglob_any_depth,
        db_like_template="{prefix}{insert}{suffix}_%.tif",
    )
    return LocalFolderFetcher(lf_cfg, None, name_map=nm_cfg)


def match_row(rows: List[Dict[str, Any]], col_map: Dict[str, str], req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Find first row where all provided request param matches the mapped column values (string compare)."""
    filters: Dict[str, Tuple[str, str]] = {}
    for req_key, col_name in col_map.items():
        val = req.get(req_key)
        if val is not None and val != "":
            filters[req_key] = (col_name, str(val))
    if not filters:
        return rows[0] if rows else None

    for row in rows:
        ok = True
        for _req_key, (col_name, expected) in filters.items():
            actual = str(row.get(col_name, ""))
            if actual != expected:
                ok = False
                break
        if ok:
            return row
    return None


def find_tif_for_filename(fetcher: LocalFolderFetcher, filename: str) -> Optional[Path]:
    """Return first matched TIF path for a filename using the fetcher."""
    paths = fetcher.fetch_batch([filename])
    return paths[0] if paths else None


def file_to_base64(p: Path) -> str:
    """Read file bytes and return base64-encoded ASCII string."""
    with p.open('rb') as f:
        raw = f.read()
    return base64.b64encode(raw).decode('ascii')


# ------------------------------
# Flask endpoint
# ------------------------------

def create_app(config: Dict[str, Any]) -> Flask:
    """Attach config to the Flask app and return the app instance."""
    app.config['MOCK_CFG'] = config
    return app


@app.route('/images', methods=['GET', 'POST'])
def images() -> tuple[object, int]:
    """Return a base64-encoded TIF for a matched filename with optional on-disk save.

    Returns:
        tuple[object, int]: (Flask JSON response, HTTP status)
    """
    cfg = app.config.get('MOCK_CFG') or {}
    # Collect request payload (JSON for POST; query args for GET)
    if request.method == 'POST':
        payload = request.get_json(force=True, silent=True) or {}
    else:
        payload = dict(request.args or {})

    # Determine filename to match (required). Other params are not used for matching.
    filename_value = str(payload.get('filename') or '').strip()
    if not filename_value:
        return jsonify({'error': 'Filename is required in request (param: filename).', 'echo_request': payload}), 400

    # Locate TIF using same matching logic as local mode
    p = find_tif_for_filename(cfg['fetcher'], filename_value)
    if not p:
        error_msg = f'No matching TIF found under search roots for filename={filename_value}'
        try:
            fetcher = cfg['fetcher']
            nm_cfg = fetcher.name_map
            if nm_cfg.enabled:  # type: ignore[attr-defined]
                # Accessing private method to get mapped name for better error
                mapped_name = fetcher._mapped_core(filename_value)  # type: ignore[attr-defined]
                if mapped_name:
                    pattern = mapped_name + nm_cfg.glob_suffix  # type: ignore[operator]
                    error_msg += f" (searched for patterns like '{pattern}')"
        except Exception:  # pragma: no cover - best-effort diagnostics
            pass
        return jsonify({'error': error_msg, 'echo_request': payload}), 404

    # Lookup metadata row in key table by filename (if available)
    row = None
    try:
        for _r in cfg['rows']:
            if str(_r.get(cfg['col_filename'], '')).strip() == filename_value:
                row = _r
                break
    except Exception:
        row = None

    # Base64 encode the matched TIF
    b64_str = file_to_base64(p)

    # Optional: save the base64 back to a TIFF on disk with the same name
    def _to_bool(v: Any) -> bool:
        """Interpret diverse truthy/falsey values as boolean.

        Accepts common string and numeric representations.
        """
        if isinstance(v, bool):
            return v
        if v is None:
            return False
        s = str(v).strip().lower()
        return s in ("1", "true", "yes", "y", "on")

    # Determine whether to save: server default, overridden by per-request param if present
    want_save = bool(cfg.get('default_save_to_folder', False))
    if 'save_to_folder' in payload:
        want_save = _to_bool(payload.get('save_to_folder'))

    saved_flag = False
    if want_save:
        try:
            save_dir_override = cfg.get('save_folder_location')
            if save_dir_override:
                save_dir = save_dir_override
            else:
                # Default to a 'searched' subfolder
                save_dir = p.parent / 'searched'
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / p.name

            raw = base64.b64decode(b64_str)
            if Image is not None:
                bio = io.BytesIO(raw)
                img = Image.open(bio)
                img.save(save_path, format='TIFF')
            else:
                with save_path.open('wb') as f:
                    f.write(raw)
            saved_flag = True
        except Exception as e:  # pragma: no cover - path and IO variability
            logger.warning("Failed to save file for '%s': %s", p.name, e)
            print(f"Error saving file: {e}")
            saved_flag = False

    # Build response
    resp = {
        'server_filename': p.name,
        'image_b64': b64_str,
        'saved_to_folder': saved_flag,
        'echo_request': payload,
        'matched_row_excerpt': {
            cfg['col_business_no']: (row.get(cfg['col_business_no']) if row else None),
            cfg['col_received_date']: (row.get(cfg['col_received_date']) if row else None),
            cfg['col_filename']: filename_value,
            cfg['col_business_name']: (row.get(cfg['col_business_name']) if row else None),
        }
    }
    return jsonify(resp), 200


# ------------------------------
# Entry
# ------------------------------

def main() -> None:
    """Run the Flask app with configured matching/search behavior."""
    parser = argparse.ArgumentParser(description='Folder-backed mock API server for TIFF retrieval')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--key-table', required=True, help='Path to key table (CSV/XLSX/JSON array)')
    parser.add_argument('--file-name-column', default='파일명')
    parser.add_argument('--search-root', action='append', required=True, help='Folder(s) to search for TIF/TIFF (repeatable)')
    parser.add_argument('--recursive', action='store_true', default=True)
    parser.add_argument('--resolve-without-extension', action='store_true', default=True)
    parser.add_argument('--tail-len', type=int, default=5)
    parser.add_argument('--insert-token', default=None, help='Optional name-mapping token. If omitted, no token insertion is applied (pipeline mapping remains in key_input).')
    parser.add_argument('--glob-suffix', default='_*.tif')
    parser.add_argument('--any-depth', action='store_true', default=True)
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug logging for filename matching.')
    parser.add_argument('--save-by-default', action='store_true', default=False,
                        help="If set, the server will save matched TIFFs to a 'searched' subfolder by default unless overridden by per-request save_to_folder=false.")
    parser.add_argument('--save-folder-location', type=Path, default=None,
                        help="If provided, save all TIFFs to this directory instead of the default 'searched' subfolder.")

    # Optional column names for other fields (allow override)
    parser.add_argument('--col-business-no', default='사업자등록번호')
    parser.add_argument('--col-received-date', default='수신일자')
    parser.add_argument('--col-business-name', default='업종명')

    args = parser.parse_args()

    # Configure logging so LocalFolderFetcher info logs are visible when --debug
    log_level = logging.INFO if args.debug else logging.WARNING
    logging.getLogger().setLevel(log_level)
    try:
        app.logger.setLevel(log_level)
    except Exception:
        pass

    # Fix for user passing escaped glob like \*
    glob_suffix = str(args.glob_suffix).replace('\\*', '*')

    key_table_path = Path(args.key_table).resolve()
    rows = load_key_table_rows(key_table_path, args.file_name_column)

    search_roots = [Path(p).resolve() for p in args.search_root]
    fetcher = build_fetcher(
        search_roots=search_roots,
        recursive=bool(args.recursive),
        allowed_exts=(".tif", ".tiff"),
        resolve_without_extension=bool(args.resolve_without_extension),
        tail_len=int(args.tail_len),
        insert_token=args.insert_token,
        glob_suffix=glob_suffix,
        use_rglob_any_depth=bool(args.any_depth),
        debug=bool(args.debug),
    )

    cfg: Dict[str, Any] = {
        'rows': rows,
        'fetcher': fetcher,
        'col_filename': args.file_name_column,
        'col_business_no': args.col_business_no,
        'col_received_date': args.col_received_date,
        'col_business_name': args.col_business_name,
        'default_save_to_folder': bool(args.save_by_default),
        'save_folder_location': args.save_folder_location,
    }

    create_app(cfg)
    print(f"Starting folder-backed mock API on http://{args.host}:{args.port}")
    print(f"Key table: {key_table_path}")
    for r in search_roots:
        print(f"Search root: {r}")
    print(f"Default save_to_folder: {'enabled' if args.save_by_default else 'disabled'} (can be overridden per request)")
    print("Endpoint: POST/GET /images (params: comany_registration, reception_date, filename, comapany_name, save_to_folder)")

    app.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
