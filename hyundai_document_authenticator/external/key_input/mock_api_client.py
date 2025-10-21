#!/usr/bin/env python
"""Mock API client for folder-backed TIFF server.

- Sends GET/POST requests to the mock API server
- Supports server-side save_to_folder toggle per request
- Accepts arbitrary extra key=value params from CLI
- Optional: build JSON payload from a key table row (CSV/XLSX/JSON) by filename
- Optional: decode returned base64 image and save to a file for verification

Behavior is CLI-oriented. User-facing prints are preserved. Internal logging is
available via the module logger for library-style reuse.

Features
- Send GET/POST requests to the mock API server
- Support save_to_folder toggling per request
- Add arbitrary extra parameters from CLI
- (Optional) Build payload from a key table row (CSV/XLSX/JSON) by filename
- Decode the returned base64 image and save to a file for verification

Usage
  pip install requests pandas openpyxl pillow

  # Minimal: POST with filename, asking server to re-save the file
  python mock_api_client.py --endpoint http://127.0.0.1:5001/images --method POST \
      --filename N20231004THA00001 --save-to-folder

  # Include extra fields and save the response image to a local file
  python mock_api_client.py --endpoint http://127.0.0.1:5001/images --method POST \
      --filename N20231004THA00001 \
      --extra comany_registration=1234567890 --extra reception_date=2024-08-15 --extra company_name=업종ABC \
      --save-to-folder ./response_copy.tif

  # Auto-payload from key table (look up row by filename)
  python mock_api_client.py --endpoint http://127.0.0.1:5001/images --method POST \
      --filename N20231004THA00001 --key-table "D:/real_data_key/filtered_rows.xlsx" \
      --file-name-column "파일명" --map comany_registration=사업자등록번호 --map reception_date=수신일자 --map company_name=업종명

  # Batch mode: send requests for all filenames in a key table
  python mock_api_client.py --endpoint http://127.0.0.1:5001/images --method POST \
      --key-table "D:/real_data_key/filtered_rows.xlsx" --file-name-column "파일명" \
      --map comany_registration=사업자등록번호 --map reception_date=수신일자 --map company_name=업종명

Notes
- The server uses 'filename' for matching; other params are echoed and metadata is pulled from the key table row by filename when available.
- --save-to-folder: If used as a flag, asks the server to re-save the file. If given a path, saves the response image to that path on the client.
- Batch mode is enabled by providing --key-table without --filename.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    import pandas as pd
except Exception:  # pragma: no cover - environment may lack pandas
    pd = None  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover - environment may lack Pillow
    Image = None  # type: ignore

# Module-level logger (handlers set by invoking CLI or parent app)
logger = logging.getLogger(__name__)


def _parse_kv_list(items: Optional[List[str]]) -> Dict[str, str]:
    """Parse a list of key=value strings into a dict.

    Args:
        items: List of strings in the form "key=value".

    Returns:
        Mapping where keys and values are stripped strings. Invalid entries are ignored.
    """
    out: Dict[str, str] = {}
    if not items:
        return out
    for item in items:
        if "=" not in item:
            logger.debug("Ignoring malformed key/value pair: %s", item)
            continue
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _load_key_table(key_table: Path) -> List[Dict[str, Any]]:
    """Load key rows from CSV/XLSX/JSON array into list-of-dicts.

    Raises FileNotFoundError for missing path; ValueError for unsupported format.
    """
    if not key_table.exists():
        raise FileNotFoundError(f"Key table not found: {key_table}")
    ext = key_table.suffix.lower()
    if ext in (".csv", ".tsv"):
        if pd is None:
            import csv
            with key_table.open("r", encoding="utf-8-sig", newline="") as f:
                return list(csv.DictReader(f))
        return pd.read_csv(key_table).to_dict(orient="records")
    if ext in (".xlsx", ".xls"):
        if pd is None:
            raise RuntimeError("pandas required for Excel key-table usage. Install pandas/openpyxl.")
        return pd.read_excel(key_table).to_dict(orient="records")
    try:
        data = json.loads(key_table.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception as e:
        raise ValueError(f"Unsupported key table format: {ext}") from e
    return []


def _build_payload_from_row(
    row: Dict[str, Any],
    filename: str,
    map_req_to_col: Dict[str, str],
) -> Dict[str, Any]:
    """Build request payload by mapping request keys to columns from a row."""
    payload: Dict[str, Any] = {"filename": filename}
    for req_k, col in map_req_to_col.items():
        if req_k == "filename":
            continue
        payload[req_k] = row.get(col)
    return payload


def _save_b64_to_file(b64_str: str, out_path: Path) -> None:
    """Decode base64 string and write an image to the specified path.

    Attempts to use Pillow for transcoding; falls back to raw bytes if Pillow is absent.
    """
    try:
        raw = base64.b64decode(b64_str)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if Image is not None:
            import io
            img = Image.open(io.BytesIO(raw))
            img.save(out_path, format="TIFF")
        else:
            with out_path.open("wb") as f:
                f.write(raw)
        print(f"Saved response image to: {out_path.resolve()}")
    except Exception as e:
        logger.warning("Failed to write response image: %s", e)
        print(f"Warning: Failed to write response image: {e}")


def main() -> None:
    """Entry point for the mock API client CLI."""
    parser = argparse.ArgumentParser(description="Mock API Client for TIFF retrieval")
    parser.add_argument("--endpoint", required=True, help="Mock API endpoint, e.g., http://127.0.0.1:5001/images")
    parser.add_argument("--method", default="POST", choices=["GET", "POST"], help="HTTP method")

    # Core request inputs
    parser.add_argument("--filename", default=None, help="Filename for single request. Omit for batch mode.")
    parser.add_argument(
        "--save-to-folder",
        nargs="?",
        const=True,
        default=None,
        help="If used as a flag, asks server to re-save. If a path is provided, saves response image to that path.",
    )
    parser.add_argument("--extra", action="append", help="Extra request params as key=value (repeatable)")

    # Build from key table
    parser.add_argument("--key-table", type=Path, default=None, help="Path to key table for batch mode or single payload building.")
    parser.add_argument("--file-name-column", default="파일명", help="Column name in key table for filenames")
    parser.add_argument("--map", action="append", help="Map request param to key table column (e.g., comany_registration=사업자등록번호)")
    args = parser.parse_args()

    if args.filename is None and args.key_table is None:
        parser.error("Either --filename or --key-table must be provided.")

    # Determine mode: single request or batch
    is_batch_mode = args.filename is None and args.key_table is not None

    map_req_to_col = _parse_kv_list(args.map)
    extra_payload = _parse_kv_list(args.extra)

    # Client-side save logic
    client_save_path: Optional[Path] = None
    save_arg = args.save_to_folder
    if save_arg is not None and save_arg is not True:
        try:
            client_save_path = Path(save_arg)
        except Exception:
            print(f"Invalid path for --save-to-folder: {save_arg}", file=sys.stderr)
            sys.exit(1)

    def process_request(payload: Dict[str, Any], request_num: int = 0) -> None:
        """Send the request and optionally save the returned image."""
        # Add extras and server-side save flag
        final_payload = payload.copy()
        final_payload.update(extra_payload)
        if save_arg is True:
            final_payload["save_to_folder"] = "true"

        prefix = f"[Request {request_num}] " if is_batch_mode else ""
        print(f"\n{prefix}Sending request for filename: {payload['filename']}")

        try:
            if args.method.upper() == "POST":
                resp = requests.post(args.endpoint, json=final_payload, timeout=30)
            else:
                resp = requests.get(args.endpoint, params=final_payload, timeout=30)
            resp.raise_for_status()
            data: Dict[str, Any] = resp.json()

            print(f"\n{prefix}=== Server Response (JSON) ===")
            try:
                print(json.dumps(data, indent=2, ensure_ascii=False))
            except Exception:
                print(data)

            # Client-side save for this request
            if client_save_path:
                b64 = data.get("image_b64") if isinstance(data, dict) else None
                if b64:
                    # In batch mode, append filename to save path to avoid overwriting
                    save_name: Path = client_save_path / f"{payload['filename']}.tif" if is_batch_mode and client_save_path.is_dir() else client_save_path
                    _save_b64_to_file(b64, save_name)
                else:
                    print(f"{prefix}No image_b64 in response; skipping save.")

        except Exception as e:
            logger.error("Request failed: %s", e)
            print(f"{prefix}Request failed: {e}", file=sys.stderr)
            try:
                print(f"{prefix}Server response: {resp.text}", file=sys.stderr)  # type: ignore[name-defined]
            except Exception:
                pass

    if is_batch_mode:
        # Batch mode
        rows = _load_key_table(args.key_table)  # type: ignore[arg-type]
        for i, row in enumerate(rows):
            filename = str(row.get(args.file_name_column, "")).strip()
            if not filename:
                continue
            payload = _build_payload_from_row(row, filename, map_req_to_col)
            process_request(payload, i + 1)
    else:
        # Single request mode
        payload: Dict[str, Any] = {"filename": args.filename}
        if args.key_table:
            rows = _load_key_table(args.key_table)
            match_row = next((r for r in rows if str(r.get(args.file_name_column, "")).strip() == args.filename), None)
            if match_row:
                payload = _build_payload_from_row(match_row, args.filename, map_req_to_col)
        process_request(payload)


if __name__ == "__main__":
    main()
