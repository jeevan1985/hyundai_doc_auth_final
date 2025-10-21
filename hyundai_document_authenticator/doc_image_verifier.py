#!/usr/bin/env python
"""
TIF Document Similarity System - Local CLI (TIF-focused)

This CLI exposes ONLY the TIF workflows required by your request:
  - search-with-tif: TIF batch search (photo extraction + image similarity + doc aggregation)
  - build-index-with-tif: Build an image index from photos extracted from TIF documents

Examples
  # Build index from TIFs by extracting photos
  python doc_image_verifier.py build-image-index --folder ./data_real --photo-extraction-mode bbox \
      --bbox-list "[[171,236,1480,1100],[171,1168,1480,2032]]" --bbox-format xyxy

  # TIF batch search
  python doc_image_verifier.py search-doc --folder ./data_real --top-doc 7 --top-k 5 --aggregation-strategy max \
      --photo-extraction-mode yolo --yolo-model-path trained_model/weights/best.pt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import typer
import yaml
import tempfile
from dotenv import load_dotenv, find_dotenv

# Ensure core_engine/external import paths
sys.path.append(str(Path(__file__).resolve().parent))
external_dir = Path(__file__).resolve().parent / "external"
if external_dir.is_dir():
    sys.path.append(str(external_dir))

# Optional key-driven pipeline orchestrator (keeps backward compatibility if absent)
try:
    from external.key_input.key_input_orchestrator import run_key_input_pipeline, run_key_input_pipeline_for_index
except Exception:
    run_key_input_pipeline = None  # type: ignore
    run_key_input_pipeline_for_index = None  # type: ignore

try:
    from core_engine.image_similarity_system.workflow import (
        build_index_from_tif_folder_workflow,
        execute_tif_batch_search_workflow,
    )
    from core_engine.image_similarity_system.config_loader import load_and_merge_configs
except ImportError as e:
    print(
        "Error: Failed to import necessary modules. Ensure project structure and dependencies are correct.",
        file=sys.stderr,
    )
    print(f"Details: {e}", file=sys.stderr)
    sys.exit(1)

app: typer.Typer = typer.Typer(add_completion=False, help="TIF Document Similarity System - Local CLI")


# -----------------
# Helper functions
# -----------------

def _bool_opt(value: Optional[bool]) -> Optional[bool]:
    """Identity helper for optional boolean CLI flags.

    Some Typer click-style flags pass None when not provided; this helper allows
    direct passthrough while making intent explicit and keeping audit tools happy.

    Args:
        value: Optional boolean flag value.

    Returns:
        The same value unmodified.
    """
    return value


def _json_list(value: Optional[str]) -> Optional[List[str]]:
    """Parse a JSON-encoded list string into a Python list of strings.

    Args:
        value: JSON string representing a list (e.g., '["a", "b"]'). If None or
            empty, returns None.

    Returns:
        Optional[List[str]]: Parsed list on success; None on failure.
    """
    if not value:
        return None
    try:
        data = json.loads(value)
        if isinstance(data, list):
            return data
    except Exception as _e:
        print(f"Warning: Could not parse JSON list: {_e}. Ignoring.", file=sys.stderr)
    return None


def _load_project_dotenv(start: Optional[Path] = None, filename: str = ".env") -> Optional[Path]:
    """Locate and load the project-level .env file.

    This attempts to locate a .env file in a robust, execution-mode-agnostic way:
    1) Search upward from the current working directory using python-dotenv's ``find_dotenv``.
    2) Fall back to the repository's root, computed relative to this file (one directory above the
       ``hyundai_document_authenticator/`` source package).

    Args:
        start: Optional starting directory for the search. If None, the current working directory
            is used implicitly by ``find_dotenv``.
        filename: Name of the environment file to search for (default: ".env").

    Returns:
        Optional[Path]: The resolved path to the loaded .env file if one was found and loaded;
        otherwise ``None``.

    Raises:
        None: This function is defensive; errors are swallowed to avoid impacting CLI execution.
    """
    try:
        # Step 1: search from current working directory upward
        path_str: str = find_dotenv(filename=filename, usecwd=True)
        if path_str:
            load_dotenv(dotenv_path=path_str, override=False)
            return Path(path_str).resolve()

        # Step 2: explicit fallback to repository root relative to this file
        repo_root = Path(__file__).resolve().parents[1]
        candidate = repo_root / filename
        if candidate.exists():
            load_dotenv(dotenv_path=str(candidate), override=False)
            return candidate.resolve()
    except Exception:
        # Intentional no-op: env loading must not break CLI execution
        return None
    return None


# Load environment variables early so downstream components see them.
try:
    _ = _load_project_dotenv()
except Exception:
    # Do not fail if dotenv is misconfigured or missing
    pass

# Initialize application logging (console + rotating file) so that warnings/errors are visible.
try:
    from core_engine.image_similarity_system.log_utils import setup_logging  # type: ignore
    setup_logging()  # default folder/name/level; respects transformers/httpx integrations
except Exception:
    # Logging configuration should never break CLI startup; fall back silently.
    pass

# Defensive import of shared log maintenance utility
try:
    from core_engine.image_similarity_system.utils import maintain_log_files  # type: ignore
except Exception:
    maintain_log_files: Optional[Callable[..., int]] = None  # type: ignore


# --------------
# TIF Sub-commands
# --------------

@app.command("search-doc")
def cmd_search_with_tif(
    config_path: Path = typer.Option(Path("configs/image_similarity_config.yaml"), help="Path to YAML config."),
    doc_input_start: Optional[str] = typer.Option(None, "--doc-input-start", help="Override input_mode.doc_input_start: tif|key"),
    folder: Optional[Path] = typer.Option(None, "--folder", help="Folder of TIF files for server-side batch search (overrides config)."),
    # Similarity search behavior
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Per-photo neighbor count."),
    top_doc: Optional[int] = typer.Option(None, "--top-doc", help="Number of docs per TIF to return."),
    aggregation_strategy: Optional[str] = typer.Option(None, "--aggregation-strategy", help="max|sum|mean"),
    bruteforce_db_folder: Optional[Path] = typer.Option(None, "--bruteforce-db-folder", help="Fallback DB folder."),

    # Output/persistence
    output_folder_for_results: Optional[Path] = typer.Option(None, "--output-folder", help="Output folder for results."),
    copy_parent_tif_docs_to_output: Optional[bool] = typer.Option(None, "--copy-parent-tif-docs/--no-copy-parent-tif-docs", help="Copy ranked parent TIF documents into the run output."),
    generate_tif_previews: Optional[bool] = typer.Option(None, "--generate-tif-previews/--no-generate-tif-previews"),
    create_per_query_subfolders_for_tif: Optional[bool] = typer.Option(None, "--per-query-subfolders/--no-per-query-subfolders"),
    save_results_to_postgresql: Optional[bool] = typer.Option(None, "--save-to-postgres/--no-save-to-postgres"),
    doc_sim_img_check: Optional[bool] = typer.Option(None, "--doc-sim-img-check/--no-doc-sim-img-check"),
    doc_sim_img_check_max_k: Optional[int] = typer.Option(None, "--doc-sim-img-check-max-k"),

    # Photo extraction
    photo_extraction_mode: Optional[str] = typer.Option(None, "--photo-extraction-mode", help="yolo|bbox"),
    photo_extractor_debug: Optional[bool] = typer.Option(None, "--photo-extractor-debug/--no-photo-extractor-debug"),
    yolo_model_path: Optional[str] = typer.Option(None, "--yolo-model-path"),
    yolo_confidence_threshold: Optional[float] = typer.Option(None, "--yolo-conf-thresh"),
    yolo_iou_threshold: Optional[float] = typer.Option(None, "--yolo-iou-thresh"),
    yolo_imgsz: Optional[int] = typer.Option(None, "--yolo-imgsz"),
    yolo_target_object_names: Optional[str] = typer.Option(None, "--yolo-target-object-names", help='JSON list, e.g. ["photo"]'),
    bbox_list_json: Optional[str] = typer.Option(None, "--bbox-list", help='JSON list, e.g. [[171,236,1480,1100]]'),
    bbox_format: Optional[str] = typer.Option(None, "--bbox-format", help="xyxy|xywh|cxcywh"),
    bbox_normalized: Optional[bool] = typer.Option(None, "--bbox-normalized/--no-bbox-normalized"),

    # Optional OCR/text search config pass-through for TIF workflows
    search_text: Optional[str] = typer.Option(None, "--search-text"),
    language: Optional[str] = typer.Option(None, "--language"),
    ocr_backend: Optional[str] = typer.Option(None, "--ocr-backend"),
    search_mode: Optional[str] = typer.Option(None, "--search-mode"),
    allow_normalization: Optional[bool] = typer.Option(None, "--allow-normalization/--no-allow-normalization"),
    remove_spaces_in_normalization: Optional[bool] = typer.Option(None, "--remove-spaces/--no-remove-spaces"),
    recognized_text_debug: Optional[bool] = typer.Option(None, "--recognized-text-debug/--no-recognized-text-debug"),
    search_location_top: Optional[float] = typer.Option(None, "--search-location-top"),
    use_offline_models: Optional[bool] = typer.Option(None, "--use-offline-models/--no-use-offline-models"),
    use_angle_cls: Optional[bool] = typer.Option(None, "--use-angle-cls/--no-use-angle-cls"),
    use_gpu_for_paddle: Optional[bool] = typer.Option(None, "--use-gpu-for-paddle/--no-use-gpu-for-paddle"),
    paddle_batch_size: Optional[int] = typer.Option(None, "--paddle-batch-size"),
) -> None:
    """Run TIF batch similarity search (photo extraction + image similarity + document aggregation).

    Args:
        config_path (Path): Path to YAML config.
        doc_input_start (Optional[str]): Override input mode. One of {"tif", "key"}.
        folder (Optional[Path]): Folder of TIF files for server-side batch search; overrides config.
        top_k (Optional[int]): Per-photo neighbor count.
        top_doc (Optional[int]): Number of docs per TIF to return.
        aggregation_strategy (Optional[str]): Aggregation strategy: "max" | "sum" | "mean".
        bruteforce_db_folder (Optional[Path]): Fallback DB folder for brute-force search.

        output_folder_for_results (Optional[Path]): Output folder for results.
        copy_parent_tif_docs_to_output (Optional[bool]): Copy ranked parent TIF documents into the run output.
        generate_tif_previews (Optional[bool]): Generate TIF page previews.
        create_per_query_subfolders_for_tif (Optional[bool]): Create per-query subfolders in the output.
        save_results_to_postgresql (Optional[bool]): Save results to PostgreSQL.
        doc_sim_img_check (Optional[bool]): Enable optional doc-sim image check.
        doc_sim_img_check_max_k (Optional[int]): Max K used for the image check.

        photo_extraction_mode (Optional[str]): Photo extractor mode: "yolo" or "bbox".
        photo_extractor_debug (Optional[bool]): Enable extractor debug output.
        yolo_model_path (Optional[str]): Path to YOLO weights.
        yolo_confidence_threshold (Optional[float]): YOLO confidence threshold.
        yolo_iou_threshold (Optional[float]): YOLO IOU threshold.
        yolo_imgsz (Optional[int]): YOLO inference image size.
        yolo_target_object_names (Optional[str]): JSON list of target object names (e.g., '["photo"]').
        bbox_list_json (Optional[str]): JSON list of bounding boxes.
        bbox_format (Optional[str]): Box format: "xyxy" | "xywh" | "cxcywh".
        bbox_normalized (Optional[bool]): Whether bbox coordinates are normalized.

        search_text (Optional[str]): OCR search text used to locate pages.
        language (Optional[str]): OCR language.
        ocr_backend (Optional[str]): OCR backend identifier.
        search_mode (Optional[str]): OCR search mode.
        allow_normalization (Optional[bool]): Normalize OCR text.
        remove_spaces_in_normalization (Optional[bool]): Remove spaces during normalization.
        recognized_text_debug (Optional[bool]): Emit recognized text for debugging.
        search_location_top (Optional[float]): Top offset for search window.
        use_offline_models (Optional[bool]): Use offline OCR models.
        use_angle_cls (Optional[bool]): Enable angle classification in OCR.
        use_gpu_for_paddle (Optional[bool]): Use GPU acceleration for PaddleOCR.
        paddle_batch_size (Optional[int]): PaddleOCR batch size.

    Returns:
        None

    Raises:
        typer.Exit: If invalid input mode is requested or required configuration is missing.
    """
    config = load_and_merge_configs(config_path)

    # Optional: honor log maintenance policy after config load (does not impact CLI execution)
    try:
        if maintain_log_files is not None:
            logs_dir = Path(__file__).resolve().parent / "logs"
            lm_cfg = (config.get("logging") or {}).get("log_maintenance", {})
            maintain_log_files(
                logs_dir,
                stem="image_similarity_system",
                remove_logs_days=int(lm_cfg.get("remove_logs_days", 7)),
                backup_logs=bool(lm_cfg.get("backup_logs", False)),
            )
    except Exception:
        pass

    # Key-driven mode: allow CLI override and delegate to external orchestrator
    on_disk_start = str(config.get("input_mode", {}).get("doc_input_start", "key")).lower()
    effective_start = str(doc_input_start or on_disk_start).lower()
    # Normalize invalid values to the default 'key' and warn
    if effective_start not in {"tif", "key"}:
        print(
            f"Warning: input_mode.doc_input_start='{effective_start}' is not supported; "
            "falling back to default 'key' input mode.",
            file=sys.stderr,
        )
        effective_start = "key"
    if effective_start == "key":
        if run_key_input_pipeline is None:
            print(
                "Error: key-driven input requested but external.filename_mapper module is unavailable.",
                file=sys.stderr,
            )
            raise typer.Exit(code=2)
        cfg_path_for_orch = config_path
        tmp_cfg_file: Optional[Path] = None
        if (doc_input_start is not None) and (on_disk_start != "key"):
            try:
                cfg_copy = dict(config)
                im = cfg_copy.setdefault("input_mode", {})
                im["doc_input_start"] = "key"
                with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml", encoding="utf-8") as tf:
                    yaml.safe_dump(cfg_copy, tf, allow_unicode=True)
                    tmp_cfg_file = Path(tf.name)
                cfg_path_for_orch = tmp_cfg_file
            except Exception as _e:
                print(f"Warning: failed to write temp config for key-mode override: {_e}. Falling back to config file.", file=sys.stderr)
        result = run_key_input_pipeline(cfg_path_for_orch)
        if tmp_cfg_file:
            try:
                tmp_cfg_file.unlink()
            except Exception:
                pass
        # Pretty-print results like TIF folder mode (avoid raw JSON dump)
        if isinstance(result, dict):
            batch_results = result.get("batch_results") or []
            for bi, br in enumerate(batch_results, 1):
                res = br.get("result") or {}
                if not isinstance(res, dict):
                    continue
                print(
                    "\n=== Per-Query Results (batch %d/%d, mode=%s, requested=%s, resolved=%s) ==="
                    % (bi, len(batch_results), br.get("mode"), br.get("requested"), br.get("resolved"))
                )
                for entry in res.get("per_query", []) or []:
                    try:
                        print(
                            f"Query: {entry['query_document']}  photos={entry['num_query_photos']}  "
                            f"agg={entry['aggregation_strategy_used']}  t={entry['elapsed_seconds']:.3f}s"
                        )
                        for j, d in enumerate(entry.get("top_docs", []), 1):
                            print(f"   {j:02d}. {d['document']}  score={d['score']:.4f}")
                    except Exception:
                        print(entry)
                if res.get("fallback_to_bruteforce_message"):
                    print(f"\n[Fallback] {res['fallback_to_bruteforce_message']}")
                if res.get("tif_run_output_path"):
                    print(f"\nRun output folder: {res['tif_run_output_path']}")
                if res.get("tif_run_summary_json_path"):
                    print(f"Summary JSON: {res['tif_run_summary_json_path']}")
                if res.get("tif_run_db_export_csv_path"):
                    print(f"PostgreSQL CSV export: {res['tif_run_db_export_csv_path']}")
        else:
            print(result)
        return

    st = config.setdefault("search_task", {})
    if folder is not None:
        st["input_tif_folder_for_search"] = str(folder)
    if top_k is not None:
        st["top_k"] = int(top_k)
    if top_doc is not None:
        st["top_doc"] = int(top_doc)
    if aggregation_strategy is not None:
        st["aggregation_strategy"] = str(aggregation_strategy)
    if bruteforce_db_folder is not None:
        st["bruteforce_db_folder"] = str(bruteforce_db_folder)

    if output_folder_for_results is not None:
        st["output_folder_for_results"] = str(output_folder_for_results)
    if copy_parent_tif_docs_to_output is not None:
        # Preserve both keys for backward compatibility while aligning with workflow expectations
        st["copy_parent_tif_docs_to_output"] = _bool_opt(copy_parent_tif_docs_to_output)
        st["copy_similar_images_to_output"] = _bool_opt(copy_parent_tif_docs_to_output)
    if generate_tif_previews is not None:
        st["generate_tif_previews"] = _bool_opt(generate_tif_previews)
    if create_per_query_subfolders_for_tif is not None:
        st["create_per_query_subfolders_for_tif"] = _bool_opt(create_per_query_subfolders_for_tif)
    if save_results_to_postgresql is not None:
        st["save_results_to_postgresql"] = _bool_opt(save_results_to_postgresql)
    if doc_sim_img_check is not None:
        st["doc_sim_img_check"] = _bool_opt(doc_sim_img_check)
    if doc_sim_img_check_max_k is not None:
        st["doc_sim_img_check_max_k"] = int(doc_sim_img_check_max_k)

    pec = config.setdefault("photo_extractor_config", {})
    if photo_extraction_mode is not None:
        pec["photo_extraction_mode"] = str(photo_extraction_mode)
    if photo_extractor_debug is not None:
        pec["photo_extractor_debug"] = _bool_opt(photo_extractor_debug)
    if yolo_model_path is not None:
        pec.setdefault("yolo_object_detection", {})["model_path"] = str(yolo_model_path)
    if any(v is not None for v in [yolo_confidence_threshold, yolo_iou_threshold, yolo_imgsz, yolo_target_object_names]):
        yinf = pec.setdefault("yolo_object_detection", {}).setdefault("inference", {})
        if yolo_confidence_threshold is not None:
            yinf["confidence_threshold"] = float(yolo_confidence_threshold)
        if yolo_iou_threshold is not None:
            yinf["iou_threshold"] = float(yolo_iou_threshold)
        if yolo_imgsz is not None:
            yinf["imgsz"] = int(yolo_imgsz)
        names = _json_list(yolo_target_object_names)
        if names is not None:
            yinf["target_object_names"] = names
    if any(v is not None for v in [bbox_list_json, bbox_format, bbox_normalized]):
        bex = pec.setdefault("bbox_extraction", {})
        if bbox_list_json:
            try:
                bex["bbox_list"] = json.loads(bbox_list_json)
            except Exception as _e:
                print(f"Warning: --bbox-list could not be parsed as JSON: {_e}. Ignoring.", file=sys.stderr)
        if bbox_format is not None:
            bex["bbox_format"] = str(bbox_format)
        if bbox_normalized is not None:
            bex["normalized"] = _bool_opt(bbox_normalized)

    if any(
        v is not None
        for v in [
            search_text,
            language,
            ocr_backend,
            search_mode,
            allow_normalization,
            remove_spaces_in_normalization,
            recognized_text_debug,
            search_location_top,
            use_offline_models,
            use_angle_cls,
            use_gpu_for_paddle,
            paddle_batch_size,
        ]
    ):
        scfg = config.setdefault("searcher_config", {})
        if search_text is not None:
            scfg["search_text"] = str(search_text)
        if language is not None:
            scfg["language"] = str(language)
        if ocr_backend is not None:
            scfg["ocr_backend"] = str(ocr_backend)
        if search_mode is not None:
            scfg["search_mode"] = str(search_mode)
        if allow_normalization is not None:
            scfg["allow_normalization"] = _bool_opt(allow_normalization)
        if remove_spaces_in_normalization is not None:
            scfg["remove_spaces_in_normalization"] = _bool_opt(remove_spaces_in_normalization)
        if recognized_text_debug is not None:
            scfg["recognized_text_debug"] = _bool_opt(recognized_text_debug)
        if search_location_top is not None:
            scfg.setdefault("search_location", {})["top"] = float(search_location_top)
        if use_offline_models is not None:
            scfg["use_offline_models"] = _bool_opt(use_offline_models)
        if use_angle_cls is not None:
            scfg["use_angle_cls"] = _bool_opt(use_angle_cls)
        if use_gpu_for_paddle is not None:
            scfg["use_gpu_for_paddle"] = _bool_opt(use_gpu_for_paddle)
        if paddle_batch_size is not None:
            scfg["paddle_batch_size"] = int(paddle_batch_size)

    project_root = Path(__file__).resolve().parent
    result = execute_tif_batch_search_workflow(config, project_root)

    # Summarize
    print("\n=== Per-Query Results ===")
    for entry in result.get("per_query", []) or []:
        try:
            print(
                f"Query: {entry['query_document']}  photos={entry['num_query_photos']}  "
                f"agg={entry['aggregation_strategy_used']}  t={entry['elapsed_seconds']:.3f}s"
            )
            for j, d in enumerate(entry.get("top_docs", []), 1):
                print(f"   {j:02d}. {d['document']}  score={d['score']:.4f}")
        except Exception:
            print(entry)

    if result.get("fallback_to_bruteforce_message"):
        print(f"\n[Fallback] {result['fallback_to_bruteforce_message']}")
    if result.get("tif_run_output_path"):
        print(f"\nRun output folder: {result['tif_run_output_path']}")
    if result.get("tif_run_summary_json_path"):
        print(f"Summary JSON: {result['tif_run_summary_json_path']}")
    if result.get("tif_run_db_export_csv_path"):
        print(f"PostgreSQL CSV export: {result['tif_run_db_export_csv_path']}")


@app.command("build-image-index")
def cmd_build_index_with_tif(
    config_path: Path = typer.Option(Path("configs/image_similarity_config.yaml"), help="Path to YAML config."),
    doc_input_start: Optional[str] = typer.Option(None, "--doc-input-start", help="Override input_mode.doc_input_start: tif|key"),
    key_input_config_path: Optional[Path] = typer.Option(None, "--key-input-config", help="Override input_mode.key_input_config_path (path to key_input_config.yaml)."),
    input_table_path: Optional[Path] = typer.Option(None, "--input-table-path", help="Key table path (xlsx/xls/csv/json/ndjson). Overrides key_input.input_table_path for this run."),
    folder: Optional[Path] = typer.Option(None, "--folder", help="Folder of TIF files to derive image index from (overrides config)."),

    # Index build parameters
    engine: Optional[str] = typer.Option(None, "--engine", help="Vector DB provider: faiss|qdrant|bruteforce."),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Feature extraction batch size."),
    force_rebuild_index: Optional[bool] = typer.Option(None, "--force-rebuild-index/--no-force-rebuild-index"),

    # Photo extraction
    photo_extraction_mode: Optional[str] = typer.Option(None, "--photo-extraction-mode", help="yolo|bbox"),
    photo_extractor_debug: Optional[bool] = typer.Option(None, "--photo-extractor-debug/--no-photo-extractor-debug"),
    yolo_model_path: Optional[str] = typer.Option(None, "--yolo-model-path"),
    yolo_confidence_threshold: Optional[float] = typer.Option(None, "--yolo-conf-thresh"),
    yolo_iou_threshold: Optional[float] = typer.Option(None, "--yolo-iou-thresh"),
    yolo_imgsz: Optional[int] = typer.Option(None, "--yolo-imgsz"),
    yolo_target_object_names: Optional[str] = typer.Option(None, "--yolo-target-object-names"),
    bbox_list_json: Optional[str] = typer.Option(None, "--bbox-list"),
    bbox_format: Optional[str] = typer.Option(None, "--bbox-format", help="xyxy|xywh|cxcywh"),
    bbox_normalized: Optional[bool] = typer.Option(None, "--bbox-normalized/--no-bbox-normalized"),

    # OCR/Text searcher config to determine pages/photos of interest prior to cropping
    search_text: Optional[str] = typer.Option(None, "--search-text"),
    language: Optional[str] = typer.Option(None, "--language"),
    ocr_backend: Optional[str] = typer.Option(None, "--ocr-backend"),
    search_mode: Optional[str] = typer.Option(None, "--search-mode"),
    allow_normalization: Optional[bool] = typer.Option(None, "--allow-normalization/--no-allow-normalization"),
    remove_spaces_in_normalization: Optional[bool] = typer.Option(None, "--remove-spaces/--no-remove-spaces"),
    recognized_text_debug: Optional[bool] = typer.Option(None, "--recognized-text-debug/--no-recognized-text-debug"),
    search_location_top: Optional[float] = typer.Option(None, "--search-location-top"),
    use_offline_models: Optional[bool] = typer.Option(None, "--use-offline-models/--no-use-offline-models"),
    use_angle_cls: Optional[bool] = typer.Option(None, "--use-angle-cls/--no-use-angle-cls"),
    use_gpu_for_paddle: Optional[bool] = typer.Option(None, "--use-gpu-for-paddle/--no-use-gpu-for-paddle"),
    paddle_batch_size: Optional[int] = typer.Option(None, "--paddle-batch-size"),
) -> None:
    """Build an index from photos extracted from TIF documents.

    The build process can operate in two modes: "tif" (scan folders of TIFs) or
    "key" (key-driven mode via an external orchestrator). When "key" mode is
    active, optional overrides may be applied for the key input configuration.

    Args:
        config_path (Path): Path to YAML config.
        doc_input_start (Optional[str]): Override input mode. One of {"tif", "key"}.
        key_input_config_path (Optional[Path]): Override input_mode.key_input_config_path.
        input_table_path (Optional[Path]): Override the key table path for this run.
        folder (Optional[Path]): Folder of TIF files to derive image index from; overrides config.

        engine (Optional[str]): Vector DB provider: "faiss" | "qdrant" | "bruteforce".
        batch_size (Optional[int]): Feature extraction batch size.
        force_rebuild_index (Optional[bool]): Force a full rebuild of the index.

        photo_extraction_mode (Optional[str]): Photo extractor mode: "yolo" or "bbox".
        photo_extractor_debug (Optional[bool]): Enable extractor debug output.
        yolo_model_path (Optional[str]): Path to YOLO weights.
        yolo_confidence_threshold (Optional[float]): YOLO confidence threshold.
        yolo_iou_threshold (Optional[float]): YOLO IOU threshold.
        yolo_imgsz (Optional[int]): YOLO inference image size.
        yolo_target_object_names (Optional[str]): JSON list of target object names.
        bbox_list_json (Optional[str]): JSON list of bounding boxes.
        bbox_format (Optional[str]): Box format: "xyxy" | "xywh" | "cxcywh".
        bbox_normalized (Optional[bool]): Whether bbox coordinates are normalized.

        search_text (Optional[str]): OCR search text used to locate pages of interest.
        language (Optional[str]): OCR language.
        ocr_backend (Optional[str]): OCR backend identifier.
        search_mode (Optional[str]): OCR search mode.
        allow_normalization (Optional[bool]): Normalize OCR text.
        remove_spaces_in_normalization (Optional[bool]): Remove spaces during normalization.
        recognized_text_debug (Optional[bool]): Emit recognized text for debugging.
        search_location_top (Optional[float]): Top offset for search window.
        use_offline_models (Optional[bool]): Use offline OCR models.
        use_angle_cls (Optional[bool]): Enable angle classification in OCR.
        use_gpu_for_paddle (Optional[bool]): Use GPU acceleration for PaddleOCR.
        paddle_batch_size (Optional[int]): PaddleOCR batch size.

    Returns:
        None

    Raises:
        typer.Exit: If invalid input mode is requested or the input folder cannot be resolved.
    """
    config = load_and_merge_configs(config_path)

    # Optional: honor log maintenance policy after config load (does not impact CLI execution)
    try:
        if maintain_log_files is not None:
            logs_dir = Path(__file__).resolve().parent / "logs"
            lm_cfg = (config.get("logging") or {}).get("log_maintenance", {})
            maintain_log_files(
                logs_dir,
                stem="image_similarity_system",
                remove_logs_days=int(lm_cfg.get("remove_logs_days", 7)),
                backup_logs=bool(lm_cfg.get("backup_logs", False)),
            )
    except Exception:
        pass

    # Support key-driven build mode when configured or overridden via CLI
    on_disk_start = str(config.get("input_mode", {}).get("doc_input_start", "key")).lower()
    effective_start = str(doc_input_start or on_disk_start).lower()
    if effective_start not in {"tif", "key"}:
        effective_start = "key"
    if effective_start == "key":
        if run_key_input_pipeline_for_index is None:
            print(
                "Error: key-driven input requested for build, but key input orchestrator is unavailable.",
                file=sys.stderr,
            )
            raise typer.Exit(code=2)
        cfg_path_for_orch = config_path
        tmp_cfg_file_ki: Optional[Path] = None
        tmp_key_cfg_file: Optional[Path] = None

        # If a key table path override is provided, generate a temporary key_input_config
        # that points to the given table and ensure input_mode.key_input_config_path references it.
        if input_table_path is not None:
            try:
                base_key_cfg_path: Path
                if key_input_config_path is not None:
                    base_key_cfg_path = key_input_config_path
                else:
                    raw = str(config.get("input_mode", {}).get("key_input_config_path", "external/key_input/key_input_config.yaml"))
                    base_key_cfg_path = Path(raw)
                    if not base_key_cfg_path.is_absolute():
                        base_key_cfg_path = (Path(__file__).resolve().parent / base_key_cfg_path).resolve()
                base_kcfg: Dict[str, Any] = {}
                if base_key_cfg_path.exists():
                    try:
                        with base_key_cfg_path.open("r", encoding="utf-8") as f:
                            base_kcfg = yaml.safe_load(f) or {}
                    except Exception:
                        base_kcfg = {}
                ki = base_kcfg.setdefault("key_input", {})
                ki["input_table_path"] = str(input_table_path)
                with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml", encoding="utf-8") as tfk:
                    yaml.safe_dump(base_kcfg, tfk, allow_unicode=True)
                    tmp_key_cfg_file = Path(tfk.name)
                # Ensure main config will reference the temp key cfg
                key_input_config_path = tmp_key_cfg_file
            except Exception as _e:
                print(f"Warning: failed to prepare temporary key_input_config with input_table_path override: {_e}", file=sys.stderr)

        # If mode override or key-input-config override is present, write a temp main config
        if (doc_input_start is not None and on_disk_start != "key") or (key_input_config_path is not None):
            try:
                cfg_copy = dict(config)
                im = cfg_copy.setdefault("input_mode", {})
                im["doc_input_start"] = "key"
                if key_input_config_path is not None:
                    im["key_input_config_path"] = str(key_input_config_path)
                with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml", encoding="utf-8") as tf:
                    yaml.safe_dump(cfg_copy, tf, allow_unicode=True)
                    tmp_cfg_file_ki = Path(tf.name)
                cfg_path_for_orch = tmp_cfg_file_ki
            except Exception as _e:
                print(
                    f"Warning: failed to write temp config for key-mode build override: {_e}. Falling back to config file.",
                    file=sys.stderr,
                )
        result = run_key_input_pipeline_for_index(cfg_path_for_orch)
        if tmp_cfg_file_ki:
            try:
                tmp_cfg_file_ki.unlink()
            except Exception:
                pass
        if tmp_key_cfg_file:
            try:
                tmp_key_cfg_file.unlink()
            except Exception:
                pass
        print(result)
        return

    # Vector DB and indexing bits
    vdb = config.setdefault("vector_database", {})
    if engine is not None:
        vdb["provider"] = str(engine)
    it = config.setdefault("indexing_task", {})
    if batch_size is not None:
        it["batch_size"] = int(batch_size)
    if force_rebuild_index is not None:
        it["force_rebuild_index"] = _bool_opt(force_rebuild_index)

    # Photo extraction overrides
    pec = config.setdefault("photo_extractor_config", {})
    if photo_extraction_mode is not None:
        pec["photo_extraction_mode"] = str(photo_extraction_mode)
    if photo_extractor_debug is not None:
        pec["photo_extractor_debug"] = _bool_opt(photo_extractor_debug)
    if yolo_model_path is not None:
        pec.setdefault("yolo_object_detection", {})["model_path"] = str(yolo_model_path)
    if any(v is not None for v in [yolo_confidence_threshold, yolo_iou_threshold, yolo_imgsz, yolo_target_object_names]):
        yinf = pec.setdefault("yolo_object_detection", {}).setdefault("inference", {})
        if yolo_confidence_threshold is not None:
            yinf["confidence_threshold"] = float(yolo_confidence_threshold)
        if yolo_iou_threshold is not None:
            yinf["iou_threshold"] = float(yolo_iou_threshold)
        if yolo_imgsz is not None:
            yinf["imgsz"] = int(yolo_imgsz)
        names = _json_list(yolo_target_object_names)
        if names is not None:
            yinf["target_object_names"] = names
    if any(v is not None for v in [bbox_list_json, bbox_format, bbox_normalized]):
        bex = pec.setdefault("bbox_extraction", {})
        if bbox_list_json:
            try:
                bex["bbox_list"] = json.loads(bbox_list_json)
            except Exception as _e:
                print(f"Warning: --bbox-list could not be parsed as JSON: {_e}. Ignoring.", file=sys.stderr)
        if bbox_format is not None:
            bex["bbox_format"] = str(bbox_format)
        if bbox_normalized is not None:
            bex["normalized"] = _bool_opt(bbox_normalized)

    # Text searcher overrides (find pages with photos first)
    if any(
        v is not None
        for v in [
            search_text,
            language,
            ocr_backend,
            search_mode,
            allow_normalization,
            remove_spaces_in_normalization,
            recognized_text_debug,
            search_location_top,
            use_offline_models,
            use_angle_cls,
            use_gpu_for_paddle,
            paddle_batch_size,
        ]
    ):
        scfg = config.setdefault("searcher_config", {})
        if search_text is not None:
            scfg["search_text"] = str(search_text)
        if language is not None:
            scfg["language"] = str(language)
        if ocr_backend is not None:
            scfg["ocr_backend"] = str(ocr_backend)
        if search_mode is not None:
            scfg["search_mode"] = str(search_mode)
        if allow_normalization is not None:
            scfg["allow_normalization"] = _bool_opt(allow_normalization)
        if remove_spaces_in_normalization is not None:
            scfg["remove_spaces_in_normalization"] = _bool_opt(remove_spaces_in_normalization)
        if recognized_text_debug is not None:
            scfg["recognized_text_debug"] = _bool_opt(recognized_text_debug)
        if search_location_top is not None:
            scfg.setdefault("search_location", {})["top"] = float(search_location_top)
        if use_offline_models is not None:
            scfg["use_offline_models"] = _bool_opt(use_offline_models)
        if use_angle_cls is not None:
            scfg["use_angle_cls"] = _bool_opt(use_angle_cls)
        if use_gpu_for_paddle is not None:
            scfg["use_gpu_for_paddle"] = _bool_opt(use_gpu_for_paddle)
        if paddle_batch_size is not None:
            scfg["paddle_batch_size"] = int(paddle_batch_size)

    project_root = Path(__file__).resolve().parent
    # Determine folder to use: CLI override > config.indexing_task.input_tif_folder_for_indexing
    use_folder = folder
    if use_folder is None:
        cfg_folder = config.get("indexing_task", {}).get("input_tif_folder_for_indexing")
        if cfg_folder:
            try:
                use_folder = Path(cfg_folder)
            except Exception:
                use_folder = None
    if use_folder is None:
        print(
            "Error: --folder not provided and indexing_task.input_tif_folder_for_indexing is not set in config.",
            file=sys.stderr,
        )
        raise typer.Exit(code=2)
    # Build index directly from TIF folder (includes page filtering by text && photo extraction)
    result = build_index_from_tif_folder_workflow(config, project_root, use_folder)
    print(result)


@app.command("diagnostics")
def cmd_diagnostics(
    config_path: Path = typer.Option(Path("configs/image_similarity_config.yaml"), help="Path to YAML config."),
) -> None:
    """Print a diagnostic summary of the configured vector provider and datasets.

    The command performs a non-destructive inspection of the configured provider
    (FAISS or Qdrant) and reports:
    - Provider selection and readiness
    - Total items indexed
    - Shard/collection counts with per-unit sizes
    - Mapping parity (FAISS only): index.ntotal vs mapping length
    - Effective key config values used for provider discovery

    Args:
        config_path (Path): Path to YAML configuration file.

    Returns:
        None

    Raises:
        typer.Exit: With nonzero code if diagnostics cannot be performed due to invalid configuration.
    """
    try:
        cfg = load_and_merge_configs(config_path)
    except Exception as e:
        typer.echo(f"Error: Failed to load config '{config_path}': {e}")
        raise typer.Exit(code=2)

    vdb = cfg.get("vector_database", {}) or {}
    provider = str(vdb.get("provider", "faiss")).lower()
    typer.echo(f"Provider: {provider}")

    # Shared display of effective FAISS/Qdrant settings
    def _pp(k: str, v: object) -> str:
        try:
            return f"{k}={v}"
        except Exception:
            return f"{k}=(unprintable)"

    if provider == "faiss":
        faiss_conf = vdb.get("faiss", {}) or {}
        out_dir_raw = faiss_conf.get("output_directory")
        if not out_dir_raw:
            typer.echo("Error: vector_database.faiss.output_directory is required for diagnostics.")
            raise typer.Exit(code=2)
        out_dir = Path(out_dir_raw)
        if not out_dir.is_absolute():
            out_dir = (Path(__file__).resolve().parent / out_dir).resolve()
        stem = str(faiss_conf.get("filename_stem", "faiss_index"))
        idx_type = str(faiss_conf.get("index_type", "flat")).lower()
        model_name = str((cfg.get("feature_extractor", {}) or {}).get("model_name", "model")).lower()

        # Shard file pattern and legacy adoption
        descriptive_core = f"{stem}_{model_name}_{idx_type}"
        shard_glob = f"{descriptive_core}_s*.index"
        shard_files = sorted(out_dir.glob(shard_glob))
        legacy_index = out_dir / f"{descriptive_core}.index"
        legacy_map = out_dir / f"{descriptive_core}_mapping.pkl"

        # Attempt to import faiss lazily
        try:
            import faiss  # type: ignore
        except Exception as e_imp:
            typer.echo(f"Error: faiss not available for diagnostics: {e_imp}")
            raise typer.Exit(code=2)

        total = 0
        shard_report: List[str] = []
        mapping_issues: List[str] = []

        # Prefer shards; otherwise check legacy files
        if shard_files:
            for sidx in sorted(shard_files):
                try:
                    idx = faiss.read_index(str(sidx))
                    ntotal = int(idx.ntotal)
                    map_path = sidx.with_name(sidx.stem + "_mapping.pkl")
                    mlen = None
                    if map_path.exists():
                        try:
                            import pickle
                            with open(map_path, "rb") as f:
                                idmap = pickle.load(f)
                                mlen = int(len(idmap))
                        except Exception:
                            mlen = None
                    shard_report.append(f"- {sidx.name}: ntotal={ntotal}, mapping={mlen if mlen is not None else 'missing'}")
                    total += ntotal
                    if (mlen is not None) and (mlen != ntotal):
                        mapping_issues.append(f"mapping size mismatch for {sidx.name}: index.ntotal={ntotal}, mapping={mlen}")
                except Exception as e_rd:
                    shard_report.append(f"- {sidx.name}: load error: {e_rd}")
        else:
            # Legacy single index naming (pre-sharded adoption)
            idx_path = out_dir / f"{stem}_{model_name}_{idx_type}.index"
            map_path = out_dir / f"{stem}_{model_name}_{idx_type}_mapping.pkl"
            if idx_path.exists():
                try:
                    idx = faiss.read_index(str(idx_path))
                    ntotal = int(idx.ntotal)
                    total = ntotal
                    mlen = None
                    if map_path.exists():
                        import pickle
                        with open(map_path, "rb") as f:
                            idmap = pickle.load(f)
                            mlen = int(len(idmap))
                    shard_report.append(f"- legacy {idx_path.name}: ntotal={ntotal}, mapping={mlen if mlen is not None else 'missing'}")
                    if (mlen is not None) and (mlen != ntotal):
                        mapping_issues.append(
                            f"mapping size mismatch for legacy {idx_path.name}: index.ntotal={ntotal}, mapping={mlen}"
                        )
                except Exception as e_legacy:
                    shard_report.append(f"- legacy {idx_path.name}: load error: {e_legacy}")
            else:
                shard_report.append("- no shards or legacy index found")

        # Print effective config & results
        typer.echo("Effective FAISS settings:")
        typer.echo("  " + _pp("output_directory", str(out_dir)))
        typer.echo("  " + _pp("filename_stem", stem))
        typer.echo("  " + _pp("index_type", idx_type))
        typer.echo("  " + _pp("model_name", model_name))
        typer.echo("Shards:")
        for line in shard_report:
            typer.echo("  " + line)
        typer.echo(f"Total indexed items (sum of ntotal): {total}")
        if mapping_issues:
            typer.echo("Mapping parity issues detected:")
            for m in mapping_issues:
                typer.echo("  - " + m)
        else:
            typer.echo("Mapping parity: OK (no mismatches detected or mappings missing)")
        return

    if provider == "qdrant":
        qconf = vdb.get("qdrant", {}) or {}
        # Lazy import to avoid optional dependency at CLI import time
        try:
            from qdrant_client import QdrantClient  # type: ignore
        except Exception as e_imp:
            typer.echo(f"Error: qdrant-client not available for diagnostics: {e_imp}")
            raise typer.Exit(code=2)

        # Connect (embedded or server)
        client: Optional[QdrantClient] = None
        try:
            if qconf.get("location"):
                loc = qconf.get("location")
                client = QdrantClient(location=loc)
                mode = f"embedded:{loc}"
            else:
                host = qconf.get("host", "localhost")
                port = int(qconf.get("port", 6333))
                client = QdrantClient(host=host, port=port)
                mode = f"server:{host}:{port}"
        except Exception as e_cli:
            typer.echo(f"Error: Failed to initialize Qdrant client: {e_cli}")
            raise typer.Exit(code=2)

        base = str(qconf.get("collection_name_stem", "image_similarity_collection"))
        model = str((cfg.get("feature_extractor", {}) or {}).get("model_name", "model")).lower()
        base_name = f"{base}_{model}"
        import re as _re
        pat = _re.compile(rf"^{_re.escape(base_name)}(_s\d{{4}})?$")

        total = 0
        detail: List[str] = []
        try:
            resp = client.get_collections()
            names = [c.name for c in getattr(resp, "collections", [])]
            names = [n for n in names if pat.match(n)]
            names.sort()
            for n in names:
                try:
                    c = client.count(collection_name=n, exact=False)
                    cnt = int(getattr(c, "count", 0))
                except Exception:
                    cnt = 0
                total += cnt
                detail.append(f"- {n}: points={cnt}")
        except Exception as e_ls:
            typer.echo(f"Error: Failed to list Qdrant collections: {e_ls}")
            raise typer.Exit(code=2)

        typer.echo("Effective Qdrant settings:")
        typer.echo("  " + _pp("mode", mode))
        typer.echo("  " + _pp("collection_base", base_name))
        for line in detail:
            typer.echo("  " + line)
        typer.echo(f"Total points across collections: {total}")
        return

    # Bruteforce or unknown provider
    typer.echo("Diagnostics: provider has no persisted index (bruteforce or unknown). Nothing to diagnose.")


@app.command("validate-config")
def cmd_validate_config(
    config_path: Path = typer.Option(Path("configs/image_similarity_config.yaml"), help="Path to YAML config."),
) -> None:
    """Validate configuration schema and emit actionable findings.

    This performs structural checks on the YAML configuration without mutating
    files or connecting to external services. It reports missing required keys,
    invalid enum values, and questionable paths.

    Args:
        config_path (Path): Path to YAML configuration file.

    Returns:
        None

    Raises:
        typer.Exit: With nonzero code if validation fails.
    """
    try:
        cfg = load_and_merge_configs(config_path)
    except Exception as e:
        typer.echo(f"Error: Failed to load config '{config_path}': {e}")
        raise typer.Exit(code=2)

    errors: List[str] = []
    warnings: List[str] = []

    # feature_extractor
    fe = cfg.get("feature_extractor", {}) or {}
    if not isinstance(fe.get("model_name"), str) or not fe.get("model_name"):
        errors.append("feature_extractor.model_name (str) is required.")

    # vector_database
    vdb = cfg.get("vector_database", {}) or {}
    provider = str(vdb.get("provider", "faiss")).lower()
    if provider not in {"faiss", "qdrant", "bruteforce"}:
        errors.append("vector_database.provider must be one of {'faiss','qdrant','bruteforce'}.")

    # provider-specific checks
    if provider == "faiss":
        fc = vdb.get("faiss", {}) or {}
        if not fc.get("output_directory"):
            errors.append("vector_database.faiss.output_directory is required for FAISS.")
        part = fc.get("partition_capacity") or fc.get("total_indexes_per_file")
        if part is not None:
            try:
                if int(part) <= 0:
                    errors.append("FAISS partition_capacity must be a positive integer when provided.")
            except Exception:
                errors.append("FAISS partition_capacity must be an integer when provided.")
    elif provider == "qdrant":
        qc = vdb.get("qdrant", {}) or {}
        # Either embedded 'location' or server 'host' must be provided
        if not (qc.get("location") or qc.get("host")):
            warnings.append("Qdrant: neither 'location' nor 'host' provided; defaults will be used (localhost:6333).")
        part = qc.get("partition_capacity") or qc.get("max_points_per_collection")
        if part is not None:
            try:
                if int(part) <= 0:
                    errors.append("Qdrant partition_capacity must be a positive integer when provided.")
            except Exception:
                errors.append("Qdrant partition_capacity must be an integer when provided.")

    # path checks (do not create or mutate)
    def _optional_dir(path_str: Optional[str], key: str) -> None:
        if not path_str:
            return
        p = Path(path_str)
        if not p.is_absolute():
            warnings.append(f"'{key}' is relative; it will be resolved against project root at runtime: {path_str}")

    it = cfg.get("indexing_task", {}) or {}
    _optional_dir(it.get("image_folder_to_index"), "indexing_task.image_folder_to_index")
    st = cfg.get("search_task", {}) or {}
    _optional_dir(st.get("input_tif_folder_for_search"), "search_task.input_tif_folder_for_search")

    if errors:
        typer.echo("Validation: FAILED")
        for e in errors:
            typer.echo("  - " + e)
        if warnings:
            typer.echo("Warnings:")
            for w in warnings:
                typer.echo("  - " + w)
        raise typer.Exit(code=1)

    typer.echo("Validation: OK")
    if warnings:
        typer.echo("Warnings:")
        for w in warnings:
            typer.echo("  - " + w)


if __name__ == "__main__":
    app()
