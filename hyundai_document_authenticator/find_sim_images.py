#!/usr/bin/env python
"""Image Similarity Tester - Local CLI (image-only).

This module provides a standalone CLI tool for validating the image similarity
method used by the TIF document workflows without invoking any TIF-specific
components. It reuses the system configuration (configs/image_similarity_config.yaml)
so all model and storage settings are honored consistently.

The tool produces per-query JSON summaries and a consolidated CSV file for an
entire run. It optionally copies the query image and the ranked similar images
into the run output folder, depending on configuration flags.

Examples:
    Single image using FAISS (when index is available):
        python find_sim_images.py search-img --query ./instance/database_images/sample.jpg --top-k 5

    Batch search for all images in a folder (globs supported by the scanner):
        python find_sim_images.py search-img --folder ./instance/database_images --top-k 5

    Brute-force fallback (or when index is not ready):
        python find_sim_images.py search-img --query ./instance/database_images/a.jpg --top-k 5 \
            --bruteforce-db-folder ./instance/database_images

Raises:
    None
"""
from __future__ import annotations

import sys
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

import typer
from dotenv import find_dotenv, load_dotenv

# Ensure package import paths
sys.path.append(str(Path(__file__).resolve().parent))

# Core engine imports
from core_engine.image_similarity_system.config_loader import load_and_merge_configs
from core_engine.image_similarity_system.feature_extractor import FeatureExtractor
from core_engine.image_similarity_system.faiss_manager import FaissIndexManager
from core_engine.image_similarity_system.log_utils import setup_logging
from core_engine.image_similarity_system.constants import JSON_SUMMARY_DEFAULT_FILENAME
from core_engine.image_similarity_system.searcher import ImageSimilaritySearcher
from core_engine.image_similarity_system.utils import (
    detect_requesting_username,
    save_similar_images_to_folder,
)

# Optional: Qdrant provider (fallback silently if unavailable)
try:
    from core_engine.image_similarity_system.qdrant_manager import QdrantManager  # type: ignore
except Exception:  # noqa: BLE001 - Optional dependency
    QdrantManager = None  # type: ignore[misc, assignment]

app: typer.Typer = typer.Typer(
    add_completion=False, help="Image Similarity Tester - Local CLI (image-only)"
)


class CsvSchema(str, Enum):
    """Enum for CSV output schemas used by the image-only tester.

    Values:
        tif_compat: TIF-compatible per-query schema (default).
        image_only: Compact, image-focused per-query schema.
    """

    tif_compat = "tif_compat"
    image_only = "image_only"


# -----------------
# Helper functions
# -----------------

def _load_project_dotenv(start: Optional[Path] = None, filename: str = ".env") -> Optional[Path]:
    """Load a project-level .env file (best-effort, non-fatal).

    The function searches for an .env file starting from the current working
    directory up the tree and, if not found, attempts to load it from the
    repository root (one level above the package root).

    Args:
        start: Optional starting directory for the search. If ``None``, the
            current working directory is used implicitly.
        filename: The environment file name to search for.

    Returns:
        Optional[Path]: The resolved path of the loaded .env file if found,
        otherwise ``None``.

    Raises:
        None
    """
    try:
        path_str: str = find_dotenv(filename=filename, usecwd=True)
        if path_str:
            load_dotenv(dotenv_path=path_str, override=False)
            return Path(path_str).resolve()
        repo_root: Path = Path(__file__).resolve().parents[1]
        candidate: Path = repo_root / filename
        if candidate.exists():
            load_dotenv(dotenv_path=str(candidate), override=False)
            return candidate.resolve()
    except Exception:  # noqa: BLE001 - defensive: env loading must not break execution
        return None
    return None


def _resolve_path(path_str: Optional[str], project_root: Optional[Path] = None) -> Optional[Path]:
    """Resolve a string path relative to a project root when needed.

    Args:
        path_str: Path string to resolve. May be absolute or relative.
        project_root: Base directory to resolve relative paths against.

    Returns:
        Optional[Path]: Resolved absolute path, or ``None`` when ``path_str``
        is falsy.

    Raises:
        None
    """
    if not path_str:
        return None
    path_obj: Path = Path(path_str)
    if path_obj.is_absolute():
        return path_obj.resolve()
    base: Path = project_root if project_root else Path(__file__).resolve().parent
    return (base / path_obj).resolve()


def _collect_query_images(folder: Path) -> List[Path]:
    """Collect valid image file paths from a directory (non-recursive).

    Supported extensions: png, jpg, jpeg, bmp, gif, tif, tiff, webp.

    Args:
        folder: Directory to scan for query images.

    Returns:
        List[Path]: Sorted list of resolved file paths found in ``folder``.

    Raises:
        None
    """
    exts: List[str] = [
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.bmp",
        "*.gif",
        "*.tif",
        "*.tiff",
        "*.webp",
    ]
    paths: List[Path] = []
    for ext in exts:
        paths.extend(Path(folder).glob(ext))
    # Deduplicate and ensure only files remain
    return sorted(set(p.resolve() for p in paths if p.is_file()))


def _write_image_only_csv(
    export_dir: Path,
    run_id: str,
    user: str,
    per_query_results: List[Dict[str, Any]],
    per_query_timestamps: Dict[str, str],
    per_query_meta: Dict[str, Dict[str, Any]],
    top_k: int,
    provider: str,
    index_identifier: Optional[str],
    filename_stem: str = "image_only_per_query",
) -> Path:
    """Write a compact, image-only per-query CSV.

    The schema is purpose-built for image-to-image search results and avoids
    TIF-centric columns. One row is written per query image.

    Columns:
        - run_identifier (str)
        - requesting_username (str)
        - search_time_stamp (ISO str)
        - query_image (str)
        - top_k (int)
        - results (JSON array of {"path": str, "score": float})
        - provider (str)                # e.g., faiss|qdrant|bruteforce
        - model_name (str)
        - index_identifier (str|None)   # FAISS path or Qdrant descriptor
        - elapsed_seconds (float)

    Args:
        export_dir: Destination directory for the CSV file.
        run_id: Unique run identifier for this search job.
        user: Requesting username for audit purposes.
        per_query_results: Per-query entries as produced by the search.
        per_query_timestamps: Map of query image name to ISO timestamp.
        per_query_meta: Map of query image name to meta fields (elapsed, method, model).
        top_k: Number of neighbors requested per query.
        provider: Provider used (faiss|qdrant|bruteforce).
        index_identifier: Optional identifier for the underlying index.
        filename_stem: Output filename stem (default: "image_only_per_query").

    Returns:
        Path: Path to the written CSV file.

    Raises:
        None
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    csv_path: Path = export_dir / f"{filename_stem}_{run_id}.csv"

    headers: List[str] = [
        "run_identifier",
        "requesting_username",
        "search_time_stamp",
        "query_image",
        "top_k",
        "results",
        "provider",
        "model_name",
        "index_identifier",
        "elapsed_seconds",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(headers)
        for entry in per_query_results:
            qname: str = str(entry.get("query_document", "")).strip()
            # Build results payload from the internal top_docs structure
            results_json: List[Dict[str, Any]] = []
            for td in entry.get("top_docs") or []:
                try:
                    pth = td.get("document")
                    sc = td.get("score")
                    if pth is not None:
                        # Ensure scores are rounded to 4 decimals in the CSV JSON payload
                        results_json.append({
                            "path": str(pth),
                            "score": (round(float(sc), 4) if sc is not None else None),
                        })
                except Exception:
                    continue
            meta: Dict[str, Any] = per_query_meta.get(qname, {})
            row: List[Any] = [
                run_id,
                user,
                per_query_timestamps.get(qname, ""),
                qname,
                int(top_k),
                json.dumps(results_json, ensure_ascii=False),
                str(provider),
                str(meta.get("model", "")),
                index_identifier or "",
                round(float(meta.get("elapsed_seconds", 0.0)), 4),
            ]
            writer.writerow(row)

    return csv_path


# -----------------
# CLI command
# -----------------

@app.command("search-img")
def search_images(
    config_path: Path = typer.Option(
        Path("configs/image_similarity_config.yaml"), help="Path to YAML config."
    ),
    query: Optional[Path] = typer.Option(
        None, "--query", help="Path to a single query image (jpg/png/...)."
    ),
    folder: Optional[Path] = typer.Option(
        None,
        "--folder",
        help=(
            "Folder of query images. When provided, all supported images in the"
            " folder are used as queries."
        ),
    ),
    top_k: Optional[int] = typer.Option(
        None, "--top-k", help="Number of neighbors to retrieve per query."
    ),
    output_folder_for_results: Optional[Path] = typer.Option(
        None,
        "--output-folder",
        help="Override search_task.output_folder_for_results.",
    ),
    bruteforce_db_folder: Optional[Path] = typer.Option(
        None,
        "--bruteforce-db-folder",
        help="Folder used for brute-force fallback when index is unavailable.",
    ),
    csv_schema: CsvSchema = typer.Option(
        CsvSchema.tif_compat,
        "--csv-schema",
        help="CSV output schema: 'tif_compat' (default) or 'image_only'.",
    ),
) -> None:
    """Run image-only similarity search using configured vector DB or brute-force.

    The function loads configuration, initializes the feature extractor and the
    selected vector database (when available), and then executes a per-image
    similarity search. It writes per-query JSON summaries and a consolidated
    CSV. All file outputs respect the flags under ``search_task`` in the
    configuration.

    Args:
        config_path: Path to the YAML configuration file.
        query: Optional path to a single image to search for similar images.
        folder: Optional directory containing multiple images to process.
        top_k: Optional override for the number of neighbors to retrieve.
        output_folder_for_results: Optional override for run output folder.
        bruteforce_db_folder: Optional override for brute-force DB folder.

    Returns:
        None

    Raises:
        typer.Exit: When required inputs are missing or an unrecoverable error
            occurs (e.g., provider unavailable and fallback disabled).
    """
    # Load env and logging early. They must not abort the CLI on failure.
    try:
        _ = _load_project_dotenv()
    except Exception:  # noqa: BLE001
        pass
    try:
        _ = setup_logging()
    except Exception:  # noqa: BLE001
        pass

    project_root: Path = Path(__file__).resolve().parent
    config: Dict[str, Any] = load_and_merge_configs(config_path)

    # Apply simple overrides to config in-memory (non-destructive to disk).
    if top_k is not None:
        config.setdefault("search_task", {}).setdefault("top_k", int(top_k))
    if output_folder_for_results is not None:
        config.setdefault("search_task", {})[
            "output_folder_for_results"
        ] = str(output_folder_for_results)
    if bruteforce_db_folder is not None:
        config.setdefault("search_task", {})[
            "bruteforce_db_folder"
        ] = str(bruteforce_db_folder)

    # Resolve locations and run root
    search_task: Dict[str, Any] = config.get("search_task", {}) or {}
    base_output_folder: Optional[Path] = _resolve_path(
        search_task.get("output_folder_for_results"), project_root
    )
    if not base_output_folder:
        typer.echo(
            "Error: search_task.output_folder_for_results is not configured in YAML or via --output-folder.",
            err=True,
        )
        raise typer.Exit(code=2)

    from datetime import datetime

    run_identifier: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    requesting_username: str = detect_requesting_username()
    run_type: str = "img_runs"
    save_outputs_to_folder: bool = bool(search_task.get("save_outputs_to_folder", True))
    create_new_subfolder: bool = bool(search_task.get("new_query_new_subfolder", True))
    run_root: Path = (
        base_output_folder / requesting_username / run_type / run_identifier
        if create_new_subfolder
        else base_output_folder
    )
    if save_outputs_to_folder:
        run_root.mkdir(parents=True, exist_ok=True)

    # Prepare query list
    query_images: List[Path] = []
    if query and query.is_file():
        query_images = [query.resolve()]
    elif folder and folder.exists() and folder.is_dir():
        query_images = _collect_query_images(folder.resolve())
    else:
        typer.echo(
            "Error: Provide --query <image> or --folder <dir> containing images.",
            err=True,
        )
        raise typer.Exit(code=2)

    # Initialize feature extractor
    feature_extractor: FeatureExtractor = FeatureExtractor(
        config.get("feature_extractor", {}), project_root
    )

    # Initialize vector DB (FAISS or Qdrant) if available
    db_conf: Dict[str, Any] = config.get("vector_database", {}) or {}
    provider: str = str(db_conf.get("provider", "faiss")).lower()
    faiss_conf: Dict[str, Any] = db_conf.get("faiss", {}) or {}
    qdrant_conf: Dict[str, Any] = db_conf.get("qdrant", {}) or {}
    allow_fallback: bool = bool(db_conf.get("allow_fallback", True))

    vdb: Optional[Any] = None
    final_index_identifier: Optional[str] = None
    indexed_item_count: Optional[int] = None

    if provider == "faiss":
        try:
            vdb = FaissIndexManager(
                feature_dim=feature_extractor.feature_dim,
                output_directory=faiss_conf.get(
                    "output_directory", "instance/faiss_indices"
                ),
                filename_stem=faiss_conf.get("filename_stem", "faiss_collection"),
                index_type=faiss_conf.get("index_type", "flat"),
                model_name=feature_extractor.model_name,
                faiss_config=faiss_conf,
                project_root_path=project_root,
            )
            loaded_ok: bool = vdb.load_index()
            if loaded_ok and vdb.is_index_loaded_and_ready():
                indexed_item_count = vdb.get_total_indexed_items()
                final_index_identifier = (
                    f"FAISS sharded index: {len(vdb.index_identifiers)} shards (first: {vdb.index_identifiers[0]})"
                    if hasattr(vdb, 'index_identifiers') and len(getattr(vdb, 'index_identifiers', [])) > 1
                    else str(vdb.index_path.resolve())
                )
            else:
                if allow_fallback:
                    vdb = None
                else:
                    typer.echo(
                        "Error: FAISS index is not ready and fallback is disabled.",
                        err=True,
                    )
                    raise typer.Exit(code=1)
        except Exception as e_f:  # noqa: BLE001
            if allow_fallback:
                vdb = None
            else:
                typer.echo(
                    f"Error: Failed to initialize FAISS manager: {e_f}", err=True
                )
                raise typer.Exit(code=1)
    elif provider == "qdrant":
        if QdrantManager is None:
            if not allow_fallback:
                typer.echo(
                    "Error: Qdrant provider requested but unavailable and fallback is disabled.",
                    err=True,
                )
                raise typer.Exit(code=1)
            vdb = None
        else:
            try:
                vdb_q: Any = QdrantManager(
                    feature_dim=feature_extractor.feature_dim,
                    collection_name_stem=qdrant_conf.get(
                        "collection_name_stem", "image_similarity_collection"
                    ),
                    model_name=feature_extractor.model_name,
                    qdrant_config=qdrant_conf,
                    project_root_path=project_root,
                )
                vdb_ready: bool = vdb_q.load_index() and vdb_q.is_index_loaded_and_ready()
                if vdb_ready:
                    vdb = vdb_q
                    indexed_item_count = vdb_q.get_total_indexed_items()
                    if qdrant_conf.get("location"):
                        loc: str = str(qdrant_conf.get("location"))
                        mode_desc: str = f"embedded:{loc}"
                    else:
                        host: str = str(qdrant_conf.get("host", "localhost"))
                        port: int = int(qdrant_conf.get("port", 6333))
                        mode_desc = f"server:{host}:{port}"
                    final_index_identifier = (
                        f"Qdrant sharded collections: {len(vdb_q.collection_names)} (first: {vdb_q.collection_names[0]}) ({mode_desc})"
                        if hasattr(vdb_q, 'collection_names') and len(getattr(vdb_q, 'collection_names', [])) > 1
                        else f"qdrant:{vdb_q.collection_name} ({mode_desc})"
                    )
                else:
                    if allow_fallback:
                        vdb = None
                    else:
                        typer.echo(
                            "Error: Qdrant index not ready and fallback is disabled.",
                            err=True,
                        )
                        raise typer.Exit(code=1)
            except Exception as e_q:  # noqa: BLE001
                if allow_fallback:
                    vdb = None
                else:
                    typer.echo(
                        f"Error: Failed to initialize Qdrant manager: {e_q}",
                        err=True,
                    )
                    raise typer.Exit(code=1)
    elif provider == "bruteforce":
        vdb = None
    else:
        typer.echo(
            f"Error: Unsupported provider '{provider}'. Use faiss|qdrant|bruteforce.",
            err=True,
        )
        raise typer.Exit(code=1)

    # When index is unavailable and fallback is allowed, explicitly switch provider to 'bruteforce'
    if vdb is None and allow_fallback:
        provider = "bruteforce"
        try:
            typer.echo("Index not ready. Falling back to brute-force search.", err=False)
        except Exception:
            # Be silent if echo fails; fallback still proceeds
            pass

    # Resolve brute-force DB folder when needed
    bf_db_folder: Optional[str] = (
        search_task.get("bruteforce_db_folder")
        or config.get("indexing_task", {}).get("image_folder_to_index")
    )
    bruteforce_db_folder_path: Optional[Path] = (
        _resolve_path(bf_db_folder, project_root) if bf_db_folder else None
    )

    # Initialize searcher
    searcher: ImageSimilaritySearcher = ImageSimilaritySearcher(
        feature_extractor, vdb
    )

    # Run per-query and collect CSV-like results
    per_query_results: List[Dict[str, Any]] = []
    per_query_timestamps: Dict[str, str] = {}
    per_query_meta: Dict[str, Dict[str, Any]] = {}

    for qpath in query_images:
        # Per-query output folder selection mirrors search_task settings.
        create_per_query_subfolders: bool = bool(
            search_task.get("create_per_query_subfolders_for_tif", True)
        )
        per_query_folder: Path = (
            run_root / qpath.name if create_per_query_subfolders else run_root
        )
        try:
            results, method_used, model_name_used, elapsed = searcher.search_similar_images(
                query_image_path=qpath,
                top_k=int(search_task.get("top_k", 5)),
                db_folder_for_bruteforce=(
                    bruteforce_db_folder_path if vdb is None else None
                ),
                bruteforce_batch_size=int(search_task.get("bruteforce_batch_size", 32)),
                ivf_nprobe_search=(db_conf.get("faiss", {}) or {}).get(
                    "ivf_nprobe_search"
                ),
                hnsw_efsearch_search=(db_conf.get("faiss", {}) or {}).get(
                    "hnsw_efsearch_search"
                ),
            )
        except FileNotFoundError as e_nf:
            typer.echo(f"Warning: {e_nf}")
            continue
        except Exception as e:  # noqa: BLE001
            typer.echo(f"Warning: search failed for {qpath.name}: {e}")
            continue

        # Round scores and elapsed time to 4 decimals for cleaner outputs
        rounded_results: List[tuple[str, float]] = [
            (str(p), round(float(s), 4)) for (p, s) in results
        ]
        elapsed_rounded: float = round(float(elapsed), 4)

        # Persist JSON summary and optional copies, using the shared utility
        per_query_meta[qpath.name] = {
            "elapsed_seconds": elapsed_rounded,
            "method": str(method_used),
            "model": str(model_name_used),
        }
        _, json_path, summary_payload, _ = save_similar_images_to_folder(
            search_results=rounded_results,
            output_folder=per_query_folder if save_outputs_to_folder else run_root,
            query_image_path=qpath,
            config_for_summary=config,
            search_method_actually_used=method_used,
            model_name_used=model_name_used,
            total_search_time_seconds=elapsed_rounded,
            json_filename=JSON_SUMMARY_DEFAULT_FILENAME,
        )
        # json_path and summary_payload are intentionally not used further here.

        # Build per-query entry shaped like TIF per-query for CSV reuse
        top_docs: List[Dict[str, Any]] = [
            {"document": p, "score": s} for (p, s) in rounded_results
        ]
        per_query_results.append(
            {
                "query_document": qpath.name,
                "matched_query_document": qpath.name,
                "num_query_photos": 1,
                "top_docs": top_docs,
                "elapsed_seconds": elapsed_rounded,
                "top_k_used": int(search_task.get("top_k", 5)),
                "top_doc_used": len(top_docs),
                "aggregation_strategy_used": "max",
                "threshold_match_count": 0,
            }
        )
        per_query_timestamps[qpath.name] = datetime.now().isoformat()

    # Export consolidated per-query CSV (image-only)
    csv_path: Optional[Path] = None
    try:
        # Import lazily to avoid overhead when no results
        from core_engine.image_similarity_system.persistence import (
            write_tif_per_query_results_csv,
        )

        if per_query_results and save_outputs_to_folder:
            top_k_used: int = int(search_task.get("top_k", 5))
            if csv_schema == CsvSchema.image_only:
                csv_path = _write_image_only_csv(
                    export_dir=run_root,
                    run_id=run_identifier,
                    user=requesting_username,
                    per_query_results=per_query_results,
                    per_query_timestamps=per_query_timestamps,
                    per_query_meta=per_query_meta,
                    top_k=top_k_used,
                    provider=provider,
                    index_identifier=final_index_identifier,
                    filename_stem="image_only_per_query",
                )
            else:
                removal_cols: List[str] = [
                    "sim_img_check",
                    "image_authenticity",
                    "fraud_doc_probability",
                    "global_top_docs",
                ]
                csv_path = write_tif_per_query_results_csv(
                    per_query_results=per_query_results,
                    export_dir=run_root,
                    run_id=run_identifier,
                    user=requesting_username,
                    per_query_sim_checks=None,
                    per_query_timestamps=per_query_timestamps,
                    filename_stem="postgres_export_img_per_query",
                    global_top_docs=[],
                    remove_columns=removal_cols,
                )
    except Exception as e_csv:  # noqa: BLE001
        typer.echo(f"Warning: failed to write consolidated CSV: {e_csv}")

    # Emit console summary
    typer.echo("\n=== Image Similarity Tester Results ===")
    if save_outputs_to_folder:
        typer.echo(f"Run output folder: {run_root}")
        if csv_path:
            typer.echo(f"Per-query CSV: {csv_path}")
    else:
        typer.echo(
            "Filesystem outputs are disabled (search_task.save_outputs_to_folder=false)"
        )


@app.command("build-image-index")
def build_image_index(
    config_path: Path = typer.Option(
        Path("configs/image_similarity_config.yaml"), help="Path to YAML config."
    ),
    folder: Optional[Path] = typer.Option(
        None,
        "--folder",
        help=(
            "Folder of database images to index (jpg/png/...). If omitted, the"
            " value from indexing_task.image_folder_to_index is used."
        ),
    ),
    engine: Optional[str] = typer.Option(
        None, "--engine", help="Vector DB provider override: faiss|qdrant|bruteforce."
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", help="Feature extraction batch size for indexing."
    ),
    force_rebuild_index: Optional[bool] = typer.Option(
        None, "--force-rebuild-index/--no-force-rebuild-index", help="Clear and rebuild index."
    ),
    scan_subfolders: Optional[bool] = typer.Option(
        None, "--scan-subfolders/--no-scan-subfolders", help="Recursively scan image folder."
    ),
) -> None:
    """Build an image index from an image folder using FAISS or Qdrant.

    This command builds or updates a vector index from a folder of database
    images according to the provider specified in the configuration (or via
    the ``--engine`` override). For FAISS, IVF/HNSW parameters are honored
    from the config. For Qdrant, it can rebuild a collection when configured.

    Args:
        config_path: Path to the YAML configuration file.
        folder: Optional path to the image folder to index. If omitted, uses
            indexing_task.image_folder_to_index from the configuration.
        engine: Optional override for vector database provider (faiss|qdrant|bruteforce).
        batch_size: Optional batch size for feature extraction during indexing.
        force_rebuild_index: Optional flag to clear and rebuild (FAISS) or
            force recreation (Qdrant via config force_recreate_collection).
        scan_subfolders: Optional flag to recursively scan the image folder.

    Returns:
        None

    Raises:
        typer.Exit: When configuration is incomplete, the image folder is
            missing, or the provider is unsupported with no fallback option.
    """
    # Load env and logging early; failures must not abort execution.
    try:
        _ = _load_project_dotenv()
    except Exception:  # noqa: BLE001
        pass
    try:
        _ = setup_logging()
    except Exception:  # noqa: BLE001
        pass

    project_root: Path = Path(__file__).resolve().parent
    config: Dict[str, Any] = load_and_merge_configs(config_path)

    # Resolve and apply overrides
    if engine is not None:
        config.setdefault("vector_database", {})["provider"] = str(engine)
    it_cfg: Dict[str, Any] = config.setdefault("indexing_task", {})
    if batch_size is not None:
        it_cfg["batch_size"] = int(batch_size)
    if force_rebuild_index is not None:
        it_cfg["force_rebuild_index"] = bool(force_rebuild_index)
    if scan_subfolders is not None:
        it_cfg["scan_for_database_subfolders"] = bool(scan_subfolders)

    # Determine image folder
    img_folder: Optional[Path] = folder
    if img_folder is None:
        cfg_folder_str: Optional[str] = it_cfg.get("image_folder_to_index")
        img_folder = _resolve_path(cfg_folder_str, project_root) if cfg_folder_str else None
    if not (img_folder and img_folder.exists() and img_folder.is_dir()):
        typer.echo(
            "Error: Image folder for indexing is not provided or invalid. Use --folder or set indexing_task.image_folder_to_index.",
            err=True,
        )
        raise typer.Exit(code=2)

    # Initialize feature extractor
    feature_extractor: FeatureExtractor = FeatureExtractor(
        config.get("feature_extractor", {}), project_root
    )

    # Provider selection
    db_conf: Dict[str, Any] = config.get("vector_database", {}) or {}
    provider: str = str(db_conf.get("provider", "faiss")).lower()

    if provider == "faiss":
        faiss_conf: Dict[str, Any] = db_conf.get("faiss", {}) or {}
        vdb: FaissIndexManager = FaissIndexManager(
            feature_dim=feature_extractor.feature_dim,
            output_directory=faiss_conf.get("output_directory", "instance/faiss_indices"),
            filename_stem=faiss_conf.get("filename_stem", "faiss_collection"),
            index_type=faiss_conf.get("index_type", "flat"),
            model_name=feature_extractor.model_name,
            faiss_config=faiss_conf,
            project_root_path=project_root,
        )
        exit_code_build: int = vdb.build_index_from_folder(
            feature_extractor=feature_extractor,
            image_folder=str(img_folder),
            batch_size=int(it_cfg.get("batch_size", 32)),
            force_rebuild=bool(it_cfg.get("force_rebuild_index", False)),
            scan_subfolders=bool(it_cfg.get("scan_for_database_subfolders", False)),
            ivf_train_samples_ratio=float(it_cfg.get("ivf_train_samples_ratio", 0.1)),
            ivf_train_samples_max=int(it_cfg.get("ivf_train_samples_max", 50000)),
        )
        if exit_code_build == 0:
            ident = (
                f"FAISS sharded index: {len(vdb.index_identifiers)} shards (first: {vdb.index_identifiers[0]})"
                if hasattr(vdb, 'index_identifiers') and len(getattr(vdb, 'index_identifiers', [])) > 1
                else str(vdb.index_path)
            )
            typer.echo(
                f"FAISS index built successfully at: {ident} (items={vdb.get_total_indexed_items()})"
            )
        else:
            typer.echo("Error: FAISS index build failed.", err=True)
            raise typer.Exit(code=1)

    elif provider == "qdrant":
        if QdrantManager is None:
            typer.echo(
                "Error: Qdrant provider requested but not available in this environment.",
                err=True,
            )
            raise typer.Exit(code=1)
        qdrant_conf: Dict[str, Any] = db_conf.get("qdrant", {}) or {}
        vdb_q: Any = QdrantManager(
            feature_dim=feature_extractor.feature_dim,
            collection_name_stem=qdrant_conf.get(
                "collection_name_stem", "image_similarity_collection"
            ),
            model_name=feature_extractor.model_name,
            qdrant_config=qdrant_conf,
            project_root_path=project_root,
        )
        # Optional force recreate (collection clear) governed by config
        try:
            if bool(qdrant_conf.get("force_recreate_collection", False)):
                vdb_q.clear_index()
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass
        exit_code_build: int = vdb_q.build_index_from_folder(
            feature_extractor=feature_extractor,
            image_folder=str(img_folder),
            batch_size=int(it_cfg.get("batch_size", 32)),
            force_rebuild=False,  # handled via clear_index above
            scan_subfolders=bool(it_cfg.get("scan_for_database_subfolders", False)),
        )
        if exit_code_build == 0:
            total_items: int = vdb_q.get_total_indexed_items()
            if qdrant_conf.get("location"):
                loc: str = str(qdrant_conf.get("location"))
                mode_desc: str = f"embedded:{loc}"
            else:
                host: str = str(qdrant_conf.get("host", "localhost"))
                port: int = int(qdrant_conf.get("port", 6333))
                mode_desc = f"server:{host}:{port}"
            ident = (
                f"Qdrant sharded collections: {len(vdb_q.collection_names)} (first: {vdb_q.collection_names[0]})"
                if hasattr(vdb_q, 'collection_names') and len(getattr(vdb_q, 'collection_names', [])) > 1
                else vdb_q.collection_name
            )
            typer.echo(
                f"Qdrant index built successfully: collection={ident} (mode={mode_desc}, items={total_items})"
            )
        else:
            typer.echo("Error: Qdrant index build failed.", err=True)
            raise typer.Exit(code=1)

    elif provider == "bruteforce":
        typer.echo(
            "Provider 'bruteforce' selected. No index is built for brute-force mode.",
            err=False,
        )
    else:
        typer.echo(
            f"Error: Unsupported provider '{provider}'. Use faiss|qdrant|bruteforce.",
            err=True,
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
