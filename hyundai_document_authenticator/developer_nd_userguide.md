# JK Image Similarity System — Complete User & Developer Guide

This document is a go-to guide for everyone:
- Novices who have never used this repository before
- Operators and end-users who run searches and pipelines
- Developers who will maintain and extend the codebase

It explains the what, why, and how of each component, with copy-paste commands, annotated examples, and deep references to the code.

-----

## Quick Orientation (For Absolute Beginners)
- Images and TIFs: We compare pictures using numeric vectors (embeddings). For multi-page TIF files, we first detect which pages contain relevant text (OCR), then extract photos from those pages for similarity search.
- Feature extractor: A deep learning model converts each image into a vector. Similar images have similar vectors.
- Vector database: Stores vectors and returns nearest neighbors fast. Here we use FAISS (local files). Qdrant is also supported.
- OCR: Optical Character Recognition (PaddleOCR/EasyOCR/Tesseract) reads text on TIF pages. You can limit OCR to zones (top/bottom/center slices) for speed and reliability — called zonal OCR.
- YOLO vs BBox: We can detect photos on a TIF page via a trained YOLO model, or crop fixed bounding boxes from a page.
- Workflows: The CLI and API orchestrate a few main tasks:
  - Build an index from images
  - Search with a single image or a folder of images
  - Build an index from TIFs (OCR → photos → index)
  - Search using TIFs (OCR → photos → query → aggregate per document)

-----

## What You’ll Do Most Often
- Prepare a Conda environment using the provided environment.yml
- Configure configs/image_similarity_config.yaml for your dataset and model
- Build a FAISS index from a folder of images
- Run a search with a query image or with TIF documents
- Optionally, start the API server and use the client CLI or GUI

-----

## Table of Contents
- [JK Image Similarity System — Complete User \& Developer Guide](#jk-image-similarity-system--complete-user--developer-guide)
  - [Quick Orientation (For Absolute Beginners)](#quick-orientation-for-absolute-beginners)
  - [What You’ll Do Most Often](#what-youll-do-most-often)
  - [Table of Contents](#table-of-contents)
  - [Overview and Architecture](#overview-and-architecture)
  - [Project Structure (Key Paths)](#project-structure-key-paths)
  - [Setup and Environments (Conda)](#setup-and-environments-conda)
  - [Docker and Compose](#docker-and-compose)
  - [Configuration Deep Dive](#configuration-deep-dive)
  - [Zonal OCR (Detailed)](#zonal-ocr-detailed)
  - [Key-Driven TIF Input (Key Input Helper)](#key-driven-tif-input-key-input-helper)
  - [Core CLI (find\_sim\_images.py)](#core-cli-find_sim_imagespy)
  - [API Client CLI (find\_sim\_images\_api.py)](#api-client-cli-find_sim_images_apipy)
  - [HTTP API](#http-api)
  - [GUI Application](#gui-application)
  - [Standalone Utilities](#standalone-utilities)
  - [Datastores and Persistence](#datastores-and-persistence)
  - [Outputs, Logging, and Folders](#outputs-logging-and-folders)
  - [Production Checklist and Backups](#production-checklist-and-backups)
  - [Troubleshooting (Common Pitfalls)](#troubleshooting-common-pitfalls)
  - [Developer Reference](#developer-reference)
  - [Appendix A — Cookbook](#appendix-a--cookbook)
  - [Appendix B — Known oddities](#appendix-b--known-oddities)

-----

## Overview and Architecture
- Core engine: core_engine/image_similarity_system/
- Vendored libs: external/
  - external/tif_searcher: OCR-backed TIF text search
  - external/photo_extractor: YOLO and bbox cropping for photos
- CLIs:
  - find_sim_images.py (main local workflows)
  - find_sim_images_api.py (client for HTTP API)
  - photo_extractor.py, tif_search.py (standalone utilities)
- API server: api_server/ (Flask app, user and API key management)
- GUI: gui_app/ (Flask web client for API)
- Configs: configs/ (engine), api_server/configs/ (API)

Data flow:
1) Build index: Images → Feature vectors → FAISS/Qdrant index
2) Search (single or batch): Query → Feature → Vector DB → Top-K similar
3) TIF → OCR find pages → Extract photos (YOLO/BBox) → either index or query
4) TIF batch search: For each photo’s top-K, aggregate scores per parent document

Document score aggregation (TIF search): default is max — the highest per-photo match becomes the document’s score. You can choose sum or mean.

-----

## Project Structure (Key Paths)
- core_engine/image_similarity_system/
  - workflow.py — orchestrates workflows (build, search, TIF flows)
  - feature_extractor.py — loads TorchVision/HF models, extracts embeddings
  - faiss_manager.py — FAISS index build/load/search
  - searcher.py — search orchestration with brute-force fallback
  - config_loader.py — robust config load/merge/save
  - utils.py — helpers for saving results and DB persistence
- external/
  - tif_searcher/ — TifTextSearcher (OCR)
  - photo_extractor/ — PhotoExtractor (YOLO detection; bbox cropping)
- find_sim_images.py — Main local CLI (Typer)
- find_sim_images_api.py — HTTP API client CLI
- photo_extractor.py — Standalone photo extraction utility
- tif_search.py — Standalone OCR page search utility
- api_server/
  - app/ — Flask API, auth, models
  - configs/api_config.yaml — API defaults, DB URIs
  - run_api_server.py — local server runner
  - manage_api_users.py — user and API key management
- gui_app/
  - app.py — GUI development runner
  - web_app/ — GUI app factory and blueprints
- Docker_for_airgapped/
  - Dockerfiles_Ubuntu_conda/ — Dockerfile.conda, docker-compose.*.yaml, environment.yml (and cpu/dev variants)
- configs/
  - image_similarity_config.yaml — Main engine config (copy from template if needed)

-----

## Setup and Environments (Conda)
Option A: Local root environment.yml
- conda env create -f environment.yml
- conda activate image-similarity-env
- python smoke_test.py

Option B: Docker-aligned Conda files (recommended for parity)
- GPU env file: Docker_for_airgapped/Dockerfiles_Ubuntu_conda/environment.yml
- CPU env file: Docker_for_airgapped/Dockerfiles_Ubuntu_conda/environment.cpu.yml

Example:
- conda env create -f Docker_for_airgapped/Dockerfiles_Ubuntu_conda/environment.yml
- conda activate image-similarity-env
- python smoke_test.py

Notes
- Uses Python 3.10 with numpy 1.26.4 (pinned to prevent NumPy 2.x compatibility issues)
- PyTorch 2.0.1 with CUDA 11.7 support for GPU acceleration
- Windows: Prefer running scripts directly rather than complex command-line JSON quoting
- Optional: install_from_env_yml.py sequentially installs and verifies imports, auto-commenting failing lines in YAML to iterate fixes

-----

## Docker and Compose
Build GPU image from project root:
- docker build -f Docker_for_airgapped/Dockerfiles_Ubuntu_conda/Dockerfile.conda -t img-sim:gpu .
- docker run --rm --gpus all -p 5001:5001 img-sim:gpu

CPU variants:
- Use Docker_for_airgapped/Dockerfiles_Ubuntu_conda/environment.cpu.yml in a Dockerfile variant or alternate compose profile

Compose profiles (common): postgres, qdrant, flask_api
- docker compose -f Docker_for_airgapped/Dockerfiles_Ubuntu_conda/docker-compose.conda.yaml --profile postgres --profile qdrant --profile flask_api up -d
- docker compose -f Docker_for_airgapped/Dockerfiles_Ubuntu_conda/docker-compose.conda.yaml logs -f flask_api

Entrypoint modes depend on the compose file; typical include: flask-api (gunicorn), cli. Paths inside containers are adjusted by the compose entrypoint and envs.

-----

## Configuration Deep Dive
Primary engine config: configs/image_similarity_config.yaml

Key sections and examples:
- feature_extractor
  - model_name: resnet (TorchVision) or HF model in constants
  - pretrained_model_path_or_id: local .pth/.pt or HF repo ID
  - enable_feature_dimension_fixing: true/false (auto-corrects feature_dimension in config)
- vector_database
  - provider: faiss | qdrant | bruteforce
  - allow_fallback: true|false
  - faiss:
    - output_directory: instance/faiss_indices
    - filename_stem: faiss_collection
    - index_type: flat | ivf | hnsw
    - ivf_nlist, ivf_nprobe_search
    - hnsw_m, hnsw_ef_construction, hnsw_efsearch_search
  - qdrant:
    - location (embedded) or host/port (server), collection_name_stem, distance_metric
- indexing_task
  - image_folder_to_index: instance/database_images
  - batch_size, force_rebuild_index, scan_for_database_subfolders
  - ivf_train_samples_ratio, ivf_train_samples_max
- search_task
  - query_image_path or batch_query_image_folder_path
  - output_folder_for_results (e.g., instance/search_results)
  - new_query_new_subfolder: true/false
  - top_k
  - bruteforce_db_folder
  - bruteforce_batch_size
  - save_results_to_postgresql: true/false
  - TIF batch settings: input_tif_folder_for_search, top_doc, aggregation_strategy (max|sum|mean)
  - doc_sim_img_check (true/false) and doc_sim_img_check_max_k
- photo_extractor_config
  - photo_extraction_mode: yolo | bbox
  - yolo_object_detection.model_path: models/your_yolo_model.pt
  - yolo_object_detection.inference: confidence_threshold, iou_threshold, imgsz, target_object_names
  - bbox_extraction: bbox_list, bbox_format (xyxy|xywh|cxcywh), normalized (bool)
- searcher_config (optional)
  - Can override TIF searcher parameters (see Zonal OCR)
- results_postgresql
  - database_name, host, port, user, password, table names, debug flags
- jwt_config
  - For GUI JWT sessions (secret_key, algorithm, expiry_hours); SECRET_KEY should be provided via env or API config

API config: api_server/configs/api_config.yaml
- Mirrors many engine keys (search_task, vector_database, etc.)
- user_database: default SQLite URI; consider adjusting to a valid path in this repo
- user_database_postgresql: optional separate PostgreSQL for user/keys
- jwt_config: secret_key and algorithm used for GUI

Path resolution rules
- Relative paths are resolved from the project root (for engine) or from the respective app’s known base
- Prefer absolute paths for Windows when in doubt

-----

## Zonal OCR (Detailed)
Zonal OCR restricts OCR to specific vertical slices of a page to speed processing and reduce false positives.

Zones dictionary examples (as JSON strings or dicts):
- {} → full-page OCR (default)
- {"top": 0.1} → top 10% of page height
- {"bottom": 0.1} → bottom 10%
- {"center": 0.2} → centered slice covering 20% of height
- You can combine keys: {"top": 0.1, "bottom": 0.1}
- Values outside [0,1] are clamped

Using zonal OCR via tif_search.py:
- Windows PowerShell:
  python tif_search.py --tif-path .\docs --search-text "가맹점 실사 사진" --search-location '{"top":0.1,"bottom":0.1}'
- Windows CMD (escape quotes):
  python tif_search.py --tif-path .\docs --search-text "가맹점 실사 사진" --search-location "{\"top\":0.1,\"bottom\":0.1}"

Zonal OCR via workflows
- Set searcher_config.search_location in configs/image_similarity_config.yaml, or pass via CLI overrides where supported
- The internal TifTextSearcher supports language, OCR backend choice, normalization flags, and offline model directories

Debugging OCR
- tif_search.py --create-csv-report true and recognized_text_debug true will emit a CSV containing raw and normalized text snippets per page-zone and OCR timings (ms)

-----

## Key-Driven TIF Input (Key Input Helper)
Purpose: Enable an alternative input mode where TIF query documents are driven by a key table (CSV/Excel/JSON). The helper module resolves filenames to actual TIF files in scalable batches (local folders, API, or PostgreSQL/EDB) and then invokes the existing TIF batch pipeline unchanged.

Location
- external/key_input/
  - key_input_orchestrator.py — loader, local/API/DB fetchers, name-mapping, and per-batch handoff
  - key_input_config.yaml — configuration for key file, name mapping, and data source

Enablement
- configs/image_similarity_config.yaml (optional section)
```
input_mode:
  doc_input_start: "key"
  key_input_config_path: "external/key_input/key_input_config.yaml"
```
- If input_mode is missing or doc_input_start != "key", behavior is unchanged and the helper is not used.

What it does (and does not do)
- Does: stream filenames from a key file, apply filename mapping rules, resolve to TIF/TIFF files via local folders, API, or database, batch them, and for each batch call the existing execute_tif_batch_search_workflow(input_folder_override=temp_batch_dir).
- Does not: change core_engine configurations, endpoints, or pipelines; introduce new pipeline knobs; modify find_sim_images_api.py behavior unless you choose to delegate there similarly. Key-mode is TIF/TIFF-only by contract; plain images (jpg/jpeg/png) are not ingested.

Configuration (external/key_input/key_input_config.yaml)
- key_input
  - input_table_path: path to the CSV/Excel/JSON key file
  - file_name_column: column name with filenames (default "파일명")
  - format: csv | excel | json | auto
  - json_array_field: when JSON top-level is an object
  - json_records_is_lines: true for NDJSON/JSONL
  - batch_size: batch of filenames per iteration (scalability control)
  - deduplicate, strip_whitespace: cleaning options
- name_mapping
  - enabled: true/false — enable filename transformation and pattern matching
  - tail_len: integer tail length used to split filename as prefix + tail
  - insert_token: token inserted between prefix and tail (e.g., "001")
  - glob_suffix: appended to mapped core when scanning local roots (e.g., "_*.tif")
  - use_rglob_any_depth: scan any depth under roots
  - db_like_template: template to build SQL LIKE pattern (see below)
- data_source
  - mode: local | api | database (database deprecated; prefer api)
  - local: search_roots, recursive, allowed_extensions, resolve_without_extension, stop_on_first_match
  - api:
    - request_mapping.send_mapped_filename: client-side filename mapping toggle
    - persist_downloads: true to persist, false to use transient temp folder
    - transient_download_root: optional base for transient folders
    - response_mapping: image_payload_type (base64|url|binary) and fields
  - database: PostgreSQL/EDB connection and query mapping (deprecated)
    - fetch_mode: path (returns file path) or blob (returns binary to write)
    - query_template: must accept %(file_name)s placeholder
    - path_column/blob_column: select columns from query results
    - blob_temp_dir: optional temp dir for binaries

Name mapping and database LIKE pattern
- The helper supports storage naming conventions that differ from the original key filenames.
- Example rule (as provided): for file_name like N2023100401437THA00001, build "{prefix}001{suffix}" where prefix = file_name[:-5], suffix = file_name[-5:].
- Local search: find files matching mapped_core + glob_suffix (e.g., N...00100001_*.tif), using rglob when enabled.
- Database fallback: if exact equality returns no rows and name mapping is enabled, build a LIKE pattern using db_like_template.
  - db_like_template (default): "{prefix}{insert}{suffix}_%.tif"
  - The code replaces {prefix}, {insert}, {suffix} from the mapped core and retries the query with LIKE.
  - Guard: the fallback LIKE is attempted only if exact query returned no rows, mapping is enabled, and the template contains "%".

Behavior in CLIs
- Local CLI (find_sim_images.py): For search-with-tif, when doc_input_start == "key", the CLI delegates to the helper. The helper prepares per-batch temp folders and calls execute_tif_batch_search_workflow. Output structure and downstream steps are identical to the folder-based flow.
- API client (find_sim_images_api.py): Delegates when doc_input_start == "key" is detected; falls back gracefully otherwise.

Key-mode temp indexing
- data_source.key_mode.use_tmp_for_indexing_input: when true, the orchestrator sets indexing_task.input_tif_folder_for_indexing to the temp batch folder while action includes indexing.

Non-persistent API downloads
- persist_downloads=false with optional transient_download_root saves to a transient directory per batch (e.g., api_transient_<ts>) and cleans up after the batch completes.

Contract
- Key-mode/API/local resolution accept TIF/TIFF documents only. If an API responds with a non-TIF extension, it is normalized to .tif to maintain downstream consistency. Any non-TIF files located by local resolution are skipped with a warning.

Validation and visualization tools
- search_tif_files_with_key.py (project root)
  - Streams the key file, resolves batches via local/API/DB, reports per-batch counts and sample resolved paths, optionally writes a manifest, and optionally triggers run_key_input_pipeline.
  - Examples:
    - Local/API dry-run: `python search_tif_files_with_key.py --key-config external/key_input/key_input_config.yaml --limit-batches 2 --show-samples 10`
    - DB validation: `python search_tif_files_with_key.py --key-config external/key_input/key_input_config.yaml --mode database --db-host 127.0.0.1 --db-name postgres --db-user postgres --limit-batches 1`
    - Manifest + execute pipeline: `python search_tif_files_with_key.py --key-config external/key_input/key_input_config.yaml --manifest-out pipeline_inputs.json --execute-now --main-config configs/image_similarity_config.yaml`
- Key_Driven_TIF_Validation.ipynb (project root)
  - Visual walkthrough to load config, preview filenames, demonstrate mapping, run limited batches for local/API/DB resolution, and optionally run the pipeline.

Removal and backward compatibility
- The module is optional. If removed or disabled (doc_input_start != "key"), all workflows behave exactly as before.

-----

## Core CLI (find_sim_images.py)
Purpose: Unified local runner for image and TIF workflows, backed by YAML configuration with optional CLI overrides.

Command model (subcommands)
- search
  - Single image or batch folder search against an image DB
- build-index
  - Build/update FAISS or Qdrant index from an image folder
- build-index-with-tif
  - TIF pipeline: find pages (optional OCR), extract photos (YOLO or BBox), and build index from the extracted photos
- search-with-tif
  - TIF batch search: OCR text filter → photo extraction → per-photo image search → per-document aggregation

Global option
- --config-path configs/image_similarity_config.yaml
  - All unspecified options are sourced from YAML. CLI values override YAML.

Quick examples
- Build FAISS index from images
  - python find_sim_images.py build-index --image-folder-to-index instance/database_images --engine faiss --faiss-index-type ivf
- Single image search
  - python find_sim_images.py search --query-image path/to/query.jpg --top-k 10
- Batch image search (folder of queries)
  - python find_sim_images.py search --batch-folder tests --top-k 15 --engine faiss
- Build index from TIFs (BBox mode)
  - python find_sim_images.py build-index-with-tif --folder ./data_real --photo-extraction-mode bbox \
    --bbox-list "[[171,236,1480,1100],[171,1168,1480,2032]]" --bbox-format xyxy
- Build index from TIFs (YOLO mode)
  - python find_sim_images.py build-index-with-tif --folder ./data_real --photo-extraction-mode yolo \
    --yolo-model-path trained_model/weights/best.pt --yolo-conf-thresh 0.25 --yolo-iou-thresh 0.45 --yolo-imgsz 640
- TIF batch search (server-side folder path)
  - python find_sim_images.py search-with-tif --folder ./data_real --top-doc 7 --top-k 5 --aggregation-strategy max \
    --photo-extraction-mode bbox --output-folder instance/search_results

Search options (common)
- --top-k <int>                    Per-query neighbors (image search) or per-photo neighbors (TIF)
- --bruteforce-db-folder <path>    Fallback database folder if vector index is unavailable
- Output behavior: --output-folder, --new-subfolder/--no-new-subfolder,
  --copy-query/--no-copy-query, --copy-similar/--no-copy-similar, --save-summary/--no-save-summary

TIF-specific options
- Aggregation across per-photo scores: --aggregation-strategy max|sum|mean; --top-doc <int>
- PhotoExtractor overrides
  - --photo-extraction-mode yolo|bbox; --photo-extractor-debug/--no-photo-extractor-debug
  - YOLO: --yolo-model-path, --yolo-conf-thresh, --yolo-iou-thresh, --yolo-imgsz,
          --yolo-target-object-names '["photo","image"]'
  - BBox: --bbox-list '[[171,236,1480,1100],[171,1168,1480,2032]]', --bbox-format xyxy|xywh|cxcywh, --bbox-normalized/--no-bbox-normalized
- Optional OCR text search overrides (to filter pages):
  - --search-text, --language, --ocr-backend, --search-mode,
    --allow-normalization/--no-allow-normalization,
    --remove-spaces/--no-remove-spaces,
    --recognized-text-debug/--no-recognized-text-debug,
    --search-location-top <float>, --use-offline-models/--no-use-offline-models,
    --use-angle-cls/--no-use-angle-cls, --use-gpu-for-paddle/--no-use-gpu-for-paddle,
    --paddle-batch-size <int>

Vector DB selection and tuning
- --engine faiss|qdrant|bruteforce
- FAISS: --faiss-index-type flat|ivf|hnsw, --faiss-output-dir <path>, --faiss-filename-stem <str>,
         --faiss-ivf-nlist <int>, --faiss-ivf-nprobe <int>,
         --faiss-hnsw-m <int>, --faiss-hnsw-efc <int>, --faiss-hnsw-efs <int>
- Qdrant (server/embedded): --qdrant-host, --qdrant-port, --qdrant-grpc-port,
         --qdrant-prefer-grpc/--no-qdrant-prefer-grpc, --qdrant-api-key,
         --qdrant-https/--no-qdrant-https, --qdrant-location <folder>,
         --qdrant-collection-stem <str>, --qdrant-distance Cosine|Dot|Euclid,
         --qdrant-quantization/--no-qdrant-quantization,
         --qdrant-on-disk-hnsw/--no-qdrant-on-disk-hnsw,
         --qdrant-force-recreate/--no-qdrant-force-recreate

Outputs
- Results and summaries are written under search_task.output_folder_for_results
- With new_query_new_subfolder=true, a timestamped folder hierarchy is created per run
- TIF runs may produce a run summary JSON and optional PostgreSQL CSV export

Fallback
- If FAISS/Qdrant is missing or incompatible and allow_fallback is true, brute-force search is used when bruteforce_db_folder is configured

Troubleshooting
- Ensure model weights and FAISS paths are valid in YAML
- For BBox mode, provide bbox-list (CLI or YAML); an empty list yields zero crops
- For YOLO mode, specify a valid .pt via --yolo-model-path (or YAML)
- Paths are resolved relative to project root unless absolute

-----

## API Client CLI (find_sim_images_api.py)
Purpose: Control the HTTP API to build indices and perform searches remotely. Unspecified options come from the API’s YAML (api_server/configs/api_config.yaml) and the master engine YAML. CLI overrides take precedence.

Global flags (must be before subcommand)
- --api-base-url http://127.0.0.1:5001/api/v1
- --api-key YOUR_API_KEY
  - Alternatively set environment variables: IMAGE_SIM_API_KEY and API_BASE_URL
  - If --api-key is omitted and env is unset, you will be prompted securely

Authentication header
- The client sends Authorization: Bearer <YOUR_API_KEY>
- Some endpoints require admin role (e.g., building indices)

Subcommands
- build-index
  - Build index from a server-side image folder
  - Example:
    - python find_sim_images_api.py --api-base-url http://127.0.0.1:5001/api/v1 --api-key YOUR_KEY \
      build-index --folder-path instance/database_images --engine faiss --batch-size 32
- search
  - Single local image (upload):
    - python find_sim_images_api.py --api-base-url http://127.0.0.1:5001/api/v1 --api-key YOUR_KEY \
      search --query-image "D:\\some.jpg" --top-k 10 --engine faiss
  - Server-side batch folder:
    - python find_sim_images_api.py --api-base-url http://127.0.0.1:5001/api/v1 --api-key YOUR_KEY \
      search --batch-folder tests --top-k 15 --engine qdrant
- search-with-tif (two modes)
  - Local folder upload (no pre-copy):
    - python find_sim_images_api.py --api-base-url http://127.0.0.1:5001/api/v1 --api-key YOUR_KEY \
      search-with-tif --folder "D:\\test_data" --photo-extraction-mode bbox \
      --bbox-list "[[171,236,1480,1100],[171,1168,1480,2032]]" --bbox-format xyxy \
      --top-doc 7 --top-k 5 --aggregation-strategy max --output-folder instance\\api_search_results
  - Server-side folder (no upload):
    - python find_sim_images_api.py --api-base-url http://127.0.0.1:5001/api/v1 --api-key YOUR_KEY \
      search-with-tif --folder instance\\query_tifs --photo-extraction-mode yolo --yolo-model-path trained_model\\weights\\best.pt \
      --top-doc 7 --top-k 5 --aggregation-strategy max
- build-index-with-tif (two modes)
  - Local folder upload (no pre-copy):
    - python find_sim_images_api.py --api-base-url http://127.0.0.1:5001/api/v1 --api-key YOUR_KEY \
      build-index-with-tif --folder "D:\\test_data" --photo-extraction-mode bbox --batch-size 32 --force-rebuild-index true
  - Server-side folder (no upload):
    - python find_sim_images_api.py --api-base-url http://127.0.0.1:5001/api/v1 --api-key YOUR_KEY \
      build-index-with-tif --folder instance\\query_tifs --photo-extraction-mode yolo --yolo-model-path trained_model\\weights\\best.pt

Important notes
- Global flags must precede the subcommand (e.g., python find_sim_images_api.py --api-base-url ... --api-key ... search-with-tif ...)
- For bbox mode, supply a bbox list or configure it in YAML; an empty list yields zero crops
- For YOLO mode, provide a valid model path (CLI or YAML)
- If you see “output_folder_for_results is not configured”, either set it in api_server/configs/api_config.yaml under search_task, or pass --output-folder on the CLI
- Building indices is admin-only unless security is disabled

Troubleshooting
- If you pass a local --folder path that doesn’t exist, the client assumes it is a server path
- Use quotes around Windows paths with backslashes
- If the server routes were changed, restart the API server; confirm /api/v1/status works with your key

-----

## HTTP API
Run locally
- python api_server/run_api_server.py
- Or Gunicorn:
  gunicorn --bind 0.0.0.0:5001 --workers 4 --preload "api_server.run_api_server:app"

User and key management
- cd api_server
- python manage_api_users.py init_db
- python manage_api_users.py add_user admin --role admin
- python manage_api_users.py gen_key admin --label "Default Key"
- python manage_api_users.py list_users --format json
- python manage_api_users.py list_keys admin --format text

Authentication
- Header required: Authorization: Bearer <API_KEY>
- The server also accepts JWT tokens created by the GUI (Authorization: Bearer <JWT>)

Sample endpoints (depending on branch/routes)
- GET /api/v1/health — returns server status
- POST /api/v1/search — multipart (query_image) or JSON batch; accepts config_overrides
- POST /api/v1/index or /api/v1/index/build — admin-only; triggers indexing

Examples (curl)
- Health:
  curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:5001/api/v1/health
- Single search (multipart):
  curl -X POST \
       -H "Authorization: Bearer YOUR_API_KEY" \
       -F "query_image=@/path/to/query.jpg" \
       -F "config_overrides={\"search_task\":{\"top_k\":10}}" \
       http://localhost:5001/api/v1/search
- Batch search (server folder, JSON):
  curl -X POST \
       -H "Authorization: Bearer YOUR_API_KEY" \
       -H "Content-Type: application/json" \
       -d '{"config_overrides":{"search_task":{"batch_query_image_folder_path":"instance/batch_queries","top_k":5}}}' \
       http://localhost:5001/api/v1/search

Python requests example
- import json, requests
  API = "http://127.0.0.1:5001/api/v1"
  headers = {"Authorization": f"Bearer {"YOUR_KEY"}"}
  files = {"query_image": open("samples/query.jpg", "rb")}
  form = {"config_overrides": json.dumps({"search_task": {"top_k": 10}})}
  r = requests.post(f"{API}/search", files=files, data=form, headers=headers)
  print(r.json())

-----

## GUI Application
Run in development mode
- python gui_app/app.py

Behavior
- Uses the same user database schema as the API
- SECRET_KEY is loaded via env → API config → GUI config → default; change for production
- GUI reads the main engine config path and resolves output folders
- GUI’s upload and database folders are configurable in gui_app/configs/web_app_config.yaml

-----

## Standalone Utilities
photo_extractor.py — YOLO or bbox photo extraction from TIFs
- YOLO example:
  python photo_extractor.py yolo --tif-path path/to/doc_or_folder --all-pages --model-path models/photo_extractor/best.pt --output-dir extracted_crops/yolo
- BBox example:
  python photo_extractor.py bbox --tif-path path/to/doc_or_folder --page-numbers 1 2 --bbox-list "[[50,60,300,260]]" --bbox-format xyxy --normalized false --output-dir extracted_crops/bbox

Notes
- In workflows, set photo_extractor_config.yolo_object_detection.model_path in the engine YAML
- BBox mode does not require a YOLO model

tif_search.py — OCR page detection
- Single file:
  python tif_search.py --tif-path path/to/doc.tif --search-text "Your Title" --output-folder search_results --img-format png
- Directory batch with CSV and zonal OCR:
  python tif_search.py --tif-path path/to/folder --search-text "Your Title" --create-csv-report true --recognized-text-debug true --search-location '{"top":0.1,"bottom":0.1}'

Output naming conventions
- Saved pages: <stem>_page{N}.{ext}
- Extracted photos: <stem>_page{N}_photo{i}.jpg

-----

## Datastores and Persistence
FAISS
- index_type: flat|ivf|hnsw; IVF must be trained
- Tune ivf_nlist, ivf_nprobe_search; HNSW uses hnsw_m and ef parameters
- Index files and path mappings are persisted under faiss.output_directory
- Dimension must match feature_extractor.feature_dim

Qdrant
- Embedded (qdrant.location) or server (host/port); distance metrics supported
- Extend qdrant_manager for new collection parameters

PostgreSQL (optional search result persistence)
- Configure results_postgresql in YAML
- Workflows can save results and optionally export a CSV snapshot if enabled
- TIF batch search also supports a doc_sim_img_check payload (top-K explanatory db images per document)

-----

## Outputs, Logging, and Folders
- API logs: api_server/instance/logs/ (and console)
- GUI logs: gui_app/instance/logs/ by default; configurable via GUI config
- Engine results: search_task.output_folder_for_results
  - With new_query_new_subfolder=true, results go under <output>/<user>/<run_type>/<timestamp>
- TIF batch runs can emit run_summary JSON and optional PostgreSQL CSV export
- Failed key-driven API requests: hyundai_document_authenticator/logs/failed_requests.log (JSON Lines)
  - One line per failed filename request in key mode (HTTP failure, non-JSON, missing image field, URL download error, etc.)
  - Fields: timestamp, requested_name, api_endpoint, status_code, reason, correlation_id (if available), context (optional)
  - Config toggle: logging.enable_failed_key_request_logging (default true)

-----

## Production Checklist and Backups
- Build Docker image(s) and run smoke test
- Configure model weights, mount volumes for data and indices, and engine configs
- Set FLASK_CONFIG=production and logging levels
- For Qdrant: set QDRANT_MODE and credentials
- Verify /health; run a sample search
- Backups:
  - FAISS: backup faiss.output_directory (index and mapping)
  - Qdrant embedded: backup qdrant.location; server: use snapshots or volume backups
  - PostgreSQL: pg_dump/pg_restore; optionally export CSV per run

-----

## Troubleshooting (Common Pitfalls)
- API header mismatch: use Authorization: Bearer <API_KEY>. If using the provided API client as-is, it sends X-API-KEY — update it or use curl/Python examples here.
- FAISS dimension mismatch: enable enable_feature_dimension_fixing or correct feature_dimension to match your model.
- YOLO model path: ensure photo_extractor_config.yolo_object_detection.model_path points to a readable .pt file.
- OCR engines: verify PaddleOCR/EasyOCR/Tesseract are installed; set offline directories if air-gapped.
- Windows JSON quoting: for --search-location and other JSON flags, escape quotes properly (see examples above).
- Brute-force fallback: set search_task.bruteforce_db_folder if FAISS is unavailable but you still need searches.
- Index not loading: paths are resolved relative to project root; verify faiss.output_directory and filename_stem.
- Large batches: reduce batch sizes or image sizes; consider IVF or HNSW for scalability.

-----

## Developer Reference
Code map (Where to change what)
- Workflows/orchestration: core_engine/image_similarity_system/workflow.py
- Feature extraction and transforms: core_engine/image_similarity_system/feature_extractor.py
- FAISS index lifecycle and params: core_engine/image_similarity_system/faiss_manager.py
- Search orchestration and brute-force: core_engine/image_similarity_system/searcher.py
- Config loading/merging/saving: core_engine/image_similarity_system/config_loader.py
- Vendored OCR searcher: external/tif_searcher (TifTextSearcher)
- Vendored photo extractor: external/photo_extractor (PhotoExtractor)
- Local CLI entry: find_sim_images.py
- API client CLI: find_sim_images_api.py
- API server and auth: api_server/run_api_server.py, api_server/app/api/auth.py, api_server/app/models.py
- User/key admin tools: api_server/manage_api_users.py
- GUI app: gui_app/web_app/__init__.py; dev runner: gui_app/app.py

Extending the system
- Add a new model:
  - constants.py → add entry to MODEL_CONFIGS (type=torchvision/hf, input_size, output handler, loader)
  - Ensure feature_extractor supports transforms and output extraction
- Add a new vector provider:
  - Create a manager similar to faiss_manager.py; update workflow init logic
- Change TIF aggregation default:
  - workflow.execute_tif_batch_search_workflow → aggregation_strategy default
- Improve result persistence:
  - core_engine/image_similarity_system/utils.py → DB save functions and CSV export

Style and practices
- Guard clauses, minimal nesting, explicit variable names
- Configuration-driven behavior; keep secrets in env, not in code
- Use logging; avoid print except for CLI status where appropriate

-----

## Appendix A — Cookbook
- Setup environment:
  1) `conda env create -f environment.yml` (creates image-similarity-env with Python 3.10)
  2) `conda activate image-similarity-env`
  3) `python smoke_test.py`
- Index a folder of images (FAISS):
  1) Set indexing_task.image_folder_to_index
  2) python find_sim_images.py --action build_index --config-path configs/image_similarity_config.yaml
- Single image search (FAISS):
  1) Ensure index exists
  2) python find_sim_images.py --action search --config-path configs/image_similarity_config.yaml --query-image path/to/query.jpg
- Build index from TIFs (OCR → photos → index):
  python find_sim_images.py --action build_index_from_tifs --config-path configs/image_similarity_config.yaml --tif-folder path/to/tifs
- Search with TIFs (OCR → photos → search → aggregate):
  python find_sim_images.py --action search_with_tif --config-path configs/image_similarity_config.yaml --tif-folder path/to/tifs --top-k 5 --top-doc 7
- Zonal OCR quick try (standalone):
  python tif_search.py --tif-path .\docs --search-text "가맹점 실사 사진" --search-location '{"top":0.1,"bottom":0.1}'
- Photo extraction (standalone YOLO):
  python photo_extractor.py yolo --tif-path path/to/doc_or_folder --all-pages --model-path models/photo_extractor/best.pt

## Appendix B — Known oddities
- api_server/configs/server_config.py contains a placeholder-like snippet around DATABASE_OUTPUT_DIR_RELATIVE. It does not affect the main workflows documented here. Treat it as a placeholder or disable it if not used by your deployment.

-----

Start here:
1) Create conda environment: `conda env create -f environment.yml` then `conda activate image-similarity-env` and run `python smoke_test.py`
2) Configure configs/image_similarity_config.yaml (model, FAISS paths, folders)
3) Build index from your image database
4) Run a sample search (single image or a small TIF set)
5) Optionally bring up the API, create a user, generate an API key, and verify /health

Once these steps work, you are ready for production and further tuning (IVF/HNSW, zonal OCR, YOLO thresholds, and DB persistence).
