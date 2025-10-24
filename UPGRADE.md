# UPGRADE (Basic → Professional)

This guide enables Professional features on top of the Basic Edition without modifying shared core logic. The Professional overlay reintroduces optional modules and dependencies for advanced vector indexing and risk scoring.

Audience: Senior engineers, build/release owners

Safeguards
- No edits to shared interfaces (vector_db_base.py, workflows) are required.
- Optional imports and provider selection automatically activate Pro behavior when modules are present.
- Rollback is safe (see DOWNGRADE.md).

Professional feature set (restored)
- Qdrant vector database manager (embedded/server) with sharding and maintenance.
- Advanced/professional risk scoring (beyond the fallback neutral implementation).

1) Prerequisites
- Python environment activated (image-similarity-env or equivalent).
- Network access to install pro dependencies (or a local wheelhouse if air‑gapped).

Windows PowerShell
- $PSVersionTable.PSVersion
- python --version

Linux/macOS Bash
- echo $SHELL
- python3 --version

2) Restore the Professional overlay files
Place the following files back into the exact paths (overwrite if present):

- hyundai_document_authenticator/core_engine/image_similarity_system/qdrant_manager.py
- hyundai_document_authenticator/core_engine/image_similarity_system/risk_scoring.py

Notes
- These filenames and import locations are critical. The workflow.py and augmentation_orchestrator.py use try/except imports to auto‑enable Pro when modules exist.
- Do NOT alter vector_db_base.py or public interfaces.

3) Install Professional dependencies
If you do not have them already in the environment, install the Qdrant client (and any other pro dependencies your organization uses):

Windows PowerShell / CMD
- python -m pip install --upgrade qdrant-client

Linux/macOS Bash
- python3 -m pip install --upgrade qdrant-client

Air‑gapped
- Use your internal wheelhouse/index (e.g., pip install --no-index --find-links <local_dir> qdrant-client)

4) (Optional but recommended) Restore Pro-aware CLI validation/help
The Basic Edition CLI shows/validates only faiss|bruteforce providers. This does not block runtime Pro usage (workflows still load Qdrant if configured), but validate-config will complain. If you want CLI validation and --engine help to include qdrant again, restore the Pro variant of:

- hyundai_document_authenticator/doc_image_verifier.py (provider help and validate-config enum)

Alternatively, skip CLI validate-config when provider is qdrant.

5) Re-introduce Pro configuration (Qdrant)
Add (or restore) the qdrant section to your YAML (example values shown). For embedded mode:

```yaml
vector_database:
  provider: qdrant
  allow_fallback: true
  fallback_choice: transient
  build_index_on_load_failure: true
  qdrant:
    location: "instance/qdrant_db"
    unique_location_per_run: true
    # Or server mode (comment-out location and set host/port):
    # host: "localhost"
    # port: 6333
    # grpc_port: 6334
    # prefer_grpc: false
    collection_name_stem: "image_similarity_collection"
    distance_metric: "Cosine"
    enable_quantization: false
    on_disk_hnsw_indexing: true
    force_recreate_collection: false
    partition_capacity: 500000
```

6) Verification
A) Unit tests (Basic-safe; Pro modules present)
- python run_tests.py

B) Build index (Qdrant Embedded example)
- Ensure YAML has provider: qdrant (see above)

Windows PowerShell / CMD
- python hyundai_document_authenticator\doc_image_verifier.py build-image-index --folder .\hyundai_document_authenticator\data_real --engine qdrant

Linux/macOS Bash
- python3 hyundai_document_authenticator/doc_image_verifier.py build-image-index --folder ./hyundai_document_authenticator/data_real --engine qdrant

Expected:
- Logs indicating Qdrant collection creation or discovery and indexed counts.

C) TIF batch search (Pro)
Windows PowerShell / CMD
- python hyundai_document_authenticator\doc_image_verifier.py search-doc --folder .\hyundai_document_authenticator\data_real

Linux/macOS Bash
- python3 hyundai_document_authenticator/doc_image_verifier.py search-doc --folder ./hyundai_document_authenticator/data_real

Expected:
- Results printed with provider identifier set to qdrant:... when collections are ready.

7) Result GUI (if used)
- The GUI now loads .env from repository root automatically.
- For DB-backed mode, ensure POSTGRES_* variables exist in repo-root .env:
  - POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_TABLE_NAME, POSTGRES_APP_USER_TABLE_NAME

Windows PowerShell / CMD
- cd hyundai_document_authenticator\external\result_gui
- python app.py

Linux/macOS Bash
- cd hyundai_document_authenticator/external/result_gui
- python3 app.py

8) Rollback (quick)
- Delete the two professional modules restored in step 2 (qdrant_manager.py, risk_scoring.py).
- Switch provider back to faiss (or bruteforce) in YAML.
- Optional: keep CLI validation strict to Basic (faiss|bruteforce only) or re-apply Basic doc_image_verifier.py.

Troubleshooting
- ImportError: qdrant_client not found
  - Install with pip as shown above or use your internal wheelhouse.
- Provider errors when using validate-config
  - The Basic CLI rejects qdrant; either restore Pro doc_image_verifier.py or skip validate-config for Pro runs.
- Embedded Qdrant locked
  - If location is shared across processes, use unique_location_per_run: true to avoid lock contention in dev/test.

Change safety & expectations
- Professional overlay relies only on file presence and optional imports; no core refactors needed.
- Replacing or removing the two modules is sufficient to toggle Pro on/off, provided dependencies are installed and configs are set.

See also
- DOWNGRADE.md (Professional → Basic)
