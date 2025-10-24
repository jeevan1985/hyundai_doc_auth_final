# DOWNGRADE (Professional → Basic)

This guide removes Professional features and returns the codebase to the Basic Edition. The Basic Edition excludes advanced features (e.g., Qdrant, professional risk scoring) and must run with FAISS/Bruteforce only.

Audience: Senior engineers, build/release owners

Safety
- The Basic Edition uses optional imports and fallbacks. Removing Pro modules is enough to disable Pro without touching shared core logic.
- Roll-forward to Pro is covered in UPGRADE.md.

1) Identify and remove Professional modules
Remove the following files if present:

- hyundai_document_authenticator/core_engine/image_similarity_system/qdrant_manager.py
- hyundai_document_authenticator/core_engine/image_similarity_system/risk_scoring.py

Windows PowerShell
- Remove-Item -Force "hyundai_document_authenticator/core_engine/image_similarity_system/qdrant_manager.py","hyundai_document_authenticator/core_engine/image_similarity_system/risk_scoring.py"

Windows CMD
- del /Q hyundai_document_authenticator\core_engine\image_similarity_system\qdrant_manager.py
- del /Q hyundai_document_authenticator\core_engine\image_similarity_system\risk_scoring.py

Linux/macOS Bash
- rm -f hyundai_document_authenticator/core_engine/image_similarity_system/qdrant_manager.py
- rm -f hyundai_document_authenticator/core_engine/image_similarity_system/risk_scoring.py

2) Provider configuration (YAML)
Update your YAML to use Basic providers only:

```yaml
vector_database:
  provider: faiss  # or bruteforce
  allow_fallback: true
  fallback_choice: transient
  build_index_on_load_failure: true
  faiss:
    output_directory: instance/faiss_indices
    filename_stem: faiss_collection
    index_type: flat  # flat|ivf|hnsw
```

Remove or ignore any qdrant: sections.

3) CLI behavior (optional)
The Basic Edition CLI is already configured to accept only faiss|bruteforce for provider help and validation. If you previously restored a Pro CLI overlay, revert to the Basic version of:

- hyundai_document_authenticator/doc_image_verifier.py (provider help and validate-config enum should be Basic-only)

4) Dependencies
Optional: You can keep pro dependencies installed (e.g., qdrant-client). They will not be used by Basic logic. To minimize footprint you may uninstall them:

Windows PowerShell / CMD
- python -m pip uninstall -y qdrant-client

Linux/macOS Bash
- python3 -m pip uninstall -y qdrant-client

5) Documentation and training materials
- Basic Edition docs are already stripped of Pro references.
- If you had added Pro-only guides internally, remove or archive them.

6) Verification (Basic Edition)
A) Unit tests
- python run_tests.py

B) Build FAISS index
Windows PowerShell / CMD
- python hyundai_document_authenticator\doc_image_verifier.py build-image-index --engine faiss --folder .\hyundai_document_authenticator\data_real

Linux/macOS Bash
- python3 hyundai_document_authenticator/doc_image_verifier.py build-image-index --engine faiss --folder ./hyundai_document_authenticator/data_real

C) TIF batch search
Windows PowerShell / CMD
- python hyundai_document_authenticator\doc_image_verifier.py search-doc --folder .\hyundai_document_authenticator\data_real

Linux/macOS Bash
- python3 hyundai_document_authenticator/doc_image_verifier.py search-doc --folder ./hyundai_document_authenticator/data_real

Expected: Runs successfully without any Pro modules present and without qdrant-client installed.

7) Result GUI (optional)
- The GUI loads .env from the repo root by default. No changes needed for Basic mode.
- For CSV-only runs, set use_csv: true and csv_path in external/result_gui/config.yaml.

Troubleshooting
- Import errors mentioning qdrant_manager or risk_scoring
  - Ensure those files are removed.
- CLI validate-config complains about provider=qdrant
  - Switch provider to faiss or bruteforce.
- Embedded index empty
  - Enable build_index_on_load_failure: true, or build the index first using build-image-index.

Rollback to Pro
- See UPGRADE.md to restore the Pro overlay files, dependencies, and optional CLI validation/help.
