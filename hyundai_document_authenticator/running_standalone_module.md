# Running Modules Standalone: Developer Guide

This guide shows how to run and debug each module independently so you can:
- Reuse modules in other projects
- Run fast, focused tests
- Debug specific functionality in isolation

Applies to modules under:
- hyundai_document_authenticator/external
  - image_authenticity_classifier
  - key_input
  - photo_extractor
  - tif_searcher
  - result_gui (runtime UI launcher, optional)
- hyundai_document_authenticator/core_engine
  - image_similarity_system/* (core services used by external wrappers)

All examples are minimal, production-grade, and OS-aware.


## Prerequisites

- OS: Windows 10/11, Linux (Ubuntu tested), macOS
- Shells supported
  - Windows CMD (cmd.exe)
  - Windows PowerShell (powershell.exe)
  - Linux/macOS Bash/Zsh
- Python: 3.9+ available as python (Windows) or python3 (Linux/macOS)
- Project working directory (Windows): d:\\frm_git\\hyundai_doc_auth_final
- Project working directory (Linux/macOS): /path/to/frm_git/hyundai_doc_auth_final
- Recommended: create and activate the conda environment described in Docker_for_airgapped/Dockerfiles_Ubuntu_conda/environment.yml or docs/requirements.txt

Environment activation (PowerShell):
- conda activate <your-env>

Environment activation (CMD):
- conda activate <your-env>

Environment activation (Linux/macOS Bash):
- conda activate <your-env>

Note: If python command is not available, use py -3.9 or the interpreter path of your environment. On Linux/macOS, prefer python3.


## Data and Models Layout

This repository provides default data/model locations used by examples below:
- Input TIF samples: hyundai_document_authenticator/data_real/*.tif
- Trained models (if already downloaded): hyundai_document_authenticator/trained_model/
  - auth_classifier/
  - yolo_photo_extractor/
  - tif_text_searcher/
- Instance directories for runtime artifacts: hyundai_document_authenticator/instance/

On Linux/macOS, replace Windows backslashes with forward slashes in all paths.

If a model path is required but missing, use tool_universal_model_downloader.py to fetch them.

Run from repo root (PowerShell or CMD):
- python hyundai_document_authenticator/tool_universal_model_downloader.py

Run from repo root (Linux/macOS):
- python3 hyundai_document_authenticator/tool_universal_model_downloader.py


## How to run examples

- All examples create a temporary runner script in %TEMP% (Windows) or /tmp (Linux/macOS) so paths remain clean.
- Replace INPUT_PATH placeholders with actual TIF/images as needed.
- For PowerShell, you can run the same python commands as CMD.
- For Linux/macOS, use bash, python3, and forward-slash paths.


---

# 1) external.tif_searcher — TIF text searcher

Purpose: Search text occurrences in multi-page TIF documents using OCR engines (e.g., PaddleOCR).

Primary entry points:
- hyundai_document_authenticator/tif_search.py (CLI wrapper)
- Module API: from tif_searcher import TifTextSearcher

Dependencies:
- OCR backend (paddleocr recommended). GPU optional.

Quick start (API):

Example runner script creates a minimal API usage.

Steps (CMD or PowerShell):
1) Create runner_tif_searcher.py in temp and execute it.

CMD:
- echo off & setlocal & > "%TEMP%\runner_tif_searcher.py" ( echo from hyundai_document_authenticator.tif_search import TifTextSearcher
  & echo from pathlib import Path
  & echo sample = r"d:\\frm_git\\hyundai_doc_auth_final\\hyundai_document_authenticator\\data_real\\N2024030602100THA00100001_13.tif"
  & echo searcher = TifTextSearcher(search_text="가맹점 실사 사진", ocr_backend="paddleocr", use_gpu_for_paddle=False)
  & echo pages = searcher.find_text_pages(sample)
  & echo print("Matched pages:", pages)
  ) & python "%TEMP%\runner_tif_searcher.py"

PowerShell:
- Set-Content -Encoding UTF8 -Path "$env:TEMP\runner_tif_searcher.py" -Value @'
from hyundai_document_authenticator.tif_search import TifTextSearcher
from pathlib import Path
sample = r"d:\\frm_git\\hyundai_doc_auth_final\\hyundai_document_authenticator\\data_real\\N2024030602100THA00100001_13.tif"
searcher = TifTextSearcher(search_text="가맹점 실사 사진", ocr_backend="paddleocr", use_gpu_for_paddle=False)
pages = searcher.find_text_pages(sample)
print("Matched pages:", pages)
'@
python "$env:TEMP\runner_tif_searcher.py"

Linux/macOS (bash):
- cat > /tmp/runner_tif_searcher.py << 'PY'
from hyundai_document_authenticator.tif_search import TifTextSearcher
sample = "/path/to/frm_git/hyundai_doc_auth_final/hyundai_document_authenticator/data_real/N2024030602100THA00100001_13.tif"
searcher = TifTextSearcher(search_text="가맹점 실사 사진", ocr_backend="paddleocr", use_gpu_for_paddle=False)
pages = searcher.find_text_pages(sample)
print("Matched pages:", pages)
PY
- python3 /tmp/runner_tif_searcher.py

CLI alternative:
- Windows: python hyundai_document_authenticator/tif_search.py --tif "d:\\frm_git\\hyundai_doc_auth_final\\hyundai_document_authenticator\\data_real\\N2024030602100THA00100001_13.tif" --text "가맹점 실사 사진" --ocr-backend paddleocr --no-gpu
- Linux/macOS: python3 hyundai_document_authenticator/tif_search.py --tif "/path/to/frm_git/hyundai_doc_auth_final/hyundai_document_authenticator/data_real/N2024030602100THA00100001_13.tif" --text "가맹점 실사 사진" --ocr-backend paddleocr --no-gpu

Debug tips:
- If PaddleOCR or language packs missing, install dependencies from docs/requirements.txt.
- Set use_gpu_for_paddle=True only if CUDA is available.


---

# 2) external.photo_extractor — Photo extraction from TIF pages (YOLO or heuristics)

Purpose: Extract photo regions from TIF using configured pipeline. Supports YOLO inference.

Primary entry points:
- hyundai_document_authenticator/photo_extractor.py (CLI wrapper)
- Module API: from photo_extractor import PhotoExtractor

Dependencies:
- For YOLO mode, a model weights file (.pt). See trained_model/yolo_photo_extractor or your custom path.

Quick start (API, YOLO):

CMD:
- echo off & setlocal & > "%TEMP%\runner_photo_extractor_yolo.py" ( echo from hyundai_document_authenticator.photo_extractor import PhotoExtractor
  & echo sample = r"d:\\frm_git\\hyundai_doc_auth_final\\hyundai_document_authenticator\\data_real\\N2024030602100THA00100001_13.tif"
  & echo pe = PhotoExtractor(config_override={"photo_extraction_mode":"yolo","yolo_object_detection":{"model_path":r"d:\\frm_git\\hyundai_doc_auth_final\\hyundai_document_authenticator\\trained_model\\yolo_photo_extractor\\best.pt","inference":{"target_object_names":["photo"]}}})
  & echo crops = pe.extract_photos(sample, page_index=0)
  & echo print("Crops count:", len(crops))
  ) & python "%TEMP%\runner_photo_extractor_yolo.py"

PowerShell:
- Set-Content -Encoding UTF8 -Path "$env:TEMP\runner_photo_extractor_yolo.py" -Value @'
from hyundai_document_authenticator.photo_extractor import PhotoExtractor
sample = r"d:\\frm_git\\hyundai_doc_auth_final\\hyundai_document_authenticator\\data_real\\N2024030602100THA00100001_13.tif"
pe = PhotoExtractor(config_override={
    "photo_extraction_mode": "yolo",
    "yolo_object_detection": {
        "model_path": r"d:\\frm_git\\hyundai_doc_auth_final\\hyundai_document_authenticator\\trained_model\\yolo_photo_extractor\\best.pt",
        "inference": {"target_object_names": ["photo"]}
    }
})
crops = pe.extract_photos(sample, page_index=0)
print("Crops count:", len(crops))
'@
python "$env:TEMP\runner_photo_extractor_yolo.py"

Linux/macOS (bash):
- cat > /tmp/runner_photo_extractor_yolo.py << 'PY'
from hyundai_document_authenticator.photo_extractor import PhotoExtractor
sample = "/path/to/frm_git/hyundai_doc_auth_final/hyundai_document_authenticator/data_real/N2024030602100THA00100001_13.tif"
pe = PhotoExtractor(config_override={
    "photo_extraction_mode": "yolo",
    "yolo_object_detection": {
        "model_path": "/path/to/frm_git/hyundai_doc_auth_final/hyundai_document_authenticator/trained_model/yolo_photo_extractor/best.pt",
        "inference": {"target_object_names": ["photo"]}
    }
})
crops = pe.extract_photos(sample, page_index=0)
print("Crops count:", len(crops))
PY
- python3 /tmp/runner_photo_extractor_yolo.py

CLI alternative:
- Windows: python hyundai_document_authenticator/photo_extractor.py --tif "d:\\frm_git\\hyundai_doc_auth_final\\hyundai_document_authenticator\\data_real\\N2024030602100THA00100001_13.tif" --mode yolo --weights "d:\\frm_git\\hyundai_doc_auth_final\\hyundai_document_authenticator\\trained_model\\yolo_photo_extractor\\best.pt"
- Linux/macOS: python3 hyundai_document_authenticator/photo_extractor.py --tif "/path/to/frm_git/hyundai_doc_auth_final/hyundai_document_authenticator/data_real/N2024030602100THA00100001_13.tif" --mode yolo --weights "/path/to/frm_git/hyundai_doc_auth_final/hyundai_document_authenticator/trained_model/yolo_photo_extractor/best.pt"

Debug tips:
- Ensure torch and CUDA versions are compatible if using GPU inference.
- Verify imageio/pillow supports TIF decoding.


---

# 3) external.image_authenticity_classifier — Authenticity classifier

Purpose: Binary/multiclass image authenticity classification based on learned embeddings.

Primary entry points:
- Module API under hyundai_document_authenticator/external/image_authenticity_classifier
- Tests under tests/external/image_authenticity_classifier

Quick start (API):

PowerShell/CMD:
- Set-Content -Encoding UTF8 -Path "$env:TEMP\runner_auth_classifier.py" -Value @'
from hyundai_document_authenticator.external.image_authenticity_classifier.classifier import AuthenticityClassifier

# Example paths and config — adjust as needed
model_dir = r"d:\\frm_git\\hyundai_doc_auth_final\\hyundai_document_authenticator\\trained_model\\auth_classifier"
classifier = AuthenticityClassifier(model_dir=model_dir)

pred = classifier.predict(r"d:\\frm_git\\hyundai_doc_auth_final\\hyundai_document_authenticator\\data_real\\N2024030602100THA00100001_13.tif", page_index=0)
print(pred)
'@
python "$env:TEMP\runner_auth_classifier.py"

Linux/macOS (bash):
- cat > /tmp/runner_auth_classifier.py << 'PY'
from hyundai_document_authenticator.external.image_authenticity_classifier.classifier import AuthenticityClassifier
model_dir = "/path/to/frm_git/hyundai_doc_auth_final/hyundai_document_authenticator/trained_model/auth_classifier"
classifier = AuthenticityClassifier(model_dir=model_dir)
pred = classifier.predict("/path/to/frm_git/hyundai_doc_auth_final/hyundai_document_authenticator/data_real/N2024030602100THA00100001_13.tif", page_index=0)
print(pred)
PY
- python3 /tmp/runner_auth_classifier.py

Debug tips:
- Ensure model_dir has the required weights/config files.
- Check docs/api pages for exact constructor arguments if signature differs.


---

# 4) external.key_input — Key input index + mock API

Purpose: Manage key indices, lookup flows, and provide a mock API server/client for integration testing.

Primary entry points:
- Mock API server: hyundai_document_authenticator/external/key_input/mock_api_server.py
- Mock API client: hyundai_document_authenticator/external/key_input/mock_api_client.py
- Orchestrator entry: hyundai_document_authenticator/external/key_input/key_input_orchestrator.py

Examples:

4.1) Run mock API server (PowerShell or CMD):
- python hyundai_document_authenticator/external/key_input/mock_api_server.py --host 127.0.0.1 --port 8089 --data "hyundai_document_authenticator/mock_api_TEST/sample_data_keys.xlsx"

4.1) Run mock API server (Linux/macOS):
- python3 hyundai_document_authenticator/external/key_input/mock_api_server.py --host 127.0.0.1 --port 8089 --data "hyundai_document_authenticator/mock_api_TEST/sample_data_keys.xlsx"

4.2) Call mock API client to fetch records:
- Windows: python hyundai_document_authenticator/external/key_input/mock_api_client.py --url http://127.0.0.1:8089 --key "N2024030602100THA00100001"
- Linux/macOS: python3 hyundai_document_authenticator/external/key_input/mock_api_client.py --url http://127.0.0.1:8089 --key "N2024030602100THA00100001"

4.3) Use orchestrator directly (API):
- Set-Content -Encoding UTF8 -Path "$env:TEMP\runner_key_orchestrator.py" -Value @'
from hyundai_document_authenticator.external.key_input.key_input_orchestrator import KeyInputOrchestrator

o = KeyInputOrchestrator(source="mock_api", url="http://127.0.0.1:8089")
print(o.lookup("N2024030602100THA00100001"))
'@
python "$env:TEMP\runner_key_orchestrator.py"

Linux/macOS (bash):
- cat > /tmp/runner_key_orchestrator.py << 'PY'
from hyundai_document_authenticator.external.key_input.key_input_orchestrator import KeyInputOrchestrator
o = KeyInputOrchestrator(source="mock_api", url="http://127.0.0.1:8089")
print(o.lookup("N2024030602100THA00100001"))
PY
- python3 /tmp/runner_key_orchestrator.py

Debug tips:
- Verify Excel file paths for mock server.
- If SQLite index is used, check hyundai_document_authenticator/instance/query_key_index.


---

# 5) external.result_gui — Results UI

Purpose: Display results for inspection. Used primarily as part of the full pipeline.

Run:
- Windows: python -m hyundai_document_authenticator.external.result_gui
- Linux/macOS: python3 -m hyundai_document_authenticator.external.result_gui

Note: Some GUIs require proper event loop integration; consult module docs/tests if needed.


---

# 6) core_engine.image_similarity_system — Core services

Purpose: Foundational components for embedding extraction, FAISS/Qdrant indices, searchers, workflows.

Key modules include:
- feature_extractor.py
- searcher.py
- workflow.py
- qdrant_manager.py, faiss_manager.py
- embedding_store.py, persistence.py
- config_loader.py, config_schema.py

These modules are designed to be imported and driven by orchestration layers. Below are standalone runnable snippets for common tasks.

6.1) Extract embeddings for an image (API):
- Set-Content -Encoding UTF8 -Path "$env:TEMP\runner_feature_extractor.py" -Value @'
from hyundai_document_authenticator.core_engine.image_similarity_system.feature_extractor import FeatureExtractor
from hyundai_document_authenticator.core_engine.image_similarity_system.config_loader import load_minimal_config

cfg = load_minimal_config()
fe = FeatureExtractor(cfg)
vec = fe.embed_image(r"d:\\frm_git\\hyundai_doc_auth_final\\hyundai_document_authenticator\\data_real\\N2024030602100THA00100001_13.tif", page_index=0)
print("Vector shape:", getattr(vec, "shape", None))
'@
python "$env:TEMP\runner_feature_extractor.py"

Linux/macOS (bash):
- cat > /tmp/runner_feature_extractor.py << 'PY'
from hyundai_document_authenticator.core_engine.image_similarity_system.feature_extractor import FeatureExtractor
from hyundai_document_authenticator.core_engine.image_similarity_system.config_loader import load_minimal_config
cfg = load_minimal_config()
fe = FeatureExtractor(cfg)
vec = fe.embed_image("/path/to/frm_git/hyundai_doc_auth_final/hyundai_document_authenticator/data_real/N2024030602100THA00100001_13.tif", page_index=0)
print("Vector shape:", getattr(vec, "shape", None))
PY
- python3 /tmp/runner_feature_extractor.py

6.2) Build or open FAISS index and upsert vectors (API):
- Set-Content -Encoding UTF8 -Path "$env:TEMP\runner_faiss_upsert.py" -Value @'
from hyundai_document_authenticator.core_engine.image_similarity_system.faiss_manager import FaissManager
from hyundai_document_authenticator.core_engine.image_similarity_system.config_loader import load_minimal_config

cfg = load_minimal_config()
fm = FaissManager(cfg)
# Create/open index and upsert a dummy vector
import numpy as np
fm.upsert("doc_1", np.random.randn(cfg.model.embedding_dim).astype("float32"))
print("Index size:", fm.count())
'@
python "$env:TEMP\runner_faiss_upsert.py"

Linux/macOS (bash):
- cat > /tmp/runner_faiss_upsert.py << 'PY'
from hyundai_document_authenticator.core_engine.image_similarity_system.faiss_manager import FaissManager
from hyundai_document_authenticator.core_engine.image_similarity_system.config_loader import load_minimal_config
cfg = load_minimal_config()
fm = FaissManager(cfg)
import numpy as np
fm.upsert("doc_1", np.random.randn(cfg.model.embedding_dim).astype("float32"))
print("Index size:", fm.count())
PY
- python3 /tmp/runner_faiss_upsert.py

6.3) Run a similarity search query (API):
- Set-Content -Encoding UTF8 -Path "$env:TEMP\runner_similarity_search.py" -Value @'
from hyundai_document_authenticator.core_engine.image_similarity_system.searcher import SimilaritySearcher
from hyundai_document_authenticator.core_engine.image_similarity_system.config_loader import load_minimal_config

cfg = load_minimal_config()
searcher = SimilaritySearcher(cfg)
resp = searcher.search_by_key("doc_1", top_k=5)
print(resp)
'@
python "$env:TEMP\runner_similarity_search.py"

Linux/macOS (bash):
- cat > /tmp/runner_similarity_search.py << 'PY'
from hyundai_document_authenticator.core_engine.image_similarity_system.searcher import SimilaritySearcher
from hyundai_document_authenticator.core_engine.image_similarity_system.config_loader import load_minimal_config
cfg = load_minimal_config()
searcher = SimilaritySearcher(cfg)
resp = searcher.search_by_key("doc_1", top_k=5)
print(resp)
PY
- python3 /tmp/runner_similarity_search.py

6.4) Execute a workflow (API):
- Set-Content -Encoding UTF8 -Path "$env:TEMP\runner_workflow.py" -Value @'
from hyundai_document_authenticator.core_engine.image_similarity_system.workflow import Workflow
from hyundai_document_authenticator.core_engine.image_similarity_system.config_loader import load_minimal_config

cfg = load_minimal_config()
wf = Workflow(cfg)
report = wf.process_document(r"d:\\frm_git\\hyundai_doc_auth_final\\hyundai_document_authenticator\\data_real\\N2024030602100THA00100001_13.tif")
print(report)
'@
python "$env:TEMP\runner_workflow.py"

Linux/macOS (bash):
- cat > /tmp/runner_workflow.py << 'PY'
from hyundai_document_authenticator.core_engine.image_similarity_system.workflow import Workflow
from hyundai_document_authenticator.core_engine.image_similarity_system.config_loader import load_minimal_config
cfg = load_minimal_config()
wf = Workflow(cfg)
report = wf.process_document("/path/to/frm_git/hyundai_doc_auth_final/hyundai_document_authenticator/data_real/N2024030602100THA00100001_13.tif")
print(report)
PY
- python3 /tmp/runner_workflow.py

Debug tips:
- If config loader names differ, inspect hyundai_document_authenticator/core_engine/image_similarity_system/config_loader.py
- Ensure instance directories exist; the library should create them when missing.


---

# 7) Running unit tests for a single module

Run only the tests for one module:
- Windows: python -m pytest -q tests/external/tif_searcher
- Windows: python -m pytest -q tests/external/photo_extractor
- Windows: python -m pytest -q tests/external/key_input
- Windows: python -m pytest -q tests/external/image_authenticity_classifier
- Windows: python -m pytest -q tests/core_engine/image_similarity_system
- Linux/macOS: python3 -m pytest -q tests/external/tif_searcher
- Linux/macOS: python3 -m pytest -q tests/external/photo_extractor
- Linux/macOS: python3 -m pytest -q tests/external/key_input
- Linux/macOS: python3 -m pytest -q tests/external/image_authenticity_classifier
- Linux/macOS: python3 -m pytest -q tests/core_engine/image_similarity_system


---

# 8) Reusing modules in other projects

Recommended approach:
- Treat each external/* directory as a library package. Import via absolute package path when used within this repo: hyundai_document_authenticator.external.<module>.<submodule>
- For reuse in another project:
  - Add this repo to your Python path, or
  - Package and publish the desired module (e.g., via a minimal setup.py) or
  - Vendor the module directory and update imports to relative or a new namespace.

Environment variables commonly used:
- HYUNDAI_DATA_DIR: base path for large artifacts
- HYUNDAI_INSTANCE_DIR: path to instance/* for indices

Best practices:
- Provide a thin adapter layer to decouple configs from repo-specific paths.
- Centralize model paths via a config file (YAML) and resolve them at runtime.


---

# 9) Troubleshooting

- ImportError: verify the working directory and PYTHONPATH include the repository root. When running from d:\\frm_git\\hyundai_doc_auth_final, imports like hyundai_document_authenticator.* should resolve.
- Missing models: run tool_universal_model_downloader.py or update paths in the examples to your local models.
- OCR failures: switch OCR backend or install language packs.
- GPU issues: set CUDA_VISIBLE_DEVICES or disable GPU flags in configs.


---

# 10) Reference: Minimal API examples (from your prior notes)

- TifTextSearcher minimal usage:
  from tif_searcher import TifTextSearcher
  searcher = TifTextSearcher(search_text="가맹점 실사 사진", ocr_backend="paddleocr", use_gpu_for_paddle=False)
  pages = searcher.find_text_pages("path/to/doc.tif")

- PhotoExtractor minimal usage (YOLO):
  from photo_extractor import PhotoExtractor
  pe = PhotoExtractor(config_override={"photo_extraction_mode":"yolo","yolo_object_detection":{"model_path":"path/to/best.pt","inference":{"target_object_names":["photo"]}}})
  crops = pe.extract_photos("path/to/doc.tif", 1)


---

# 11) Coding Standards applied to examples

- All snippets are import-safe, avoid side effects, and show explicit parameters
- OS-aware: paths use raw strings and double backslashes where needed
- Debuggability: each snippet prints deterministic, inspectable outputs

End of guide.
