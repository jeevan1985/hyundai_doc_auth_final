## Project Overview

The **Hyundai Document Authenticator** is a sophisticated AI-powered system for document image similarity search and authentication. It combines deep learning models, vector databases (FAISS/Qdrant), OCR capabilities, and photo extraction from TIF documents to provide comprehensive document verification capabilities.

### Core Architecture

The system follows a modular architecture with these main components:

- **Core Engine** (`core_engine/image_similarity_system/`): Feature extraction, vector search, and workflow orchestration
- **External Modules** (`external/`): OCR text search, photo extraction from TIFs
- **CLI Tools**: Local processing scripts (`doc_image_verifier.py`, `find_sim_images.py`, `tif_search.py`, `photo_extractor.py`)
- **API Server** (`api_server/`): Flask-based REST API with authentication
- **GUI Application** (`gui_app/`): Web interface for the API
- **Scheduler Service**: Automated document processing with APScheduler

### Technology Stack

- **AI/ML**: PyTorch, TorchVision, Transformers, FAISS, Qdrant
- **OCR**: PaddleOCR, EasyOCR, Tesseract
- **Object Detection**: YOLO for photo extraction from documents
- **Web Framework**: Flask (API + GUI)
- **Database**: PostgreSQL, SQLite
- **Containerization**: Docker with multi-stage builds, Docker Compose
- **Environment**: Conda-based Python 3.10 with pinned dependencies

## Development Environment Setup

### Conda Environment (Recommended)

The project uses Conda for dependency management with carefully pinned versions:

```bash
# Production environment (GPU-enabled)
conda env create -f environment.yml
conda activate image-similarity-env

# CPU-only development environment
conda env create -f environment.dev.yml
conda activate image-similarity-dev-env

# Verify installation
python hyundai_document_authenticator/tool_smoke_test.py
```

### Docker Development

For containerized development with live code changes:

```bash
# Start development stack with file mounts
docker compose -f docker-compose.conda.yaml -f docker-compose.conda.dev.yaml up -d

# Interactive development shell
docker compose -f docker-compose.conda.yaml -f docker-compose.conda.dev.yaml exec app_scheduler bash

# View logs
docker compose -f docker-compose.conda.yaml logs -f app_scheduler
```

## Key Commands and Workflows

### Document Image Verification (Main CLI)

The primary interface is `hyundai_document_authenticator/doc_image_verifier.py`:

```bash
# Search for similar documents using TIF files
python hyundai_document_authenticator/doc_image_verifier.py search-doc \
  --folder ./data_real \
  --top-doc 5 \
  --top-k 5 \
  --photo-extraction-mode yolo \
  --yolo-model-path trained_model/weights/best.pt

# Build image index from TIF documents  
python hyundai_document_authenticator/doc_image_verifier.py build-index \
  --folder ./data_real \
  --photo-extraction-mode bbox \
  --bbox-list "[[171,236,1480,1100],[171,1168,1480,2032]]" \
  --bbox-format xyxy
```

### Image-Only Similarity Search

For pure image similarity testing:

```bash
# Single image search
python hyundai_document_authenticator/find_sim_images.py search-img \
  --query ./instance/database_images/sample.jpg \
  --top-k 5

# Batch search
python hyundai_document_authenticator/find_sim_images.py search-img \
  --folder ./instance/database_images \
  --top-k 5
```

### Standalone OCR and Photo Extraction

```bash
# TIF text search with zonal OCR
python hyundai_document_authenticator/tif_search.py \
  --tif-path ./docs \
  --search-text "가맹점 실사 사진" \
  --search-location '{"top":0.1,"bottom":0.1}'

# Photo extraction using YOLO
python hyundai_document_authenticator/photo_extractor.py yolo \
  --tif-path path/to/documents \
  --model-path models/photo_extractor/best.pt \
  --output-dir extracted_crops/yolo
```

### Scheduler Service

The scheduler runs `doc_image_verifier.py` at configurable intervals:

```bash
# Direct execution
python scheduler_service.py

# Docker execution
docker compose -f docker-compose.conda.yaml up -d app_scheduler

# Manual CLI run (one-time execution)
docker compose -f docker-compose.conda.yaml run --rm cli_runner cli \
  hyundai_document_authenticator/doc_image_verifier.py search-doc \
  --folder ./data_real --top-doc 5 --top-k 5
```

### API Server Operations

```bash
# Start Flask API server
cd api_server
python run_api_server.py

# User management
python manage_api_users.py init_db
python manage_api_users.py add_user admin --role admin
python manage_api_users.py gen_key admin --label "Default Key"

# API client usage
python hyundai_document_authenticator/find_sim_images_api.py \
  --api-key YOUR_KEY \
  search --query path/to/image.jpg --top-k 10
```

### Testing and Development

```bash
# Smoke tests
python hyundai_document_authenticator/tool_smoke_test.py

# Database testing
python hyundai_document_authenticator/tool_database_tester.py

# Model downloading
python hyundai_document_authenticator/tool_universal_model_downloader.py \
  torchvision --model_name resnet50
```

## Docker Deployment

### Production Deployment

```bash
# CPU production
docker compose -f docker-compose.conda.yaml up -d app_scheduler

# GPU production
docker compose -f docker-compose.gpu.conda.yaml up -d app_scheduler

# With external services (PostgreSQL + Qdrant)
docker compose -f docker-compose.conda.yaml --profile postgres --profile qdrant up -d
```

### Environment Configuration

Key environment variables (`.env` file):

```bash
# Scheduler configuration
SCHEDULE_INTERVAL_MINUTES=15
PYTHON_EXECUTABLE_PATH=/opt/conda/envs/image-similarity-env/bin/python
SCRIPT_PATH=/home/appuser/app/hyundai_document_authenticator/doc_image_verifier.py
TIMEZONE=local
ALLOW_OVERLAP=false

# Database connections
POSTGRES_HOST=db
POSTGRES_DB=document_search
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password

# Vector database
QDRANT_MODE=server
QDRANT_HOST=qdrant
QDRANT_PORT=6333
```

## Configuration Architecture

### Main Configuration Files

1. **Engine Config** (`configs/image_similarity_config.yaml`): Core AI/ML pipeline settings
   - Feature extractor model configuration
   - Vector database settings (FAISS/Qdrant)
   - Photo extraction parameters
   - OCR engine configuration

2. **API Config** (`api_server/configs/api_config.yaml`): Server-specific settings
   - Database URIs
   - JWT configuration
   - API behavior overrides

3. **Environment Files**: Runtime environment variables
   - `.env`: Main environment configuration
   - `environment.yml`: Conda dependencies (production)
   - `environment.dev.yml`: Development dependencies

### Key Configuration Sections

- **feature_extractor**: Model selection (ResNet, EfficientNet, Transformers)
- **vector_database**: Provider choice and index parameters
- **photo_extractor_config**: YOLO vs bbox extraction modes
- **searcher_config**: OCR engine settings and zonal search
- **search_task**: Input/output paths and search parameters

## Code Architecture Details

### Core Workflows

The system implements several key workflows in `core_engine/image_similarity_system/workflow.py`:

1. **Index Building**: `build_index_from_tif_folder_workflow()`
   - OCR text filtering → Photo extraction → Feature extraction → Index creation

2. **TIF Batch Search**: `execute_tif_batch_search_workflow()`
   - OCR → Photo extraction → Per-photo search → Document-level aggregation

3. **Image Search**: Direct feature-based similarity search

### Key-Driven Input Mode

Advanced feature for batch processing via external key tables:
- Supports CSV/Excel/JSON input with filename mappings
- Flexible data source resolution (local folders, API, PostgreSQL)
- Batch processing with temp folder management
- Location: `external/key_input/`

### Photo Extraction Methods

Two extraction modes for TIF documents:
1. **YOLO Detection**: ML-based photo detection using trained models
2. **BBox Extraction**: Fixed bounding box cropping

### OCR and Text Search

Supports multiple OCR engines with zonal processing:
- **Engines**: PaddleOCR (default), EasyOCR, Tesseract  
- **Zonal OCR**: Restrict processing to page regions (top/bottom/center)
- **Text Normalization**: Configurable preprocessing for robust matching

## Development Guidelines

### Testing Strategy

- Use `tool_smoke_test.py` for basic functionality verification
- Test both Docker and native environments
- Validate model downloads and index creation
- Test API endpoints with different authentication methods

### Configuration Best Practices

- Always use absolute paths in production configs
- Pin dependency versions for reproducibility
- Use environment variables for sensitive data
- Test fallback behaviors (brute-force search when index unavailable)

### Performance Considerations

- IVF/HNSW indices for large-scale deployments
- Batch size tuning for memory constraints
- GPU utilization for feature extraction and OCR
- Zonal OCR for faster text processing

### Common Development Tasks

1. **Adding New Models**: Update `constants.py` with model configurations
2. **Extending Vector Providers**: Implement new managers following FAISS/Qdrant patterns  
3. **Custom OCR Engines**: Extend `external/tif_searcher` module
4. **API Endpoints**: Add routes in `api_server/app/api/`

## Troubleshooting

### Common Issues

- **FAISS Dimension Mismatch**: Enable `enable_feature_dimension_fixing` in config
- **YOLO Model Path**: Ensure `photo_extractor_config.yolo_object_detection.model_path` is correct
- **Index Loading**: Verify paths relative to project root
- **OCR Dependencies**: Install offline models for air-gapped environments
- **Docker Permissions**: Check volume mounts and user permissions

### Debug Commands

```bash
# Check configuration loading
python -c "from core_engine.image_similarity_system.config_loader import load_and_merge_configs; print(load_and_merge_configs('configs/image_similarity_config.yaml'))"

# Test feature extraction
python hyundai_document_authenticator/tool_unit_test.py

# Validate scheduler configuration
python scheduler_service.py --help
```

This system is designed for high-volume document processing with enterprise-grade security, scalability, and reliability features.