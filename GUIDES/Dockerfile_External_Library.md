Two primary methods for handling your external packages. Both are production-ready, but they serve different scenarios.

*   **Method 1: Private Git Repositories (Recommended for CI/CD & Teams)**: This is the cleanest approach, where your custom modules live in their own Git repositories and are installed during the Docker build process.
*   **Method 2: Local Source Code (Recommended for Self-Contained Projects)**: This is simpler if you want to keep all the code in a single project repository, placing your external modules in a dedicated sub-folder.

Both methods will only require minor changes to your build process.

---

### Developer's Guide: Integrating External Packages into the Docker Build

#### 1. Preparing Your Project Structure

First, let's decide where the code for your external modules will live from the perspective of the Docker build.

**If you choose Method 2 (Local Source Code)**, you should organize your project repository like this. Create a new top-level folder, such as `external_modules`, and place the source code for your custom packages inside it.

```
jk-img-similarity/
├── configs/
├── core_engine/
├── external_modules/                # <-- NEW FOLDER
│   ├── jk-tif-ocr-classifier/       # <-- Your first custom module
│   │   ├── tif_classifier/
│   │   └── setup.py
│   └── jk-shop-photo-extractor/     # <-- Your second custom module
│       ├── photo_extractor/
│       └── setup.py
├── instance/
├── tests/
├── Dockerfile                       # <-- Will be modified for this method
├── environment.yml                  # <-- Will be modified for both methods
└── ... (other files)
```

**If you choose Method 1 (Private Git Repos)**, no changes to your project structure are needed. The code will be fetched directly from Git during the build.

---

### Method 1: Installing from a Private Git Repository

This is the most robust and scalable method. It treats your modules as true third-party dependencies.

#### Step 1: Update `environment.yml`

You will add a `pip` section to your Conda environment file to instruct it to install packages directly from Git. To access a private repository, we will use a Personal Access Token (PAT).

**File:** `environment.yml`
```yaml
name: image-similarity-env
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  # ... (your existing conda dependencies like python, faiss-cpu, etc.)
  - pip
  - pip:
    # --- ADD YOUR CUSTOM MODULES HERE ---
    # The ${GIT_PAT} is a placeholder for the token we will pass in
    - git+https://jeevankharel:${GIT_PAT}@github.com/your-username/jk-tif-ocr-classifier.git@main#egg=jk-tif-ocr-classifier
    - git+https://jeevankharel:${GIT_PAT}@github.com/your-username/jk-shop-photo-extractor.git@main#egg=jk-shop-photo-extractor
    
    # --- Your existing pip dependencies from requirements.txt go below ---
    - Flask>=2.2.0,<3.1.0
    - Flask-SQLAlchemy>=2.5.0,<3.2.0
    - qdrant-client>=1.7.0,<2.0.0
    
