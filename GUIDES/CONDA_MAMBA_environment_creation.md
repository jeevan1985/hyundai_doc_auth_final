> Refactor Notice (2025-10): Canonical data directory renamed from `/appdat` to `/appdata`. If your local scripts or custom configs reference `/appdat`, update them accordingly. Only lowercase `appdat` forms were changed; other `data` words are unaffected.

# Conda/Mamba Environment Creation Guide

This is the complete, step-by-step guide for creating and managing the Hyundai Document Authenticator environment using the actual project files.

Follow these phases in order.

---

## **Phase 0: Prerequisites and Project Setup**

**Goal:** Ensure you have the correct project structure and understand the environment files.

**Prerequisites:**
1. **Conda or Mamba installed** (Mamba is recommended for faster package resolution)
2. **Navigate to your project directory:** `d:/frm_git/hyundai_document_authenticator`
3. **Verify the environment file exists:** The project includes a pre-configured `environment.yml` file

**Available Environment Files:**
- `environment.yml` - **Main production environment** (recommended for most users)
- `environment.dev.yml` - Development variant with additional debugging tools (if available)
- `environment.cpu.yml` - CPU-only variant for systems without CUDA support (if available)

**Important:** This guide uses the existing `environment.yml` file. **Do not create a new one** - the project already includes a fully configured, tested environment definition.

---

## **Phase 1: Environment Creation**

**Goal:** Build a clean environment using the project's tested configuration.

**Commands:**

1. **Open your terminal (Anaconda Prompt or Command Prompt).** Make sure you are in the `(base)` environment. If you are in another environment, deactivate it first:
   ```bash
   conda deactivate
   ```

2. **Navigate to the project directory:**
   ```bash
   cd d:/frm_git/hyundai_document_authenticator
   ```

3. **CRITICAL - Remove any existing environment with the same name:**
   ```bash
   mamba env remove -n image-similarity-env
   ```
   or if using conda:
   ```bash
   conda env remove -n image-similarity-env
   ```

4. **Create the new environment from the project file.** This reads the `environment.yml` and installs all dependencies. This may take 10-20 minutes:
   ```bash
   mamba env create -f environment.yml --override-channels
   ```
   or if using conda:
   ```bash
   conda env create -f environment.yml
   ```

**What gets installed:**
- Python 3.10
- PyTorch 2.0.1 with CUDA 11.7 support
- FAISS-GPU for vector similarity search
- Computer vision libraries (OpenCV, EasyOCR, PaddleOCR, Ultralytics)
- Web frameworks (Flask, FastAPI, Streamlit)
- Database connectors and utilities
- 100+ additional packages with pinned versions for stability

---

## **Phase 2: Activation and Verification**

**Goal:** Enter your new environment and confirm everything works correctly.

**Commands:**

1. **Activate your new environment.** You will see the prompt change from `(base)` to `(image-similarity-env)`:
   ```bash
   mamba activate image-similarity-env
   ```
   or:
   ```bash
   conda activate image-similarity-env
   ```

2. **Verify Python version and key packages:**
   ```bash
   python --version
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
   python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
   ```

3. **Run project verification (if available):**
   ```bash
   python -m hyundai_document_authenticator.doc_image_verifier --help
   ```

4. **Test GPU availability (if you have CUDA-capable hardware):**
   ```bash
   python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
   ```

---

## **Phase 3: Creating Your Backups**

**Goal:** Create backups of your working environment for future restoration.

**Commands (run these while your environment is active):**

1. **Create a detailed "Blueprint" backup** with exact versions:
   ```bash
   conda env export > environment-backup-$(date +%Y%m%d).yml
   ```
   On Windows:
   ```bash
   conda env export > environment-backup.yml
   ```

2. **Create an "Exact Clone" backup** for fast local restoration:
   ```bash
   conda list --explicit > spec-file-$(date +%Y%m%d).txt
   ```
   On Windows:
   ```bash
   conda list --explicit > spec-file.txt
   ```

3. **Verify your backups were created:**
   ```bash
   ls -la environment-backup*.yml spec-file*.txt
   ```
   On Windows:
   ```bash
   dir environment-backup*.yml spec-file*.txt
   ```

---

## **Phase 4: Daily Usage**

**Goal:** Know how to use your environment for development and production work.

**Commands:**

1. **To start working:**
   ```bash
   conda activate image-similarity-env
   ```

2. **To run the main application:**
   ```bash
   python -m hyundai_document_authenticator.doc_image_verifier
   ```

3. **To run web interfaces:**
   ```bash
   # Flask web interface
   python -m hyundai_document_authenticator.web_app
   
   # Streamlit interface (if available)
   streamlit run hyundai_document_authenticator/streamlit_app.py
   ```

4. **To run tests:**
   ```bash
   pytest hyundai_document_authenticator/tests/
   ```

5. **To stop working and return to base:**
   ```bash
   conda deactivate
   ```

---

## **Phase 5: Environment Management**

**Goal:** Know how to maintain, update, and restore your environment.

### **Restoring from Backups:**

1. **From Blueprint backup (`environment-backup.yml`):**
   ```bash
   conda env create -f environment-backup.yml
   ```

2. **From Exact Clone backup (`spec-file.txt`):**
   ```bash
   # Step A: Create empty environment
   conda create -n image-similarity-env --no-default-packages
   
   # Step B: Install exact packages
   conda install -n image-similarity-env --file spec-file.txt
   ```

### **Environment Maintenance:**

1. **Update packages (use with caution):**
   ```bash
   conda activate image-similarity-env
   conda update --all
   ```

2. **Add new packages:**
   ```bash
   conda activate image-similarity-env
   conda install package-name
   # or
   pip install package-name
   ```

3. **Remove the environment completely:**
   ```bash
   conda env remove -n image-similarity-env
   ```

4. **List all environments:**
   ```bash
   conda env list
   ```

---

## **Phase 6: Troubleshooting**

**Common Issues and Solutions:**

### **Environment Creation Fails:**
```bash
# Clear conda cache
conda clean --all

# Try with explicit channels
mamba env create -f environment.yml --override-channels

# If still failing, try CPU-only version
mamba env create -f environment.cpu.yml  # if available
```

### **CUDA/GPU Issues:**
```bash
# Verify CUDA installation
nvidia-smi

# Test PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

### **Package Conflicts:**
```bash
# Check for conflicts
conda check

# Resolve conflicts by recreating environment
conda env remove -n image-similarity-env
mamba env create -f environment.yml --override-channels
```

### **Slow Package Resolution:**
```bash
# Use mamba instead of conda
conda install mamba -n base -c conda-forge
mamba env create -f environment.yml
```

### **Windows-Specific Issues:**
```bash
# Long path support
git config --system core.longpaths true

# PowerShell execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## **Environment File Details**

The `environment.yml` file includes:

- **Conda-managed packages:** Core binaries (Python, CUDA, PyTorch, FAISS, NumPy)
- **Pip-managed packages:** Application-level libraries and utilities
- **Pinned versions:** All packages are pinned to specific versions for reproducibility
- **Commercial compliance:** Uses conda-forge and removes defaults channel
- **GPU optimization:** CUDA-enabled versions of PyTorch, FAISS, and PaddlePaddle

**Channels used:**
- `pytorch` - Official PyTorch packages
- `nvidia` - CUDA toolkit and drivers
- `conda-forge` - Community-maintained packages
- `nodefaults` - Excludes default Anaconda commercial packages

This environment is optimized for:
- Document image processing and authentication
- Computer vision and OCR tasks
- Web application deployment
- GPU-accelerated machine learning
- Production stability and reproducibility
