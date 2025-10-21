#!/usr/bin/env python
"""
Install Dependencies from Conda Environment YAML Safely and Sequentially.

This script automates the creation and setup of a Conda environment from a YAML
file (e.g., environment.yml). It is designed for safe, controlled, and verified
package installation.

Key Behaviors:
--------------
1.  **Auto Environment Creation**:
    Reads the 'name' from the YAML and creates a new Conda environment with that name.

2.  **Handles Existing Environments**:
    - If the environment already exists, the user is prompted whether to delete
      and recreate it.
    - Selecting "n" now keeps the existing environment and proceeds to install
      any packages into it.
    - Use the `--force` flag to bypass prompts for automation or CI workflows.

3.  **Mamba-first Installation**:
    Prefers `mamba` for package installation speed, falling back to `conda`
    if mamba is unavailable.

4.  **Sequential & Verified Installation**:
    Installs each listed package one-by-one and verifies installation by importing
    the package inside the environment to detect issues early.

5.  **Auto-Fixing**:
    If a package fails to install or verify, comments out the corresponding line
    in the YAML file and exits — enabling iterative debugging.

Usage Examples:
---------------
    # Prompted on whether to remove env if it exists
    python install_sequentially.py environment.yml

    # Automatically remove and recreate env if it exists
    python install_sequentially.py environment.yml --force
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ruamel.yaml

# --- Configuration ---
PY_IMPORT_MAP = {
    # Maps a conda/pip package name (lowercase) to its Python import name.
    "pillow": "PIL",
    "opencv": "cv2",
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "faiss-cpu": "faiss",
    "faiss-gpu": "faiss",
    "qdrant-client": "qdrant_client",
    "pydantic": "pydantic",
    "pyyaml": "yaml",
    "tqdm": "tqdm",
    "python-dotenv": "dotenv",
    "paddleocr": "paddleocr",
    "paddlepaddle": "paddle",
    "paddlepaddle-gpu": "paddle",
    "easyocr": "easyocr",
    "ultralytics": "ultralytics",
    "psycopg2": "psycopg2",
    "psycopg2-binary": "psycopg2",
    "boto3": "boto3",
}


# --- Helper Functions ---

def run(cmd: List[str]) -> Tuple[int, str]:
    """Executes a shell command and returns (exit_code, combined_stdout_stderr)."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, check=False)
    return proc.returncode, proc.stdout


def get_conda_base() -> Path:
    """Finds the base directory of the Conda installation."""
    code, out = run(["conda", "info", "--base"])
    if code != 0:
        raise RuntimeError(f"Could not determine Conda base directory:\n{out}")
    return Path(out.strip())


def get_env_python_path(env_name: str) -> Path:
    """Gets the path to the python executable inside a specific Conda environment."""
    env_path = get_conda_base() / "envs" / env_name
    python_executable = env_path / "bin" / "python"  # Linux/macOS
    if not python_executable.exists():
        python_executable = env_path / "python.exe"  # Windows
    if not python_executable.exists():
        raise FileNotFoundError(
            f"Could not find python executable in environment '{env_name}'"
        )
    return python_executable


def comment_out_line(yaml_path: Path, line_to_comment: str) -> None:
    """Comments out a specific package line in the YAML file for iterative debugging."""
    text = yaml_path.read_text(encoding="utf-8").splitlines()
    new_lines, commented = [], False
    for line in text:
        stripped_line = line.strip().lstrip("-").strip()
        if not commented and stripped_line.startswith(line_to_comment):
            new_lines.append(f"# {line}")
            commented = True
        else:
            new_lines.append(line)
    yaml_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def verify_import_in_env(env_name: str, import_name: str) -> bool:
    """Attempts to import a package inside the target conda environment."""
    try:
        python_exe = get_env_python_path(env_name)
        code, out = run([str(python_exe), "-c", f"__import__('{import_name}')"])
        if code != 0:
            print(f"--- Import verification output for '{import_name}' ---\n{out}")
        return code == 0
    except Exception as e:
        print(f"Failed during import verification: {e}")
        return False


# --- Core Logic ---

def setup_environment(env_name: str, python_version: str, force: bool) -> bool:
    """
    Create or reuse a Conda environment.
    - If force=True → environment will be deleted and recreated without prompt.
    - If force=False and env exists:
         Prompt the user to delete/recreate (y) or keep existing env (n).
    """
    print(f"--- Setting up environment: '{env_name}' with Python {python_version} ---")

    # Detect if environment exists
    code, out = run(["conda", "env", "list"])
    env_exists = code == 0 and any(
        line.split() and line.split()[0] == env_name for line in out.splitlines()
    )

    if env_exists:
        print(f"WARNING: Environment '{env_name}' already exists.")
        if not force:
            choice = input("Do you want to remove it and continue? (y/n) ").lower()
            if choice != 'y':
                print("ℹ Keeping the existing environment; proceeding with package installation.")
                return True  # <-- Continue without recreating
        print(f"Removing existing environment '{env_name}'...")
        code, out = run(["conda", "env", "remove", "-n", env_name])
        if code != 0:
            print(f"FATAL: Failed to remove environment '{env_name}'.\n{out}")
            return False

    # Create environment if it doesn't exist or was just removed
    if not env_exists or force or (env_exists and choice == 'y'):
        installer = "mamba" if shutil.which("mamba") else "conda"
        print(f"Creating new environment '{env_name}' using {installer}...")
        code, out = run([installer, "create", "-n", env_name,
                         f"python={python_version}", "-y"])
        if code != 0:
            print(f"FATAL: Could not create Conda environment '{env_name}'.\n{out}")
            return False
        print(f"✅ Environment '{env_name}' created successfully.")

    return True


def install_dependencies(env_name: str, conda_pkgs: List[str],
                         pip_pkgs: List[str], yaml_path: Path) -> bool:
    """
    Installs dependencies sequentially into the specified environment,
    verifying each install success.
    """
    print("\n--- Installing Dependencies ---")
    has_mamba = shutil.which("mamba") is not None

    # Install conda packages
    for pkg in conda_pkgs:
        installer = "mamba" if has_mamba else "conda"
        print(f"\n📦 Installing ({installer}): {pkg} into '{env_name}'")
        code, out = run([installer, "install", "-n", env_name, "-y", pkg])
        print(out)
        if code != 0:
            print(f"❌ ERROR: Failed to install conda package '{pkg}'.")
            comment_out_line(yaml_path, pkg)
            return False
        # Verify import if known
        base_pkg = pkg.split("=")[0].lower()
        import_name = PY_IMPORT_MAP.get(base_pkg)
        if import_name and not verify_import_in_env(env_name, import_name):
            print(f"❌ ERROR: Installed '{pkg}' but could not import '{import_name}'.")
            comment_out_line(yaml_path, pkg)
            return False

    # Install pip packages
    if pip_pkgs:
        python_exe = str(get_env_python_path(env_name))
        for pkg in pip_pkgs:
            print(f"\n📦 Installing (pip): {pkg} into '{env_name}'")
            code, out = run([python_exe, "-m", "pip", "install", pkg])
            print(out)
            if code != 0:
                print(f"❌ ERROR: Failed to install pip package '{pkg}'.")
                comment_out_line(yaml_path, pkg)
                return False
            # Verify import if known
            base_pkg = pkg.split("=")[0].split(">")[0].split("<")[0].strip().lower()
            import_name = PY_IMPORT_MAP.get(base_pkg)
            if import_name and not verify_import_in_env(env_name, import_name):
                print(f"❌ ERROR: Installed '{pkg}' but could not import '{import_name}'.")
                comment_out_line(yaml_path, pkg)
                return False

    return True


# --- Main Execution ---

def main() -> int:
    """Main coordinator for environment creation and package installation."""
    parser = argparse.ArgumentParser(
        description="Create and install a Conda environment sequentially from a YAML file."
    )
    parser.add_argument("yaml_path", type=Path, help="Path to the environment.yml file.")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force removal of existing environment without prompting.")
    args = parser.parse_args()

    if not args.yaml_path.is_file():
        print(f"ERROR: File not found: {args.yaml_path}")
        return 1

    try:
        # Load YAML to get environment info
        yaml = ruamel.yaml.YAML(typ="safe")
        data: Dict[str, Any] = yaml.load(args.yaml_path.read_text(encoding="utf-8"))

        env_name: Optional[str] = data.get("name")
        if not env_name:
            print("FATAL: The 'name' field is missing in your YAML file.")
            return 1

        dependencies: List[Any] = data.get("dependencies", [])
        python_version: Optional[str] = None
        for dep in dependencies:
            if isinstance(dep, str) and dep.startswith("python"):
                python_version = dep.split("=")[-1]
                break
        if not python_version:
            print("FATAL: A python version (e.g., 'python=3.9') must be specified.")
            return 1

        # 1. Setup the environment (create/remove/keep)
        if not setup_environment(env_name, python_version, args.force):
            return 1

        # 2. Separate conda and pip dependencies
        conda_pkgs = [dep for dep in dependencies
                      if isinstance(dep, str) and not dep.startswith("python") and dep != "pip"]
        pip_pkgs = []
        pip_section = next((d.get("pip", []) for d in dependencies
                            if isinstance(d, dict) and "pip" in d), [])
        pip_pkgs.extend([p for p in pip_section if isinstance(p, str)])

        # 3. Install dependencies
        if not install_dependencies(env_name, conda_pkgs, pip_pkgs, args.yaml_path):
            print("\n❌ Installation failed. Please review the errors above.")
            return 1

        print("\n" + "=" * 50)
        print("🎉 All packages installed and verified successfully! 🎉")
        print(f"\nTo activate your new environment, run:\n\nconda activate {env_name}\n")
        print("=" * 50)

    except Exception as e:
        print(f"\nFATAL: An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
