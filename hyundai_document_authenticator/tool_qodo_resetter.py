"""Utility to clean up Qodo and VS Code related caches and configuration.

Provides functions to list target cleanup paths per-OS and to safely remove
files/directories with user confirmation.
"""
import os
import platform
import shutil
from typing import List

def get_cleanup_paths() -> List[str]:
    """
    Identify paths for VS Code and other specified configs based on the OS.

    This function is cross-platform and finds the correct directories
    on Windows, macOS, and Linux.

    Returns:
        List[str]: Full paths to the files and directories for cleanup.
    """
    system = platform.system()
    home_dir = os.path.expanduser("~")
    paths_to_clean = []

    print(f"🖥️  Detected Operating System: {system}")

    # --- Platform-specific path definitions ---
    if system == "Windows":
        app_data = os.getenv("APPDATA")
        if app_data:
            code_dir = os.path.join(app_data, "Code")
            paths_to_clean.extend([
                os.path.join(code_dir, "blob_storage"),
                os.path.join(code_dir, "CachedData"),
                os.path.join(code_dir, "Code Cache"),
                os.path.join(code_dir, "User"),
                os.path.join(code_dir, "GPUCache"),
                os.path.join(code_dir, "WebStorage"),
                os.path.join(code_dir, "code.lock"),
                os.path.join(code_dir, "Local State"),
                os.path.join(code_dir, "machineid"),
                os.path.join(code_dir, "SharedStorage")
            ])
        # Add other specified paths
        paths_to_clean.append(os.path.join(home_dir, ".qodo"))
        # Example of another path from your original list
        paths_to_clean.append(os.path.normpath("D:/frm_git/hyundai_document_authenticator/.qodo"))


    elif system == "Darwin":  # macOS
        library_dir = os.path.join(home_dir, "Library", "Application Support")
        code_dir = os.path.join(library_dir, "Code")
        paths_to_clean.append(code_dir) # On macOS, the whole directory is often targeted
        paths_to_clean.append(os.path.join(home_dir, ".qodo"))


    elif system == "Linux":
        config_dir = os.path.join(home_dir, ".config", "Code")
        cache_dir = os.path.join(home_dir, ".cache", "Code")
        paths_to_clean.extend([config_dir, cache_dir])
        paths_to_clean.append(os.path.join(home_dir, ".qodo"))

    return paths_to_clean

def safe_remove(path: str) -> None:
    """
    Safely remove a file or directory (recursively), if it exists.

    Args:
        path (str): The full path to the item to remove.

    Returns:
        None
    """
    # First, check if the path even exists
    if not os.path.exists(path):
        print(f"🟡  Skipping: '{path}' (Does not exist).")
        return

    try:
        # Differentiate between a file and a directory
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
            print(f"✅  Removed file: '{path}'")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            print(f"✅  Removed directory: '{path}'")
    except Exception as e:
        print(f"❌  Error removing '{path}': {e}")
        print("    ↳ It might be in use by another program or you may lack permissions.")

# --- Main execution block ---
if __name__ == "__main__":
    print("--- 🔍 Application Cleanup Utility ---")
    
    # 1. Identify all paths to be removed
    targets = get_cleanup_paths()
    
    # Filter out paths that don't actually exist to create a clean list
    existing_targets = [p for p in targets if os.path.exists(p)]
    
    if not existing_targets:
        print("\n✨ All clean! No specified folders or files were found.")
    else:
        print("\nThe following items are targeted for **PERMANENT DELETION**:")
        for path in existing_targets:
            print(f"  - {path}")
            
        print("\n" + "="*50)
        print("🛑 IMPORTANT: This action cannot be undone. All settings and data")
        print("   in the folders above will be lost.")
        print("="*50 + "\n")

        # 2. Get explicit user confirmation
        try:
            # The strip() removes accidental whitespace, lower() makes input case-insensitive
            user_consent = input("Are you absolutely sure you want to proceed? (yes/no): ").strip().lower()
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n\nOperation cancelled by user. Exiting.")
            user_consent = "no"

        # 3. Proceed with deletion only if confirmed
        if user_consent == 'yes':
            print("\n🚀 Confirmation received. Starting cleanup process...")
            for item in existing_targets:
                safe_remove(item)
            print("\n🎉 Cleanup complete!")
        else:
            print("\nOperation aborted. No files were changed.")