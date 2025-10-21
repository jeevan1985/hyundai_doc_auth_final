from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """
    Runs the full pytest suite and generates an HTML report.

    This script executes pytest and instructs it to create a self-contained
    HTML report file named 'report.html' in the project root.

    Returns:
        The exit code from the pytest process.
    """
    project_root = Path(__file__).resolve().parent
    report_path = project_root / "report.html"
    
    print("Starting pytest suite...")
    
    command = [
        sys.executable,
        "-m", "pytest",
        "-v", # Use verbose output
        f"--html={report_path}",
        "--self-contained-html" # Ensure the report is a single file
    ]
    
    try:
        result = subprocess.run(command, check=False, cwd=project_root)
        
        if result.returncode == 0:
            print(f"\n✅ Test suite passed. Report generated at: {report_path}")
        else:
            print(f"\n❌ Test suite failed. Report generated at: {report_path}", file=sys.stderr)
            
        return result.returncode
        
    except FileNotFoundError:
        print("Error: 'pytest' not found. Make sure it is installed in your environment.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
