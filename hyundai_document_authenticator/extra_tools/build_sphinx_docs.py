#!/usr/bin/env python3
"""Sphinx documentation build pipeline for the project.

This script bootstraps a complete, reproducible Sphinx documentation workflow
and builds HTML docs for this repository.

Features
- Creates a docs/ scaffold (conf.py, index.rst, requirements.txt, _static/)
- Generates API reference using sphinx.ext.apidoc
- Builds HTML docs programmatically via the Sphinx application API
- Uses Google-style docstrings (via napoleon) and type hints rendering

Usage (Windows CMD / PowerShell):
    python tools/build_sphinx_docs.py

After a successful run, open docs/_build/html/index.html in your browser.

Notes
- Requires Sphinx and listed extensions. If missing, the script prints a
  requirements file path to install from:
    pip install -r docs/requirements.txt
- This script is idempotent; re-running will refresh the generated API rst and HTML.
"""
from __future__ import annotations

import shutil
import sys
import textwrap
from pathlib import Path
from typing import List


def _project_root() -> Path:
    """Return repository root assuming this script lives under tools/.

    Returns:
        Path: Absolute path to the repository root.
    """
    return Path(__file__).resolve().parents[1]


def _write_file(path: Path, content: str) -> None:
    """Write text content to a file, creating parent directories as needed.

    Args:
        path: Destination file path.
        content: Complete file content to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _ensure_scaffold(root: Path) -> None:
    """Create Sphinx docs scaffold files if missing or refresh content.

    Files created/updated:
    - docs/requirements.txt
    - docs/conf.py
    - docs/index.rst
    - docs/_static/ (directory)

    Args:
        root: Repository root directory.
    """
    docs = root / "docs"
    static_dir = docs / "_static"

    # Minimal requirements (pin major versions loosely for stability)
    req = textwrap.dedent(
        """
        Sphinx>=7.0
        sphinx-rtd-theme>=1.3
        myst-parser>=2.0
        sphinx-autodoc-typehints>=2.0
        """
    ).strip() + "\n"
    _write_file(docs / "requirements.txt", req)

    # Sphinx configuration (Google style docstrings & type hints rendering)
    # Use a left-aligned literal string (not an f-string) to avoid accidental interpolation.
    conf_py = textwrap.dedent(
        """
        # Configuration file for the Sphinx documentation builder.
        #
        # This file is generated and maintained by tools/build_sphinx_docs.py
        #
        from __future__ import annotations

        import os
        import sys
        from datetime import date
        from pathlib import Path

        # -- Path setup --------------------------------------------------------------
        # Add project root to sys.path so autodoc can import modules
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        # -- Project information -----------------------------------------------------
        project = "Hyundai Document Authenticator"
        author = "Project Authors"
        current_year = date.today().year
        copyright = f"{current_year}, {author}"

        # If the package provides a version, use it; otherwise fallback
        try:
            import hyundai_document_authenticator as _pkg  # type: ignore
            release = getattr(_pkg, "__version__", "0.0.0")
        except Exception:
            release = os.environ.get("PROJECT_VERSION", "0.0.0")

        # -- General configuration ---------------------------------------------------
        extensions = [
            "sphinx.ext.autodoc",
            "sphinx.ext.autosummary",
            "sphinx.ext.napoleon",
            "sphinx.ext.viewcode",
            "sphinx.ext.duration",
            "sphinx.ext.intersphinx",
        ]

        # Optional: enable MyST Markdown if installed
        try:
            import myst_parser  # noqa: F401
            extensions.append("myst_parser")
        except Exception:
            pass

        # Optional: better type hints formatting if available
        try:
            import sphinx_autodoc_typehints  # noqa: F401
            extensions.append("sphinx_autodoc_typehints")
        except Exception:
            pass

        # Autosummary/Autodoc defaults
        autosummary_generate = True
        autodoc_default_options = {
            "members": True,
            "undoc-members": True,
            "show-inheritance": True,
            "inherited-members": False,
        }
        autodoc_typehints = "description"  # Render type hints in the description

        # Napoleon (Google-style docstrings)
        napoleon_google_docstring = True
        napoleon_numpy_docstring = False
        napoleon_include_init_with_doc = False
        napoleon_attr_annotations = True

        # Intersphinx: standard libs
        intersphinx_mapping = {
            "python": ("https://docs.python.org/3", {}),
        }

        # Templates and exclusions
        templates_path = ["_templates"]
        exclude_patterns: list[str] = [
            "_build",
            "Thumbs.db",
            ".DS_Store",
        ]

        # -- Options for HTML output -------------------------------------------------
        try:
            import sphinx_rtd_theme  # noqa: F401
            html_theme = "sphinx_rtd_theme"
        except Exception:
            html_theme = "alabaster"

        html_static_path = ["_static"]
        html_css_files = [
            # Add custom CSS files here relative to _static
        ]

        # -- Autosummary and TOC depth ----------------------------------------------
        # Keep module trees reasonably deep by default
        toc_object_entries = True
        """
    ).lstrip("\n")

    _write_file(docs / "conf.py", conf_py)

    # Root index with project overview and API TOC
    index_rst = textwrap.dedent(
        """
        Welcome to Hyundai Document Authenticator's Documentation
        =========================================================

        .. toctree::
           :maxdepth: 2
           :caption: Contents

           api/index

        Indices and Tables
        ==================

        * :ref:`genindex`
        * :ref:`modindex`
        * :ref:`search`
        """
    ).lstrip("\n")
    _write_file(docs / "index.rst", index_rst)

    static_dir.mkdir(parents=True, exist_ok=True)


def _ensure_clean_api_tree(api_dir: Path) -> None:
    """Remove previously generated API rst tree to avoid stale entries.

    Args:
        api_dir: API docs directory (e.g., docs/api).
    """
    if api_dir.exists():
        shutil.rmtree(api_dir)
    api_dir.mkdir(parents=True, exist_ok=True)


def _generate_api_docs(source_pkg_dir: Path, api_out_dir: Path) -> None:
    """Generate API reference rst files using sphinx.ext.apidoc.

    Args:
        source_pkg_dir: Directory containing the Python package to document.
        api_out_dir: Destination directory for generated rst files.
    """
    try:
        from sphinx.ext.apidoc import main as apidoc_main  # type: ignore
    except Exception as exc:  # pragma: no cover - requires sphinx installed
        raise RuntimeError(
            "Sphinx is required to generate API docs. Install from docs/requirements.txt"
        ) from exc

    argv: List[str] = [
        "--force",            # overwrite existing files
        "--module-first",     # put module before members
        "--separate",         # separate pages per module
        "--no-toc",           # we'll maintain our own TOC
        "-o",
        str(api_out_dir),
        str(source_pkg_dir),
    ]
    apidoc_main(argv)

    # Create api/index.rst (TOC) if not created by apidoc (since we used --no-toc)
    index = api_out_dir / "index.rst"
    if not index.exists():
        modules = sorted(p.stem for p in api_out_dir.glob("*.rst") if p.name != "index.rst")
        toc_lines: List[str] = [
            "API Reference",
            "============",
            "",
            ".. toctree::",
            "   :maxdepth: 2",
            "",
        ]
        toc_lines.extend([f"   {m}" for m in modules])
        _write_file(index, "\n".join(toc_lines) + "\n")


def _build_html(docs_dir: Path) -> None:
    """Build HTML docs programmatically using the Sphinx application API.

    Args:
        docs_dir: Directory containing conf.py and sources (docs/).
    """
    try:
        from sphinx.application import Sphinx  # type: ignore
    except Exception as exc:  # pragma: no cover - requires sphinx installed
        raise RuntimeError(
            "Sphinx is required to build HTML docs. Install from docs/requirements.txt"
        ) from exc

    srcdir = docs_dir
    confdir = docs_dir
    outdir = docs_dir / "_build" / "html"
    doctreedir = docs_dir / "_build" / "doctrees"
    buildername = "html"

    outdir.mkdir(parents=True, exist_ok=True)
    doctreedir.mkdir(parents=True, exist_ok=True)

    app = Sphinx(
        srcdir=str(srcdir),
        confdir=str(confdir),
        outdir=str(outdir),
        doctreedir=str(doctreedir),
        buildername=buildername,
        warningiserror=False,
        freshenv=True,
        verbosity=0,
    )
    app.build(force_all=True)


def main() -> int:
    """Entry point for building Sphinx HTML documentation.

    Returns:
        int: Process exit code (0 on success, non-zero on failure).
    """
    root = _project_root()
    pkg_dir = root / "hyundai_document_authenticator"
    docs_dir = root / "docs"
    api_dir = docs_dir / "api"

    if not pkg_dir.exists():
        print(f"Package directory not found: {pkg_dir}", file=sys.stderr)
        return 2

    _ensure_scaffold(root)
    _ensure_clean_api_tree(api_dir)

    try:
        _generate_api_docs(pkg_dir, api_dir)
    except RuntimeError as e:
        # Guidance to install deps if missing
        print(str(e), file=sys.stderr)
        print(f"To install: pip install -r {docs_dir / 'requirements.txt'}", file=sys.stderr)
        return 3

    try:
        _build_html(docs_dir)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        print(f"To install: pip install -r {docs_dir / 'requirements.txt'}", file=sys.stderr)
        return 4

    index_html = docs_dir / "_build" / "html" / "index.html"
    print("\nSphinx HTML documentation successfully built.")
    print(f"Open: {index_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
