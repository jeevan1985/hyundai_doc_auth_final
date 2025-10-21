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
