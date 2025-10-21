"""Setup script for the vendored 'tif_searcher' package.

This file is included to make the external package installable when extracted
or used independently.
"""
from setuptools import setup, find_packages

setup(
    name="tif-searcher",  # Distribution name (pip install tif-searcher)
    version="1.1.0",
    description="A professional, configurable OCR-based text searcher for images.",
    author="Your Company",
    packages=find_packages(include=["tif_searcher_core", "tif_searcher_core.*"]),
    python_requires=">=3.8",
    install_requires=[
        "Pillow>=9.0.0",
        "paddleocr>=2.5",
        "requests",
        "tqdm",
        "numpy",
    ],
    entry_points={
        "console_scripts": [
            "text-search-from-image=tif_searcher.main:run_text_search_workflow",
        ]
    }
)
