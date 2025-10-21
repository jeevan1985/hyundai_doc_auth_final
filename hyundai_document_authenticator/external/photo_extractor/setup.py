"""Setup script for the vendored 'photo_extractor' package.

This file is included to make the external package installable when extracted
or used independently.
"""
from setuptools import setup, find_packages

setup(
    name="photo-extractor",  # Distribution name (pip install photo-extractor)
    version="1.0.0",
    description="Detector-backed extractor for shop photos from TIF pages",
    author="Your Company",
    packages=find_packages(include=["photo_extractor", "photo_extractor.*"]),
    python_requires=">=3.8",
    install_requires=[
        "Pillow>=9.0.0",
        # ultralytics should be installed by consumers who need YOLO inference
        # e.g., pip install ultralytics
    ],
    entry_points={
        "console_scripts": [
            "photo-extractor=photo_extractor.cli:main",
        ]
    }
)
