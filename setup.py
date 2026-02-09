"""
Setup script for BSPNN package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

setup(
    name="bspnn",
    version="1.0.0",
    author="Min-Gyoung Shin",
    description="BSPNN: Biological Signal Pathway Neural Network - Pathway-based prediction pipeline using neural networks with hierarchical modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bspnn",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0,<2.0.0",
        "pandas>=1.2.0",
        "tensorflow>=2.8.0; python_version<'3.13'",
        "scikit-learn>=0.24.0",
        "shap>=0.39.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "bspnn-step1=bspnn.steps.step1_primary_prediction:main",
            "bspnn-step2=bspnn.steps.step2_prediction_level1:main",
            "bspnn-step3=bspnn.steps.step3_prediction_level2:main",
        ],
    },
)
