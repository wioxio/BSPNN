from pathlib import Path
from setuptools import setup, find_packages


this_directory = Path(__file__).parent
readme_path = this_directory / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""


setup(
    name="bspnn",
    version="2.0.0",
    author="Min-Gyoung Shin",
    description="BSPNN pathway-based prediction scripts packaged for distribution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "tensorflow>=2.8.0",
        "keras>=2.8.0",
        "scikit-learn>=0.24.0",
        "shap>=0.39.0",
    ],
    entry_points={
        "console_scripts": [
            "bspnn-step1=bspnn.cli:step1",
            "bspnn-step2=bspnn.cli:step2",
            "bspnn-step3=bspnn.cli:step3",
        ],
    },
)
