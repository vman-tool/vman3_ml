from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vman_ml",
    version="1.0.0",
    author="Isaac Lyatuu",
    author_email="ilyatuu@gmail.com",
    description="Computer Coded Verbal Autopsy (CCVA) Machine Learning Module",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vman-tool/ccva-ml",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.6.0,<2.0.0",
        "joblib>=1.3.0",
        "xgboost>=2.0.0",
        "sentence-transformers>=3.0.0",
        "shap>=0.45.0",
        "imbalanced-learn>=0.12.0",
        "openpyxl>=3.1.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    package_data={
        "vman_ml": [
            "data/*.csv",
            "data/*.xlsx",
            "data/*.json",
            "data/dictionaries/*",
        ],
    },
)
