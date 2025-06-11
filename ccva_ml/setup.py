from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ccva_ml",
    version="0.1.0",
    author="Isaac Lyatuu",
    author_email="ilyatuu@gmail.com",
    description="Computer Coded Verbal Autopsy (CCVA) Machile Learning Module",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vman-tool/ccva-ml",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'joblib',
        'vman3_dq',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    include_package_data=True,
    package_data={
        'ccva_ml': ['data/*.csv'],
    },
)