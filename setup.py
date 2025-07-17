from setuptools import setup, find_packages

setup(
    name="amp_searcher",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "torch",
        "pytorch-lightning",
        "biopython",
        "pyyaml",
        "torchmetrics",
        "fastapi",
        "uvicorn",
    ],
)
