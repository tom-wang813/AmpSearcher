# setup.py
from setuptools import setup, find_packages

setup(
    name="amp",
    version="0.1.0",
    packages=find_packages(),    # 自动把 amp/ 下的所有包都打包
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pytest>=6.2.0",
        "pytest-cov>=2.12.0",
        "black>=21.7b0",
        "flake8>=3.9.2",
        "isort>=5.9.3",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.10",
)
