"""Setup configuration for 5D Chess project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chess-5d-mcts",
    version="0.1.0",
    author="Your Name",
    description="5D Chess with Monte Carlo Tree Search AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/chess-5d-mcts",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Games/Entertainment :: Board Games",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "cupy-cuda11x>=12.0.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "chess5d=src.game_runner:main",
        ],
    },
)
