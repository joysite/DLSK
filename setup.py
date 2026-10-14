from pathlib import Path

from setuptools import setup

readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="dlsk",
    version="0.1.0",
    description="Directional Large Selective Kernel module for computer vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="joysite",
    url="https://github.com/joysite/DLSK",
    license="MIT",
    py_modules=["dlsk"],
    python_requires=">=3.8",
    install_requires=["torch>=1.10.0"],
    extras_require={"dev": ["pytest>=7.0", "pytest-cov>=4.0", "build>=1.0"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="deep learning, PyTorch, directional convolution, attention, YOLO",
)