from setuptools import setup, find_packages

setup(
    name="vlmgym",
    version="0.1.0",
    description="VLM-Gym: A reinforcement learning framework of games for VLMs",
    author="Liang Chen, Hongcheng Gao",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "gymnasium>=0.28.1",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pygame>=2.5.0",
    ],
    extras_require={
        "dev": [
            "black",
            "isort",
            "pylint",
            "pytest",
        ],
        "vllm": [
            "vllm",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment",
    ],
)
