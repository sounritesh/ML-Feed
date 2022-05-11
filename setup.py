# setup.py
# Setup installation for the application

from pathlib import Path

from setuptools import find_namespace_packages, setup

BASE_DIR = Path(__file__).parent

# Load packages from requirements.txt
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]

setup(
    name="ml-feed",
    version="0.1",
    license="MIT",
    description="Personalised ML-feed for Now&Me",
    author="Ritesh Soun",
    author_email="ritesh@nowandme.com",
    url="https://nowandme.com/",
    keywords=[
        "deep-learning",
        "artificial-intelligence",
        "now-and-me",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8.5",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
    entry_points={
        "console_scripts": [
            "get_topics = topic_modelling",
        ],
    },
)