"""
Setup script for AIFrameworkRPC.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "AIFrameworkRPC - Discord Rich Presence for AI Tools"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="ai-framework-rpc",
    version="0.2.0",
    author="AIFrameworkRPC Team",
    author_email="contact@aiframeworkrpc.com",
    description="A lightweight library for integrating Discord Rich Presence (RPC) with AI tools",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-framework-rpc",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Communications :: Chat",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "discord": [
            "discord.py>=2.0.0",
        ],
        "all": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "discord.py>=2.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ai-rpc=ai_framework_rpc.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ai_framework_rpc": [
            "config/*.json",
            "assets/*.png",
        ],
    },
    keywords="discord rpc ai stable-diffusion llm chatbot rich-presence",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ai-framework-rpc/issues",
        "Source": "https://github.com/yourusername/ai-framework-rpc",
        "Documentation": "https://ai-framework-rpc.readthedocs.io/",
        "Discord": "https://discord.gg/yourcommunity",
    },
)
