"""
AIFrameworkRPC - A lightweight library for integrating Discord Rich Presence (RPC) with AI tools.
"""
__version__ = "0.1.1"
__author__ = "AIFrameworkRPC Team"

from .core import AIFrameworkRPC
from .integrations import StableDiffusionRPC, LLMRPC
from .multi_tool import MultiToolRPC
from .config import Config

__all__ = [
    "AIFrameworkRPC",
    "StableDiffusionRPC", 
    "LLMRPC",
    "MultiToolRPC",
    "Config"
]
