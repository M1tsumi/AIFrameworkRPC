"""
AIFrameworkRPC v0.2.0 - Enhanced Discord Rich Presence for AI Tools
"""

__version__ = "0.2.0"
__author__ = "AIFrameworkRPC Team"
__description__ = "Enhanced Discord Rich Presence for AI Tools with advanced features"

# Core imports
from .core import AIFrameworkRPC
from .enhanced_main import EnhancedAIFrameworkRPC, create_enhanced_rpc, quick_start

# Configuration
from .config import Config
from .config_v2 import EnhancedConfig, initialize_config, get_config

# Integrations
from .integrations import StableDiffusionIntegration, ChatGPTIntegration

# Enhanced connection management
from .enhanced_connection import IntelligentConnectionPool, _enhanced_connection_pool

# Plugin system
from .plugin_system import (
    PluginManager, 
    BasePlugin, 
    StatusEnhancerPlugin, 
    ToolIntegrationPlugin, 
    create_plugin
)

# Predictive caching
from .predictive_cache import PredictiveCache, _predictive_cache

# Error recovery
from .error_recovery import (
    ContextAwareRetryManager, 
    retry_on_failure, 
    _global_retry_manager
)

# Performance profiling
from .performance_profiler import (
    PerformanceProfiler, 
    profile_function, 
    _global_profiler
)

# Web dashboard
from .web_dashboard import (
    WebDashboard, 
    DashboardConfig, 
    start_dashboard, 
    stop_dashboard
)

# Security
from .security import (
    SecurityManager, 
    SecurityConfig, 
    initialize_security, 
    get_security_manager
)

# Migration
from .migrate import MigrationManager

# Enhanced exports for v0.2.0
__all__ = [
    # Core classes
    "AIFrameworkRPC",
    "EnhancedAIFrameworkRPC",
    "create_enhanced_rpc",
    "quick_start",
    
    # Configuration
    "Config",
    "EnhancedConfig",
    "initialize_config",
    "get_config",
    
    # Integrations
    "StableDiffusionIntegration",
    "ChatGPTIntegration",
    
    # Enhanced connection management
    "IntelligentConnectionPool",
    "_enhanced_connection_pool",
    
    # Plugin system
    "PluginManager",
    "BasePlugin",
    "StatusEnhancerPlugin",
    "ToolIntegrationPlugin",
    "create_plugin",
    
    # Predictive caching
    "PredictiveCache",
    "_predictive_cache",
    
    # Error recovery
    "ContextAwareRetryManager",
    "retry_on_failure",
    "_global_retry_manager",
    
    # Performance profiling
    "PerformanceProfiler",
    "profile_function",
    "_global_profiler",
    
    # Web dashboard
    "WebDashboard",
    "DashboardConfig",
    "start_dashboard",
    "stop_dashboard",
    
    # Security
    "SecurityManager",
    "SecurityConfig",
    "initialize_security",
    "get_security_manager",
    
    # Migration
    "MigrationManager",
]

# Version information
VERSION_INFO = {
    "major": 0,
    "minor": 2,
    "patch": 0,
    "release": "stable"
}

# Feature flags
FEATURES = {
    "enhanced_connections": True,
    "plugin_system": True,
    "predictive_caching": True,
    "error_recovery": True,
    "performance_profiling": True,
    "web_dashboard": True,
    "enhanced_security": True,
    "migration_tools": True,
    "advanced_config": True
}

# Default configuration template
DEFAULT_CONFIG_TEMPLATE = {
    "discord": {
        "client_id": "",
        "default_status": "Working with AI tools",
        "auto_reconnect": True
    },
    "features": {
        "enhanced_connections": True,
        "plugin_system": True,
        "predictive_caching": True,
        "web_dashboard": True
    },
    "performance": {
        "cache_timeout": 1.0,
        "max_workers": 2,
        "connection_pool_size": 5
    }
}

def get_version():
    """Get version information."""
    return __version__

def get_features():
    """Get available features."""
    return FEATURES.copy()

def is_feature_enabled(feature: str) -> bool:
    """Check if a feature is enabled."""
    return FEATURES.get(feature, False)

# Welcome message for enhanced version
def print_welcome():
    """Print welcome message for AIFrameworkRPC v0.2.0."""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                AIFrameworkRPC v{__version__}                ║
║         Enhanced Discord Rich Presence for AI Tools         ║
╠══════════════════════════════════════════════════════════════╣
║ Features:                                                    ║
║ • Intelligent Connection Management                         ║
║ • Plugin System with SDK                                     ║
║ • Predictive Caching                                         ║
║ • Context-Aware Error Recovery                               ║
║ • Real-time Performance Profiling                            ║
║ • Web Dashboard with Monitoring                              ║
║ • Enhanced Security & Encryption                             ║
║ • Migration Tools                                            ║
║ • Advanced Configuration System                              ║
╚══════════════════════════════════════════════════════════════╝
""")

# Auto-print welcome on import (can be disabled)
if __name__ != "__main__":
    import os
    if os.getenv("AIFRAMEWORK_RPC_HIDE_WELCOME", "").lower() != "true":
        print_welcome()
