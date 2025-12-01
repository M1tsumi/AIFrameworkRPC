"""
Configuration management for AIFrameworkRPC with automatic detection.
"""

import json
import os
import sys
import platform
import subprocess
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration manager for AIFrameworkRPC with automatic detection.
    """
    
    DEFAULT_CONFIG = {
        "discord_client_id": "",
        "default_status": "Working with AI tools",
        "auto_share": {
            "enabled": False,
            "channel_id": "",
            "share_images": True,
            "share_text": False
        },
        "status_templates": {
            "generating": "Generating {tool} with {model}",
            "training": "Training on {dataset}",
            "chatting": "Chatting with {model}",
            "idle": "Ready for next task"
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "timeouts": {
            "connection_timeout": 30,
            "update_interval": 5
        },
        "performance": {
            "cache_timeout": 1.0,
            "max_workers": 2,
            "connection_pool_size": 5
        },
        "auto_detect": {
            "enabled": True,
            "scan_ai_tools": True,
            "detect_discord": True
        }
    }
    
    def __init__(self, config_file: str = "ai_rpc_config.json"):
        """
        Initialize configuration with automatic detection.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file)
        self._config = self.DEFAULT_CONFIG.copy()
        self._detected_tools = {}
        self._detected_discord = False
        
        # Load existing config
        self.load_config()
        
        # Run automatic detection if enabled
        if self.get("auto_detect.enabled", True):
            self.run_automatic_detection()
    
    def run_automatic_detection(self):
        """Run automatic detection of AI tools and Discord."""
        logger.info("Running automatic configuration detection...")
        
        # Detect Discord
        if self.get("auto_detect.detect_discord", True):
            self._detect_discord()
        
        # Detect AI tools
        if self.get("auto_detect.scan_ai_tools", True):
            self._detect_ai_tools()
        
        # Auto-configure based on detections
        self._auto_configure()
    
    def _detect_discord(self) -> bool:
        """Detect if Discord is running."""
        system = platform.system().lower()
        
        try:
            if system == "windows":
                # Check for Discord processes on Windows
                result = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Discord.exe"], 
                                      capture_output=True, text=True)
                self._detected_discord = "Discord.exe" in result.stdout
            elif system == "darwin":  # macOS
                result = subprocess.run(["pgrep", "-f", "Discord"], 
                                      capture_output=True, text=True)
                self._detected_discord = result.returncode == 0
            elif system == "linux":
                result = subprocess.run(["pgrep", "-f", "discord"], 
                                      capture_output=True, text=True)
                self._detected_discord = result.returncode == 0
            else:
                self._detected_discord = False
            
            if self._detected_discord:
                logger.info("Discord detected and running")
            else:
                logger.info("Discord not detected or not running")
                
        except Exception as e:
            logger.warning(f"Failed to detect Discord: {e}")
            self._detected_discord = False
        
        return self._detected_discord
    
    def _detect_ai_tools(self) -> Dict[str, Any]:
        """Detect installed AI tools and their configurations."""
        tools = {}
        
        # Detect Stable Diffusion interfaces
        tools.update(self._detect_stable_diffusion())
        
        # Detect Local LLM tools
        tools.update(self._detect_local_llms())
        
        # Detect other AI tools
        tools.update(self._detect_other_ai_tools())
        
        self._detected_tools = tools
        logger.info(f"Detected AI tools: {list(tools.keys())}")
        return tools
    
    def _detect_stable_diffusion(self) -> Dict[str, Any]:
        """Detect Stable Diffusion installations."""
        tools = {}
        
        # Check for Automatic1111
        a1111_paths = self._find_common_ai_paths("stable-diffusion-webui")
        if a1111_paths:
            tools["automatic1111"] = {
                "type": "stable_diffusion",
                "paths": a1111_paths,
                "name": "Stable Diffusion (Automatic1111)"
            }
        
        # Check for ComfyUI
        comfy_paths = self._find_common_ai_paths("ComfyUI")
        if comfy_paths:
            tools["comfyui"] = {
                "type": "stable_diffusion",
                "paths": comfy_paths,
                "name": "ComfyUI"
            }
        
        # Check for InvokeAI
        invoke_paths = self._find_common_ai_paths("InvokeAI")
        if invoke_paths:
            tools["invokeai"] = {
                "type": "stable_diffusion",
                "paths": invoke_paths,
                "name": "InvokeAI"
            }
        
        return tools
    
    def _detect_local_llms(self) -> Dict[str, Any]:
        """Detect Local LLM installations."""
        tools = {}
        
        # Check for Ollama
        if self._check_command_exists("ollama"):
            tools["ollama"] = {
                "type": "llm",
                "command": "ollama",
                "name": "Ollama"
            }
        
        # Check for LM Studio
        lmstudio_paths = self._find_common_ai_paths("LM Studio")
        if lmstudio_paths:
            tools["lmstudio"] = {
                "type": "llm",
                "paths": lmstudio_paths,
                "name": "LM Studio"
            }
        
        # Check for text-generation-webui
        tgw_paths = self._find_common_ai_paths("text-generation-webui")
        if tgw_paths:
            tools["textgenwebui"] = {
                "type": "llm",
                "paths": tgw_paths,
                "name": "Text Generation WebUI"
            }
        
        return tools
    
    def _detect_other_ai_tools(self) -> Dict[str, Any]:
        """Detect other AI tools."""
        tools = {}
        
        # Check for common AI directories
        ai_dirs = [
            "AI", "ai_tools", "machine_learning", "artificial_intelligence"
        ]
        
        for dir_name in ai_dirs:
            paths = self._find_common_ai_paths(dir_name)
            if paths:
                tools[f"ai_directory_{dir_name}"] = {
                    "type": "directory",
                    "paths": paths,
                    "name": f"AI Directory: {dir_name}"
                }
        
        return tools
    
    def _find_common_ai_paths(self, tool_name: str) -> List[str]:
        """Find common installation paths for AI tools."""
        paths = []
        system = platform.system().lower()
        
        # Common base directories
        if system == "windows":
            base_dirs = [
                Path.home() / "AppData" / "Local",
                Path.home() / "AppData" / "Roaming",
                Path("C:\\"),
                Path("D:\\"),
                Path.home() / "Desktop",
                Path.home() / "Documents"
            ]
        elif system == "darwin":
            base_dirs = [
                Path.home() / "Applications",
                Path.home() / "Documents",
                Path.home() / "Desktop",
                Path("/Applications")
            ]
        else:  # Linux
            base_dirs = [
                Path.home() / ".local" / "share",
                Path.home() / "Documents",
                Path.home() / "Desktop",
                Path("/opt"),
                Path("/usr/local")
            ]
        
        # Search for tool directories
        for base_dir in base_dirs:
            if base_dir.exists():
                try:
                    for item in base_dir.rglob(f"*{tool_name}*"):
                        if item.is_dir() and tool_name.lower() in item.name.lower():
                            paths.append(str(item))
                except (PermissionError, OSError):
                    continue
        
        return list(set(paths))  # Remove duplicates
    
    def _check_command_exists(self, command: str) -> bool:
        """Check if a command exists in the system PATH."""
        try:
            subprocess.run([command, "--version"], 
                          capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _auto_configure(self):
        """Auto-configure based on detected tools and Discord."""
        changes_made = False
        
        # Configure default status based on detected tools
        if self._detected_tools:
            tool_types = {tool["type"] for tool in self._detected_tools.values()}
            
            if "stable_diffusion" in tool_types and "llm" in tool_types:
                self.set("default_status", "Working with AI art and text generation")
            elif "stable_diffusion" in tool_types:
                self.set("default_status", "Creating AI art")
            elif "llm" in tool_types:
                self.set("default_status", "Chatting with AI")
            
            changes_made = True
        
        # Configure performance based on detected tools count
        tool_count = len(self._detected_tools)
        if tool_count > 3:
            self.set("performance.max_workers", min(4, tool_count))
            self.set("performance.connection_pool_size", min(8, tool_count * 2))
            changes_made = True
        
        # Add detected tools to configuration
        if self._detected_tools:
            self.set("detected_tools", self._detected_tools)
            changes_made = True
        
        # Add Discord detection status
        self.set("detected_discord", self._detected_discord)
        changes_made = True
        
        if changes_made:
            logger.info("Auto-configuration applied based on detected tools")
    
    def get_detected_tools(self) -> Dict[str, Any]:
        """Get information about detected AI tools."""
        return self._detected_tools.copy()
    
    def is_discord_detected(self) -> bool:
        """Check if Discord was detected."""
        return self._detected_discord
    
    def get_quick_setup_suggestions(self) -> Dict[str, Any]:
        """Get suggestions for quick setup based on detections."""
        suggestions = {
            "recommended_examples": [],
            "configuration_tips": [],
            "performance_settings": {}
        }
        
        # Suggest examples based on detected tools
        if any(tool["type"] == "stable_diffusion" for tool in self._detected_tools.values()):
            suggestions["recommended_examples"].append("stable_diffusion.py")
        
        if any(tool["type"] == "llm" for tool in self._detected_tools.values()):
            suggestions["recommended_examples"].append("llm_chat.py")
        
        if len(self._detected_tools) > 1:
            suggestions["recommended_examples"].append("multi_tool.py")
        
        # Configuration tips
        if not self._detected_discord:
            suggestions["configuration_tips"].append(
                "Discord was not detected. Make sure Discord is running before using AIFrameworkRPC."
            )
        
        if not self.get_discord_client_id():
            suggestions["configuration_tips"].append(
                "No Discord client ID found. Run the setup wizard: python -m ai_framework_rpc.setup_wizard"
            )
        
        # Performance settings
        tool_count = len(self._detected_tools)
        if tool_count > 2:
            suggestions["performance_settings"] = {
                "cache_timeout": 0.5,
                "max_workers": min(4, tool_count),
                "connection_pool_size": min(8, tool_count * 2)
            }
        
        return suggestions
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Loaded configuration dictionary
        """
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    self._merge_config(self._config, loaded_config)
                    print(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                print(f"Error loading config file: {e}. Using defaults.")
        else:
            print(f"Config file {self.config_file} not found. Using defaults.")
            
        return self._config
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            # Create directory if it doesn't exist
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            print(f"Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving config file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation like 'auto_share.enabled')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        # Set the value
        config[keys[-1]] = value
    
    def get_discord_client_id(self) -> str:
        """Get Discord client ID from config or environment."""
        client_id = self.get("discord_client_id") or os.getenv("DISCORD_CLIENT_ID")
        if not client_id:
            raise ValueError("Discord client ID not found in config or environment")
        return client_id
    
    def get_bot_token(self) -> Optional[str]:
        """Get Discord bot token from config or environment."""
        return self.get("bot_token") or os.getenv("DISCORD_BOT_TOKEN")
    
    def is_auto_share_enabled(self) -> bool:
        """Check if auto-share is enabled."""
        return self.get("auto_share.enabled", False)
    
    def get_share_channel_id(self) -> str:
        """Get auto-share channel ID."""
        channel_id = self.get("auto_share.channel_id")
        if not channel_id and self.is_auto_share_enabled():
            raise ValueError("Auto-share enabled but no channel ID configured")
        return channel_id
    
    def get_status_template(self, template_name: str) -> str:
        """
        Get status template by name.
        
        Args:
            template_name: Name of template
            
        Returns:
            Template string
        """
        return self.get(f"status_templates.{template_name}", "")
    
    def format_status_template(self, template_name: str, **kwargs) -> str:
        """
        Format status template with provided variables.
        
        Args:
            template_name: Name of template
            **kwargs: Variables to substitute in template
            
        Returns:
            Formatted template string
        """
        template = self.get_status_template(template_name)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            print(f"Missing template variable: {e}")
            return template
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get("logging", {})
    
    def get_timeouts(self) -> Dict[str, int]:
        """Get timeout configuration."""
        return self.get("timeouts", {})
    
    def _merge_config(self, base: Dict[str, Any], update: Dict[str, Any]):
        """
        Recursively merge update dict into base dict.
        
        Args:
            base: Base configuration dict
            update: Update dict to merge
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def create_default_config_file(self):
        """Create a default configuration file."""
        if not self.config_file.exists():
            self._config = self.DEFAULT_CONFIG.copy()
            self.save_config()
            print(f"Created default config file at {self.config_file}")
        else:
            print(f"Config file already exists at {self.config_file}")
    
    def validate_config(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            True if configuration is valid
        """
        try:
            # Check Discord client ID
            self.get_discord_client_id()
            
            # Check auto-share configuration
            if self.is_auto_share_enabled():
                self.get_share_channel_id()
                
            return True
            
        except ValueError as e:
            print(f"Configuration validation failed: {e}")
            return False
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return json.dumps(self._config, indent=2, ensure_ascii=False)
