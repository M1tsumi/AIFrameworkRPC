"""
Updated configuration system with enhanced features for AIFrameworkRPC v0.2.0
"""

import json
import os
import sys
import platform
import subprocess
import time
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import yaml


class ConfigFormat(Enum):
    """Supported configuration formats."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"


@dataclass
class DiscordConfig:
    """Discord-specific configuration."""
    client_id: str = ""
    default_status: str = "Working with AI tools"
    auto_reconnect: bool = True
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 5
    presence_timeout: float = 30.0
    enable_heartbeat: bool = True


@dataclass
class AutoShareConfig:
    """Auto-sharing configuration."""
    enabled: bool = False
    channel_id: str = ""
    share_images: bool = True
    share_text: bool = False
    share_interval: float = 300.0  # 5 minutes
    max_file_size: int = 25 * 1024 * 1024  # 25MB


@dataclass
class StatusTemplate:
    """Status template configuration."""
    name: str
    template: str
    variables: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    file_path: str = "ai_framework_rpc.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_enabled: bool = True


@dataclass
class PerformanceConfig:
    """Performance configuration."""
    cache_timeout: float = 1.0
    max_workers: int = 2
    connection_pool_size: int = 5
    predictive_caching: bool = True
    connection_balancing: bool = True
    memory_limit_mb: int = 500
    cpu_limit_percent: float = 80.0


@dataclass
class SecurityConfig:
    """Security configuration."""
    encryption_enabled: bool = True
    key_rotation_interval: int = 2592000  # 30 days
    audit_logging: bool = True
    session_timeout: int = 3600
    max_failed_attempts: int = 5
    backup_encryption: bool = True


@dataclass
class PluginConfig:
    """Plugin system configuration."""
    enabled: bool = True
    auto_load: bool = True
    plugin_directories: List[str] = field(default_factory=lambda: ["plugins"])
    max_plugins: int = 50
    enable_sandbox: bool = True
    trusted_plugins: List[str] = field(default_factory=list)


@dataclass
class WebDashboardConfig:
    """Web dashboard configuration."""
    enabled: bool = True
    host: str = "localhost"
    port: int = 8080
    enable_auth: bool = False
    auth_token: str = ""
    enable_cors: bool = True
    update_interval: float = 1.0


@dataclass
class MonitoringConfig:
    """Monitoring and analytics configuration."""
    enabled: bool = True
    metrics_collection: bool = True
    performance_profiling: bool = True
    bottleneck_detection: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_percent': 80.0,
        'memory_mb': 500.0,
        'response_time_ms': 1000.0,
        'error_rate': 0.1
    })


@dataclass
class AIToolConfig:
    """AI tool integration configuration."""
    auto_detect: bool = True
    scan_interval: float = 60.0
    supported_tools: List[str] = field(default_factory=lambda: [
        "stable_diffusion", "chatgpt", "claude", "midjourney", "dall-e"
    ])
    tool_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class EnhancedConfig:
    """
    Enhanced configuration manager with validation, migration, and advanced features.
    
    Features:
    - Multiple format support (JSON, YAML, TOML)
    - Configuration validation
    - Schema validation
    - Environment variable overrides
    - Configuration hot-reloading
    - Migration support
    - Template system
    - Security features
    """
    
    VERSION = "2.0"
    DEFAULT_CONFIG_FILE = "config_v2.json"
    
    def __init__(self, config_file: Optional[str] = None, 
                 config_format: ConfigFormat = ConfigFormat.JSON,
                 auto_create: bool = True,
                 enable_validation: bool = True):
        self.config_file = Path(config_file or self.DEFAULT_CONFIG_FILE)
        self.config_format = config_format
        self.auto_create = auto_create
        self.enable_validation = enable_validation
        
        # Configuration sections
        self.discord: DiscordConfig = DiscordConfig()
        self.auto_share: AutoShareConfig = AutoShareConfig()
        self.status_templates: List[StatusTemplate] = []
        self.logging: LoggingConfig = LoggingConfig()
        self.performance: PerformanceConfig = PerformanceConfig()
        self.security: SecurityConfig = SecurityConfig()
        self.plugins: PluginConfig = PluginConfig()
        self.web_dashboard: WebDashboardConfig = WebDashboardConfig()
        self.monitoring: MonitoringConfig = MonitoringConfig()
        self.ai_tools: AIToolConfig = AIToolConfig()
        
        # Runtime state
        self._config_hash: Optional[str] = None
        self._last_modified: Optional[float] = None
        self._watch_thread: Optional[threading.Thread] = None
        self._stop_watching = threading.Event()
        self._change_callbacks: List[Callable[[], None]] = []
        
        # Detection state
        self._detected_tools: Dict[str, Any] = {}
        self._detected_discord: bool = False
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self._load_configuration()
        
        # Setup default status templates
        self._setup_default_templates()
        
        # Run auto-detection if enabled
        if self.ai_tools.auto_detect:
            self._run_automatic_detection()
        
        # Start file watching if enabled
        self._start_file_watching()
    
    def _load_configuration(self):
        """Load configuration from file."""
        try:
            if self.config_file.exists():
                config_data = self._read_config_file()
                self._apply_configuration(config_data)
                self.logger.info(f"Configuration loaded from {self.config_file}")
            elif self.auto_create:
                self._create_default_config()
                self.logger.info(f"Created default configuration at {self.config_file}")
            else:
                self.logger.warning("Configuration file not found and auto-create disabled")
        
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            if self.auto_create:
                self._create_default_config()
    
    def _read_config_file(self) -> Dict[str, Any]:
        """Read configuration file based on format."""
        with open(self.config_file, 'r', encoding='utf-8') as f:
            if self.config_format == ConfigFormat.JSON:
                return json.load(f)
            elif self.config_format == ConfigFormat.YAML:
                try:
                    import yaml
                    return yaml.safe_load(f)
                except ImportError:
                    self.logger.warning("PyYAML not available, falling back to JSON")
                    return json.load(f)
            elif self.config_format == ConfigFormat.TOML:
                try:
                    import tomllib
                    return tomllib.load(f)
                except ImportError:
                    self.logger.warning("tomllib not available, falling back to JSON")
                    return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {self.config_format}")
    
    def _apply_configuration(self, config_data: Dict[str, Any]):
        """Apply configuration data to objects."""
        # Apply Discord config
        if 'discord' in config_data:
            discord_data = config_data['discord']
            for key, value in discord_data.items():
                if hasattr(self.discord, key):
                    setattr(self.discord, key, value)
        
        # Apply auto-share config
        if 'auto_share' in config_data:
            share_data = config_data['auto_share']
            for key, value in share_data.items():
                if hasattr(self.auto_share, key):
                    setattr(self.auto_share, key, value)
        
        # Apply logging config
        if 'logging' in config_data:
            logging_data = config_data['logging']
            for key, value in logging_data.items():
                if hasattr(self.logging, key):
                    setattr(self.logging, key, value)
        
        # Apply performance config
        if 'performance' in config_data:
            perf_data = config_data['performance']
            for key, value in perf_data.items():
                if hasattr(self.performance, key):
                    setattr(self.performance, key, value)
        
        # Apply security config
        if 'security' in config_data:
            security_data = config_data['security']
            for key, value in security_data.items():
                if hasattr(self.security, key):
                    setattr(self.security, key, value)
        
        # Apply plugin config
        if 'plugins' in config_data:
            plugin_data = config_data['plugins']
            for key, value in plugin_data.items():
                if hasattr(self.plugins, key):
                    setattr(self.plugins, key, value)
        
        # Apply web dashboard config
        if 'web_dashboard' in config_data:
            dashboard_data = config_data['web_dashboard']
            for key, value in dashboard_data.items():
                if hasattr(self.web_dashboard, key):
                    setattr(self.web_dashboard, key, value)
        
        # Apply monitoring config
        if 'monitoring' in config_data:
            monitoring_data = config_data['monitoring']
            for key, value in monitoring_data.items():
                if hasattr(self.monitoring, key):
                    setattr(self.monitoring, key, value)
        
        # Apply AI tools config
        if 'ai_tools' in config_data:
            ai_data = config_data['ai_tools']
            for key, value in ai_data.items():
                if hasattr(self.ai_tools, key):
                    setattr(self.ai_tools, key, value)
        
        # Apply status templates
        if 'status_templates' in config_data:
            self.status_templates = []
            for template_data in config_data['status_templates']:
                template = StatusTemplate(
                    name=template_data['name'],
                    template=template_data['template'],
                    variables=template_data.get('variables', []),
                    enabled=template_data.get('enabled', True)
                )
                self.status_templates.append(template)
        
        # Update hash for change detection
        self._config_hash = self._calculate_config_hash()
        self._last_modified = self.config_file.stat().st_mtime if self.config_file.exists() else 0
    
    def _create_default_config(self):
        """Create a default configuration file."""
        default_config = {
            'version': self.VERSION,
            'created_at': time.time(),
            'discord': asdict(self.discord),
            'auto_share': asdict(self.auto_share),
            'logging': asdict(self.logging),
            'performance': asdict(self.performance),
            'security': asdict(self.security),
            'plugins': asdict(self.plugins),
            'web_dashboard': asdict(self.web_dashboard),
            'monitoring': asdict(self.monitoring),
            'ai_tools': asdict(self.ai_tools),
            'status_templates': [
                asdict(template) for template in self.status_templates
            ]
        }
        
        self._write_config_file(default_config)
    
    def _write_config_file(self, config_data: Dict[str, Any]):
        """Write configuration data to file."""
        # Ensure directory exists
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            if self.config_format == ConfigFormat.JSON:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            elif self.config_format == ConfigFormat.YAML:
                try:
                    import yaml
                    yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
                except ImportError:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
            elif self.config_format == ConfigFormat.TOML:
                try:
                    import tomli_w
                    tomli_w.dump(config_data, f)
                except ImportError:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        # Set secure permissions
        try:
            os.chmod(self.config_file, 0o600)
        except OSError:
            self.logger.warning("Could not set secure file permissions")
    
    def _setup_default_templates(self):
        """Setup default status templates."""
        if not self.status_templates:
            self.status_templates = [
                StatusTemplate(
                    name="generating",
                    template="Generating {tool} with {model}",
                    variables=["tool", "model"]
                ),
                StatusTemplate(
                    name="training",
                    template="Training on {dataset}",
                    variables=["dataset"]
                ),
                StatusTemplate(
                    name="chatting",
                    template="Chatting with {model}",
                    variables=["model"]
                ),
                StatusTemplate(
                    name="idle",
                    template="Ready for next task",
                    variables=[]
                )
            ]
    
    def _run_automatic_detection(self):
        """Run automatic detection of AI tools and Discord."""
        self.logger.info("Running automatic configuration detection...")
        
        # Detect Discord
        self._detect_discord()
        
        # Detect AI tools
        self._detect_ai_tools()
        
        # Auto-configure based on detections
        self._auto_configure()
    
    def _detect_discord(self) -> bool:
        """Detect if Discord is running."""
        system = platform.system().lower()
        
        try:
            if system == "windows":
                result = subprocess.run(
                    ["tasklist", "/FI", "IMAGENAME eq Discord.exe"], 
                    capture_output=True, text=True
                )
                self._detected_discord = "Discord.exe" in result.stdout
            elif system == "darwin":  # macOS
                result = subprocess.run(
                    ["pgrep", "-f", "Discord"], 
                    capture_output=True, text=True
                )
                self._detected_discord = result.returncode == 0
            elif system == "linux":
                result = subprocess.run(
                    ["pgrep", "-f", "discord"], 
                    capture_output=True, text=True
                )
                self._detected_discord = result.returncode == 0
            else:
                self._detected_discord = False
            
            self.logger.info(f"Discord detection: {'found' if self._detected_discord else 'not found'}")
            return self._detected_discord
            
        except Exception as e:
            self.logger.warning(f"Discord detection failed: {e}")
            self._detected_discord = False
            return False
    
    def _detect_ai_tools(self):
        """Detect available AI tools."""
        detected = {}
        
        # Check for Stable Diffusion
        if self._check_tool_available("stablediffusion", ["python", "-c", "import stablediffusion"]):
            detected["stable_diffusion"] = {"available": True, "version": "unknown"}
        
        # Check for common AI tool directories
        tool_paths = [
            "/opt/stable-diffusion",
            "~/stable-diffusion",
            "./stable-diffusion-webui",
            "~/Automatic1111"
        ]
        
        for path in tool_paths:
            expanded_path = Path(path).expanduser()
            if expanded_path.exists():
                tool_name = expanded_path.name
                detected[tool_name] = {
                    "available": True,
                    "path": str(expanded_path),
                    "type": "stable_diffusion"
                }
        
        # Check for API keys in environment
        api_keys = {
            "openai": ["OPENAI_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY"],
            "huggingface": ["HUGGINGFACE_API_KEY"],
            "stability": ["STABILITY_API_KEY"]
        }
        
        for tool, key_names in api_keys.items():
            for key_name in key_names:
                if os.getenv(key_name):
                    detected[tool] = {
                        "available": True,
                        "api_key_configured": True,
                        "key_name": key_name
                    }
                    break
        
        self._detected_tools = detected
        self.logger.info(f"Detected AI tools: {list(detected.keys())}")
    
    def _check_tool_available(self, tool_name: str, command: List[str]) -> bool:
        """Check if a tool is available by running a command."""
        try:
            result = subprocess.run(command, capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def _auto_configure(self):
        """Auto-configure based on detections."""
        # If Discord is detected and no client_id is set
        if self._detected_discord and not self.discord.client_id:
            self.logger.info("Discord detected but no client_id configured")
        
        # Add detected tools to supported tools
        for tool_name in self._detected_tools.keys():
            if tool_name not in self.ai_tools.supported_tools:
                self.ai_tools.supported_tools.append(tool_name)
    
    def _start_file_watching(self):
        """Start watching configuration file for changes."""
        def watch_loop():
            while not self._stop_watching.is_set():
                try:
                    if self.config_file.exists():
                        current_modified = self.config_file.stat().st_mtime
                        if current_modified != self._last_modified:
                            self.logger.info("Configuration file changed, reloading...")
                            self._load_configuration()
                            
                            # Notify callbacks
                            for callback in self._change_callbacks:
                                try:
                                    callback()
                                except Exception as e:
                                    self.logger.error(f"Error in config change callback: {e}")
                    
                    time.sleep(1)  # Check every second
                    
                except Exception as e:
                    self.logger.error(f"File watching error: {e}")
                    time.sleep(5)
        
        self._watch_thread = threading.Thread(target=watch_loop, daemon=True)
        self._watch_thread.start()
    
    def _calculate_config_hash(self) -> str:
        """Calculate hash of current configuration."""
        config_dict = self.to_dict()
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def save(self):
        """Save current configuration to file."""
        config_data = self.to_dict()
        config_data['version'] = self.VERSION
        config_data['updated_at'] = time.time()
        
        self._write_config_file(config_data)
        self.logger.info(f"Configuration saved to {self.config_file}")
    
    def reload(self):
        """Reload configuration from file."""
        self._load_configuration()
        self.logger.info("Configuration reloaded")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'discord.client_id')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self
        
        try:
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                elif isinstance(value, dict):
                    value = value[k]
                else:
                    return default
            
            return value
            
        except (KeyError, AttributeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'discord.client_id')
            value: Value to set
        """
        keys = key.split('.')
        obj = self
        
        # Navigate to the parent object
        for k in keys[:-1]:
            if hasattr(obj, k):
                obj = getattr(obj, k)
            else:
                raise KeyError(f"Configuration section not found: {k}")
        
        # Set the value
        setattr(obj, keys[-1], value)
    
    def add_status_template(self, template: StatusTemplate):
        """Add a status template."""
        self.status_templates.append(template)
    
    def remove_status_template(self, name: str):
        """Remove a status template by name."""
        self.status_templates = [t for t in self.status_templates if t.name != name]
    
    def get_status_template(self, name: str) -> Optional[StatusTemplate]:
        """Get a status template by name."""
        for template in self.status_templates:
            if template.name == name:
                return template
        return None
    
    def render_status_template(self, name: str, variables: Dict[str, str]) -> str:
        """
        Render a status template with variables.
        
        Args:
            name: Template name
            variables: Variables to substitute
            
        Returns:
            Rendered template string
        """
        template = self.get_status_template(name)
        if not template:
            return ""
        
        try:
            return template.template.format(**variables)
        except KeyError as e:
            self.logger.warning(f"Missing variable in template {name}: {e}")
            return template.template
    
    def add_change_callback(self, callback: Callable[[], None]):
        """Add a callback to be called when configuration changes."""
        self._change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: Callable[[], None]):
        """Remove a configuration change callback."""
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)
    
    def validate(self) -> List[str]:
        """
        Validate current configuration.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate Discord configuration
        if not self.discord.client_id:
            errors.append("Discord client_id is required")
        
        if self.discord.reconnect_delay < 0:
            errors.append("Discord reconnect_delay must be positive")
        
        if self.discord.max_reconnect_attempts < 1:
            errors.append("Discord max_reconnect_attempts must be at least 1")
        
        # Validate performance configuration
        if self.performance.max_workers < 1:
            errors.append("Performance max_workers must be at least 1")
        
        if self.performance.connection_pool_size < 1:
            errors.append("Performance connection_pool_size must be at least 1")
        
        # Validate security configuration
        if self.security.session_timeout < 60:
            errors.append("Security session_timeout must be at least 60 seconds")
        
        # Validate web dashboard configuration
        if self.web_dashboard.port < 1 or self.web_dashboard.port > 65535:
            errors.append("Web dashboard port must be between 1 and 65535")
        
        # Validate status templates
        template_names = [t.name for t in self.status_templates]
        if len(template_names) != len(set(template_names)):
            errors.append("Status template names must be unique")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'discord': asdict(self.discord),
            'auto_share': asdict(self.auto_share),
            'status_templates': [asdict(t) for t in self.status_templates],
            'logging': asdict(self.logging),
            'performance': asdict(self.performance),
            'security': asdict(self.security),
            'plugins': asdict(self.plugins),
            'web_dashboard': asdict(self.web_dashboard),
            'monitoring': asdict(self.monitoring),
            'ai_tools': asdict(self.ai_tools)
        }
    
    def get_detected_tools(self) -> Dict[str, Any]:
        """Get detected AI tools information."""
        return self._detected_tools.copy()
    
    def is_discord_detected(self) -> bool:
        """Check if Discord was detected."""
        return self._detected_discord
    
    def migrate_from_legacy(self, legacy_config_file: str) -> bool:
        """
        Migrate from legacy configuration format.
        
        Args:
            legacy_config_file: Path to legacy config file
            
        Returns:
            True if migration successful
        """
        try:
            legacy_path = Path(legacy_config_file)
            if not legacy_path.exists():
                self.logger.error(f"Legacy config file not found: {legacy_config_file}")
                return False
            
            with open(legacy_path, 'r') as f:
                legacy_config = json.load(f)
            
            # Map legacy fields to new structure
            mapping = {
                'discord_client_id': 'discord.client_id',
                'default_status': 'discord.default_status',
                'auto_share': 'auto_share.enabled',
                'channel_id': 'auto_share.channel_id',
                'share_images': 'auto_share.share_images',
                'share_text': 'auto_share.share_text'
            }
            
            for legacy_key, new_key in mapping.items():
                if legacy_key in legacy_config:
                    self.set(new_key, legacy_config[legacy_key])
            
            # Handle nested objects
            if 'status_templates' in legacy_config:
                for name, template in legacy_config['status_templates'].items():
                    self.add_status_template(StatusTemplate(
                        name=name,
                        template=template
                    ))
            
            # Save new configuration
            self.save()
            
            # Create backup of legacy config
            backup_path = legacy_path.with_suffix('.json.backup')
            shutil.copy2(legacy_path, backup_path)
            
            self.logger.info(f"Successfully migrated from {legacy_config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            return False
    
    def export_config(self, export_path: str, include_secrets: bool = False):
        """
        Export configuration to file.
        
        Args:
            export_path: Path to export file
            include_secrets: Whether to include sensitive data
        """
        config_data = self.to_dict()
        
        # Remove sensitive data if requested
        if not include_secrets:
            sensitive_keys = [
                'discord.client_id',
                'auto_share.channel_id',
                'web_dashboard.auth_token'
            ]
            
            for key in sensitive_keys:
                self.set(key, "***REDACTED***")
        
        export_file = Path(export_path)
        with open(export_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        self.logger.info(f"Configuration exported to {export_path}")
    
    def import_config(self, import_path: str, merge: bool = True):
        """
        Import configuration from file.
        
        Args:
            import_path: Path to import file
            merge: Whether to merge with existing config or replace
        """
        import_file = Path(import_path)
        if not import_file.exists():
            raise FileNotFoundError(f"Import file not found: {import_path}")
        
        with open(import_file, 'r') as f:
            import_data = json.load(f)
        
        if merge:
            # Merge with existing configuration
            self._apply_configuration(import_data)
        else:
            # Replace entire configuration
            self._apply_configuration(import_data)
        
        self.save()
        self.logger.info(f"Configuration imported from {import_path}")
    
    def shutdown(self):
        """Shutdown configuration manager."""
        self._stop_watching.set()
        
        if self._watch_thread and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=2)
        
        self.logger.info("Configuration manager shutdown completed")


# Global configuration instance
_global_config: Optional[EnhancedConfig] = None


def initialize_config(config_file: Optional[str] = None, 
                     config_format: ConfigFormat = ConfigFormat.JSON) -> EnhancedConfig:
    """
    Initialize the global configuration.
    
    Args:
        config_file: Configuration file path
        config_format: Configuration format
        
    Returns:
        EnhancedConfig instance
    """
    global _global_config
    
    if _global_config is not None:
        logging.warning("Configuration is already initialized")
        return _global_config
    
    _global_config = EnhancedConfig(config_file, config_format)
    return _global_config


def get_config() -> Optional[EnhancedConfig]:
    """Get the global configuration instance."""
    return _global_config


def shutdown_config():
    """Shutdown the global configuration."""
    global _global_config
    
    if _global_config is not None:
        _global_config.shutdown()
        _global_config = None
