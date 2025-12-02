"""
Plugin system architecture with SDK for custom integrations - AIFrameworkRPC v0.2.0
"""

import abc
import inspect
import importlib
import json
import time
import threading
from typing import Dict, Any, List, Optional, Callable, Type, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging

from .core import AIFrameworkRPC


class PluginType(Enum):
    """Types of plugins supported by the system."""
    AI_TOOL = "ai_tool"
    STATUS_ENHANCER = "status_enhancer"
    ANALYTICS = "analytics"
    NOTIFICATION = "notification"
    INTEGRATION = "integration"


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    min_ai_framework_version: str = "0.2.0"
    max_ai_framework_version: Optional[str] = None
    config_schema: Optional[Dict[str, Any]] = None
    permissions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'plugin_type': self.plugin_type.value,
            'dependencies': self.dependencies,
            'min_ai_framework_version': self.min_ai_framework_version,
            'max_ai_framework_version': self.max_ai_framework_version,
            'config_schema': self.config_schema,
            'permissions': self.permissions
        }


class PluginEvent(Enum):
    """Events that plugins can listen to."""
    STATUS_UPDATE = "status_update"
    CONNECTION_ESTABLISHED = "connection_established"
    CONNECTION_LOST = "connection_lost"
    ERROR_OCCURRED = "error_occurred"
    PLUGIN_LOADED = "plugin_loaded"
    PLUGIN_UNLOADED = "plugin_unloaded"


@dataclass
class PluginContext:
    """Context provided to plugins during execution."""
    rpc_instance: AIFrameworkRPC
    plugin_config: Dict[str, Any]
    plugin_data_dir: Path
    event_bus: 'PluginEventBus'
    logger: logging.Logger
    
    def emit_event(self, event: PluginEvent, data: Dict[str, Any] = None):
        """Emit an event to the event bus."""
        self.event_bus.emit(event, data or {})


class PluginEventBus:
    """Event bus for plugin communication."""
    
    def __init__(self):
        self._listeners: Dict[PluginEvent, List[Callable]] = {}
        self._lock = threading.Lock()
    
    def subscribe(self, event: PluginEvent, callback: Callable):
        """Subscribe to an event."""
        with self._lock:
            if event not in self._listeners:
                self._listeners[event] = []
            self._listeners[event].append(callback)
    
    def unsubscribe(self, event: PluginEvent, callback: Callable):
        """Unsubscribe from an event."""
        with self._lock:
            if event in self._listeners:
                try:
                    self._listeners[event].remove(callback)
                except ValueError:
                    pass
    
    def emit(self, event: PluginEvent, data: Dict[str, Any]):
        """Emit an event to all listeners."""
        with self._lock:
            listeners = self._listeners.get(event, []).copy()
        
        for callback in listeners:
            try:
                callback(data)
            except Exception as e:
                logging.error(f"Error in plugin event callback for {event}: {e}")


class BasePlugin(abc.ABC):
    """
    Base class for all plugins.
    
    Plugin developers should inherit from this class and implement the required methods.
    """
    
    def __init__(self, context: PluginContext):
        """Initialize the plugin with context."""
        self.context = context
        self._enabled = False
        self._initialized = False
    
    @property
    @abc.abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    @abc.abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the plugin.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def cleanup(self):
        """Clean up plugin resources."""
        pass
    
    def enable(self) -> bool:
        """
        Enable the plugin.
        
        Returns:
            True if enabled successfully, False otherwise
        """
        if not self._initialized:
            if not self.initialize():
                return False
            self._initialized = True
        
        self._enabled = True
        self.on_enabled()
        return True
    
    def disable(self):
        """Disable the plugin."""
        self._enabled = False
        self.on_disabled()
    
    def is_enabled(self) -> bool:
        """Check if plugin is enabled."""
        return self._enabled
    
    def is_initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._initialized
    
    # Optional lifecycle methods
    def on_enabled(self):
        """Called when plugin is enabled."""
        pass
    
    def on_disabled(self):
        """Called when plugin is disabled."""
        pass
    
    def on_status_update(self, activity: str, details: str = "", state: str = ""):
        """Called when status is updated."""
        pass
    
    def on_error(self, error: Exception):
        """Called when an error occurs."""
        pass
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.context.plugin_config.get(key, default)
    
    def set_config_value(self, key: str, value: Any):
        """Set a configuration value."""
        self.context.plugin_config[key] = value
    
    def save_config(self):
        """Save plugin configuration."""
        config_file = self.context.plugin_data_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.context.plugin_config, f, indent=2)


class AIToolPlugin(BasePlugin):
    """
    Base class for AI tool integration plugins.
    """
    
    @abc.abstractmethod
    def detect_tool(self) -> bool:
        """
        Detect if the AI tool is available and running.
        
        Returns:
            True if tool is detected, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def get_tool_info(self) -> Dict[str, Any]:
        """
        Get information about the AI tool.
        
        Returns:
            Dictionary with tool information
        """
        pass
    
    @abc.abstractmethod
    def start_operation(self, operation_type: str, **kwargs) -> str:
        """
        Start an AI operation.
        
        Args:
            operation_type: Type of operation (e.g., 'generate', 'train', 'chat')
            **kwargs: Operation-specific parameters
            
        Returns:
            Operation ID for tracking
        """
        pass
    
    @abc.abstractmethod
    def update_operation_progress(self, operation_id: str, progress: float, status: str = ""):
        """
        Update operation progress.
        
        Args:
            operation_id: ID of the operation
            progress: Progress percentage (0-100)
            status: Current status description
        """
        pass
    
    @abc.abstractmethod
    def complete_operation(self, operation_id: str, result: Dict[str, Any]):
        """
        Complete an AI operation.
        
        Args:
            operation_id: ID of the operation
            result: Operation result
        """
        pass


class StatusEnhancerPlugin(BasePlugin):
    """
    Base class for status enhancement plugins.
    """
    
    @abc.abstractmethod
    def enhance_status(self, activity: str, details: str = "", state: str = "") -> Dict[str, Any]:
        """
        Enhance status information.
        
        Args:
            activity: Main activity text
            details: Details text
            state: State text
            
        Returns:
            Enhanced status dictionary
        """
        pass
    
    @abc.abstractmethod
    def get_status_suggestions(self, context: Dict[str, Any]) -> List[str]:
        """
        Get status suggestions based on context.
        
        Args:
            context: Current context information
            
        Returns:
            List of suggested status messages
        """
        pass


class AnalyticsPlugin(BasePlugin):
    """
    Base class for analytics plugins.
    """
    
    @abc.abstractmethod
    def track_event(self, event_name: str, properties: Dict[str, Any]):
        """
        Track an analytics event.
        
        Args:
            event_name: Name of the event
            properties: Event properties
        """
        pass
    
    @abc.abstractmethod
    def get_analytics(self, time_range: str = "24h") -> Dict[str, Any]:
        """
        Get analytics data.
        
        Args:
            time_range: Time range for data (e.g., "24h", "7d", "30d")
            
        Returns:
            Analytics data dictionary
        """
        pass


@dataclass
class LoadedPlugin:
    """Information about a loaded plugin."""
    plugin: BasePlugin
    metadata: PluginMetadata
    load_time: float
    enabled: bool = False
    error: Optional[str] = None


class PluginManager:
    """
    Plugin manager for loading, managing, and executing plugins.
    """
    
    def __init__(self, rpc_instance: AIFrameworkRPC, plugin_dir: str = "plugins"):
        self.rpc_instance = rpc_instance
        self.plugin_dir = Path(plugin_dir)
        self.plugin_dir.mkdir(exist_ok=True)
        
        self._loaded_plugins: Dict[str, LoadedPlugin] = {}
        self._event_bus = PluginEventBus()
        self._lock = threading.Lock()
        
        # Plugin data directory
        self.data_dir = self.plugin_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.PluginManager")
    
    def discover_plugins(self) -> List[str]:
        """
        Discover available plugins in the plugin directory.
        
        Returns:
            List of plugin file paths
        """
        plugin_files = []
        
        # Look for Python files
        for py_file in self.plugin_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            plugin_files.append(str(py_file))
        
        # Look for plugin packages
        for pkg_dir in self.plugin_dir.iterdir():
            if pkg_dir.is_dir() and not pkg_dir.name.startswith("__"):
                init_file = pkg_dir / "__init__.py"
                if init_file.exists():
                    plugin_files.append(str(init_file))
        
        return plugin_files
    
    def load_plugin(self, plugin_path: str) -> bool:
        """
        Load a plugin from file path.
        
        Args:
            plugin_path: Path to plugin file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location("plugin", plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin classes
            plugin_classes = []
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BasePlugin) and 
                    obj != BasePlugin and 
                    not inspect.isabstract(obj)):
                    plugin_classes.append(obj)
            
            if not plugin_classes:
                self.logger.error(f"No plugin classes found in {plugin_path}")
                return False
            
            # Load the first plugin class found
            plugin_class = plugin_classes[0]
            
            # Create plugin data directory
            plugin_name = plugin_path.split("/")[-1].replace(".py", "")
            plugin_data_dir = self.data_dir / plugin_name
            plugin_data_dir.mkdir(exist_ok=True)
            
            # Load plugin configuration
            config_file = plugin_data_dir / "config.json"
            plugin_config = {}
            if config_file.exists():
                with open(config_file, 'r') as f:
                    plugin_config = json.load(f)
            
            # Create plugin context
            context = PluginContext(
                rpc_instance=self.rpc_instance,
                plugin_config=plugin_config,
                plugin_data_dir=plugin_data_dir,
                event_bus=self._event_bus,
                logger=logging.getLogger(f"{__name__}.Plugin.{plugin_name}")
            )
            
            # Initialize plugin
            plugin = plugin_class(context)
            metadata = plugin.metadata
            
            # Validate plugin compatibility
            if not self._validate_plugin_compatibility(metadata):
                return False
            
            # Store loaded plugin
            loaded_plugin = LoadedPlugin(
                plugin=plugin,
                metadata=metadata,
                load_time=time.time()
            )
            
            with self._lock:
                self._loaded_plugins[plugin_name] = loaded_plugin
            
            # Subscribe to events
            self._setup_plugin_events(plugin_name, plugin)
            
            # Emit plugin loaded event
            self._event_bus.emit(PluginEvent.PLUGIN_LOADED, {
                'plugin_name': plugin_name,
                'metadata': metadata.to_dict()
            })
            
            self.logger.info(f"Loaded plugin: {plugin_name} v{metadata.version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin {plugin_path}: {e}")
            return False
    
    def _validate_plugin_compatibility(self, metadata: PluginMetadata) -> bool:
        """Validate plugin compatibility with current version."""
        # For now, just check minimum version
        # In a real implementation, this would do proper version comparison
        return True
    
    def _setup_plugin_events(self, plugin_name: str, plugin: BasePlugin):
        """Set up event subscriptions for a plugin."""
        # Subscribe to relevant events
        if hasattr(plugin, 'on_status_update'):
            self._event_bus.subscribe(PluginEvent.STATUS_UPDATE, 
                                    lambda data: plugin.on_status_update(
                                        data.get('activity', ''),
                                        data.get('details', ''),
                                        data.get('state', '')
                                    ))
        
        if hasattr(plugin, 'on_error'):
            self._event_bus.subscribe(PluginEvent.ERROR_OCCURRED,
                                    lambda data: plugin.on_error(data.get('error')))
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """
        Enable a loaded plugin.
        
        Args:
            plugin_name: Name of the plugin to enable
            
        Returns:
            True if enabled successfully, False otherwise
        """
        with self._lock:
            if plugin_name not in self._loaded_plugins:
                self.logger.error(f"Plugin {plugin_name} not loaded")
                return False
            
            loaded_plugin = self._loaded_plugins[plugin_name]
            
            try:
                success = loaded_plugin.plugin.enable()
                if success:
                    loaded_plugin.enabled = True
                    self.logger.info(f"Enabled plugin: {plugin_name}")
                else:
                    loaded_plugin.error = "Failed to enable"
                    self.logger.error(f"Failed to enable plugin: {plugin_name}")
                
                return success
                
            except Exception as e:
                loaded_plugin.error = str(e)
                self.logger.error(f"Error enabling plugin {plugin_name}: {e}")
                return False
    
    def disable_plugin(self, plugin_name: str):
        """
        Disable a plugin.
        
        Args:
            plugin_name: Name of the plugin to disable
        """
        with self._lock:
            if plugin_name not in self._loaded_plugins:
                return
            
            loaded_plugin = self._loaded_plugins[plugin_name]
            try:
                loaded_plugin.plugin.disable()
                loaded_plugin.enabled = False
                self.logger.info(f"Disabled plugin: {plugin_name}")
            except Exception as e:
                self.logger.error(f"Error disabling plugin {plugin_name}: {e}")
    
    def unload_plugin(self, plugin_name: str):
        """
        Unload a plugin completely.
        
        Args:
            plugin_name: Name of the plugin to unload
        """
        with self._lock:
            if plugin_name not in self._loaded_plugins:
                return
            
            loaded_plugin = self._loaded_plugins[plugin_name]
            
            try:
                # Disable if enabled
                if loaded_plugin.enabled:
                    loaded_plugin.plugin.disable()
                
                # Cleanup
                loaded_plugin.plugin.cleanup()
                
                # Remove from loaded plugins
                del self._loaded_plugins[plugin_name]
                
                # Emit plugin unloaded event
                self._event_bus.emit(PluginEvent.PLUGIN_UNLOADED, {
                    'plugin_name': plugin_name
                })
                
                self.logger.info(f"Unloaded plugin: {plugin_name}")
                
            except Exception as e:
                self.logger.error(f"Error unloading plugin {plugin_name}: {e}")
    
    def get_loaded_plugins(self) -> Dict[str, LoadedPlugin]:
        """Get all loaded plugins."""
        with self._lock:
            return self._loaded_plugins.copy()
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a specific plugin instance."""
        with self._lock:
            loaded = self._loaded_plugins.get(plugin_name)
            return loaded.plugin if loaded else None
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[BasePlugin]:
        """Get all plugins of a specific type."""
        plugins = []
        with self._lock:
            for loaded_plugin in self._loaded_plugins.values():
                if loaded_plugin.metadata.plugin_type == plugin_type:
                    plugins.append(loaded_plugin.plugin)
        return plugins
    
    def emit_event(self, event: PluginEvent, data: Dict[str, Any] = None):
        """Emit an event to all plugins."""
        self._event_bus.emit(event, data or {})
    
    def shutdown(self):
        """Shutdown plugin manager and cleanup all plugins."""
        with self._lock:
            for plugin_name in list(self._loaded_plugins.keys()):
                self.unload_plugin(plugin_name)


# Plugin SDK utilities
class PluginSDK:
    """SDK utilities for plugin developers."""
    
    @staticmethod
    def create_status_enhancer(name: str, version: str, description: str, 
                             author: str) -> Type[StatusEnhancerPlugin]:
        """
        Create a simple status enhancer plugin.
        
        Returns:
            A status enhancer plugin class
        """
        class SimpleStatusEnhancer(StatusEnhancerPlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name=name,
                    version=version,
                    description=description,
                    author=author,
                    plugin_type=PluginType.STATUS_ENHANCER
                )
        
        return SimpleStatusEnhancer
    
    @staticmethod
    def create_ai_tool_plugin(name: str, version: str, description: str,
                            author: str) -> Type[AIToolPlugin]:
        """
        Create a simple AI tool plugin.
        
        Returns:
            An AI tool plugin class
        """
        class SimpleAITool(AIToolPlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name=name,
                    version=version,
                    description=description,
                    author=author,
                    plugin_type=PluginType.AI_TOOL
                )
        
        return SimpleAITool


# Example plugin templates
class ExampleStatusEnhancer(StatusEnhancerPlugin):
    """Example status enhancer plugin."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Example Status Enhancer",
            version="1.0.0",
            description="An example status enhancement plugin",
            author="AIFrameworkRPC Team",
            plugin_type=PluginType.STATUS_ENHANCER
        )
    
    def initialize(self) -> bool:
        self.context.logger.info("Example status enhancer initialized")
        return True
    
    def cleanup(self):
        self.context.logger.info("Example status enhancer cleaned up")
    
    def enhance_status(self, activity: str, details: str = "", state: str = "") -> Dict[str, Any]:
        # Add emoji to activity
        enhanced_activity = f"🤖 {activity}"
        
        # Add timestamp to details
        if details:
            enhanced_details = f"{details} • {time.strftime('%H:%M')}"
        else:
            enhanced_details = f"Updated at {time.strftime('%H:%M')}"
        
        return {
            'activity': enhanced_activity,
            'details': enhanced_details,
            'state': state
        }
    
    def get_status_suggestions(self, context: Dict[str, Any]) -> List[str]:
        return [
            "Working on something amazing 🚀",
            "Creating AI magic ✨",
            "Training models 🧠",
            "Generating content 🎨"
        ]
