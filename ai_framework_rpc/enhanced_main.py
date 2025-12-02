"""
Enhanced AIFrameworkRPC v0.2.0 - Main integration module
"""

import time
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

# Import all enhanced components
from .core import AIFrameworkRPC
from .enhanced_connection import _enhanced_connection_pool
from .plugin_system import PluginManager
from .predictive_cache import _predictive_cache
from .error_recovery import _global_retry_manager
from .performance_profiler import _global_profiler, profile_function
from .web_dashboard import WebDashboard, DashboardConfig, start_dashboard, stop_dashboard
from .security import SecurityManager, SecurityConfig, initialize_security, get_security_manager
from .config_v2 import EnhancedConfig, initialize_config, get_config
from .migrate import MigrationManager


class EnhancedAIFrameworkRPC(AIFrameworkRPC):
    """
    Enhanced AIFrameworkRPC with all v0.2.0 features.
    
    This is the main class that integrates all the enhanced components:
    - Intelligent connection management
    - Plugin system
    - Predictive caching
    - Error recovery
    - Performance profiling
    - Web dashboard
    - Enhanced security
    - Advanced configuration
    """
    
    def __init__(self, discord_client_id: str, config_file: Optional[str] = None,
                 enable_dashboard: bool = True, enable_plugins: bool = True,
                 enable_security: bool = True, enable_profiling: bool = True):
        """
        Initialize Enhanced AIFrameworkRPC with all features.
        
        Args:
            discord_client_id: Discord application client ID
            config_file: Path to configuration file
            enable_dashboard: Enable web dashboard
            enable_plugins: Enable plugin system
            enable_security: Enable enhanced security
            enable_profiling: Enable performance profiling
        """
        # Initialize configuration first
        self.config = initialize_config(config_file)
        
        # Get configuration values
        discord_config = self.config.discord
        
        # Initialize base class with enhanced settings
        super().__init__(
            discord_client_id=discord_client_id,
            default_status=discord_config.default_status,
            auto_reconnect=discord_config.auto_reconnect,
            reconnect_delay=discord_config.reconnect_delay,
            max_reconnect_attempts=discord_config.max_reconnect_attempts
        )
        
        # Enhanced components
        self.plugin_manager: Optional[PluginManager] = None
        self.dashboard: Optional[WebDashboard] = None
        self.security_manager: Optional[SecurityManager] = None
        self.migration_manager: Optional[MigrationManager] = None
        
        # Feature flags
        self.enable_dashboard = enable_dashboard
        self.enable_plugins = enable_plugins
        self.enable_security = enable_security
        self.enable_profiling = enable_profiling
        
        # Initialize enhanced features
        self._initialize_enhanced_features()
        
        self.logger.info("Enhanced AIFrameworkRPC v0.2.0 initialized")
    
    def _initialize_enhanced_features(self):
        """Initialize all enhanced features."""
        
        # Initialize security
        if self.enable_security:
            security_config = SecurityConfig(
                encryption_enabled=self.config.security.encryption_enabled,
                audit_logging=self.config.security.audit_logging,
                session_timeout=self.config.security.session_timeout
            )
            self.security_manager = initialize_security(security_config)
        
        # Initialize plugin system
        if self.enable_plugins:
            self.plugin_manager = PluginManager(self, self.config.plugins.plugin_directories[0])
            
            # Auto-load plugins if enabled
            if self.config.plugins.auto_load:
                self._auto_load_plugins()
        
        # Initialize performance profiling
        if self.enable_profiling:
            # Configure profiler based on settings
            self._configure_profiler()
        
        # Initialize web dashboard
        if self.enable_dashboard:
            dashboard_config = DashboardConfig(
                host=self.config.web_dashboard.host,
                port=self.config.web_dashboard.port,
                enable_auth=self.config.web_dashboard.enable_auth,
                update_interval=self.config.web_dashboard.update_interval
            )
            self.dashboard = start_dashboard(dashboard_config)
        
        # Initialize migration manager
        self.migration_manager = MigrationManager(".")
        
        # Setup monitoring if enabled
        if self.config.monitoring.enabled:
            self._setup_monitoring()
    
    def _auto_load_plugins(self):
        """Auto-load plugins from configured directories."""
        discovered_plugins = self.plugin_manager.discover_plugins()
        
        for plugin_path in discovered_plugins:
            try:
                if self.plugin_manager.load_plugin(plugin_path):
                    plugin_name = Path(plugin_path).stem
                    if self.plugin_manager.enable_plugin(plugin_name):
                        self.logger.info(f"Auto-loaded plugin: {plugin_name}")
            except Exception as e:
                self.logger.error(f"Failed to auto-load plugin {plugin_path}: {e}")
    
    def _configure_profiler(self):
        """Configure performance profiler."""
        # Set bottleneck thresholds
        for metric, threshold in self.config.monitoring.alert_thresholds.items():
            _global_profiler.set_bottleneck_threshold(metric, threshold)
        
        # Add alert callback for bottlenecks
        def bottleneck_alert(bottleneck):
            self.logger.warning(f"Bottleneck detected: {bottleneck.description}")
        
        _global_profiler.add_alert_callback(bottleneck_alert)
    
    def _setup_monitoring(self):
        """Setup monitoring and analytics."""
        # Enable system monitoring
        if hasattr(_global_profiler, 'system_monitor') and _global_profiler.system_monitor:
            _global_profiler.system_monitor.start_monitoring()
    
    @profile_function("enhanced_connect")
    def connect(self) -> bool:
        """
        Enhanced connect method with profiling and error recovery.
        
        Returns:
            True if connection successful, False otherwise
        """
        # Use retry manager for connection
        future = _global_retry_manager.execute_with_retry(
            super().connect,
            operation="discord_connection"
        )
        
        try:
            result = future.result()
            
            if result:
                # Emit connection established event
                if self.plugin_manager:
                    self.plugin_manager.emit_event("connection_established", {
                        'client_id': self.client_id,
                        'timestamp': time.time()
                    })
                
                self.logger.info("Enhanced connection established successfully")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced connection failed: {e}")
            
            # Emit connection lost event
            if self.plugin_manager:
                self.plugin_manager.emit_event("connection_lost", {
                    'client_id': self.client_id,
                    'error': str(e),
                    'timestamp': time.time()
                })
            
            return False
    
    @profile_function("enhanced_update_status")
    def update_status(self, activity: str, details: str = "", state: str = "", 
                     large_image_key: str = "", small_image_key: str = "",
                     large_image_text: str = "", small_image_text: str = "", 
                     force_update: bool = False):
        """
        Enhanced status update with plugin support and caching.
        """
        # Apply status enhancements from plugins
        if self.plugin_manager:
            enhancer_plugins = self.plugin_manager.get_plugins_by_type("status_enhancer")
            
            for plugin in enhancer_plugins:
                try:
                    enhanced = plugin.enhance_status(activity, details, state)
                    activity = enhanced.get('activity', activity)
                    details = enhanced.get('details', details)
                    state = enhanced.get('state', state)
                except Exception as e:
                    self.logger.error(f"Status enhancement plugin error: {e}")
        
        # Use predictive cache for status updates
        cache_key = f"status_{hash((activity, details, state))}"
        
        # Check cache first if not forcing update
        if not force_update:
            cached_status = _predictive_cache.get(cache_key)
            if cached_status is not None:
                self.logger.debug("Status retrieved from cache")
                return
        
        # Update status using parent method
        super().update_status(
            activity, details, state,
            large_image_key, small_image_key,
            large_image_text, small_image_text,
            force_update
        )
        
        # Cache the status update
        status_data = {
            'activity': activity,
            'details': details,
            'state': state,
            'large_image_key': large_image_key,
            'small_image_key': small_image_key,
            'timestamp': time.time()
        }
        _predictive_cache.put(cache_key, status_data, ttl=300)  # Cache for 5 minutes
        
        # Emit status update event
        if self.plugin_manager:
            self.plugin_manager.emit_event("status_update", status_data)
    
    def get_status_suggestions(self, context: Dict[str, Any] = None) -> List[str]:
        """
        Get status suggestions from plugins.
        
        Args:
            context: Context information for suggestions
            
        Returns:
            List of status suggestions
        """
        suggestions = []
        
        if self.plugin_manager:
            enhancer_plugins = self.plugin_manager.get_plugins_by_type("status_enhancer")
            
            for plugin in enhancer_plugins:
                try:
                    plugin_suggestions = plugin.get_status_suggestions(context or {})
                    suggestions.extend(plugin_suggestions)
                except Exception as e:
                    self.logger.error(f"Status suggestion plugin error: {e}")
        
        # Add default suggestions
        if not suggestions:
            suggestions = [
                "Working with AI tools 🤖",
                "Creating something amazing ✨",
                "Processing data 📊",
                "Training models 🧠"
            ]
        
        return suggestions[:10]  # Return top 10 suggestions
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = super().get_performance_metrics()
        
        # Add enhanced metrics
        metrics.update({
            'connection_pool': _enhanced_connection_pool.get_pool_stats(),
            'cache': _predictive_cache.get_stats().to_dict() if hasattr(_predictive_cache.get_stats(), 'to_dict') else _predictive_cache.get_stats().__dict__,
            'retry_manager': _global_retry_manager.get_retry_stats(),
            'profiler': _global_profiler.get_performance_report(),
            'dashboard': {
                'enabled': self.enable_dashboard,
                'url': self.dashboard.get_url() if self.dashboard else None,
                'connections': self.dashboard.websocket_manager.get_connection_count() if self.dashboard else 0
            },
            'plugins': {
                'enabled': self.enable_plugins,
                'loaded': len(self.plugin_manager.get_loaded_plugins()) if self.plugin_manager else 0
            },
            'security': {
                'enabled': self.enable_security,
                'status': self.security_manager.get_security_status() if self.security_manager else {}
            }
        })
        
        return metrics
    
    def optimize_performance(self):
        """Optimize performance across all components."""
        self.logger.info("Starting performance optimization...")
        
        # Optimize connection pool
        _enhanced_connection_pool.optimize_pool()
        
        # Optimize cache
        _predictive_cache.optimize()
        
        # Optimize retry manager
        # (No specific optimization needed for retry manager)
        
        # Clean up expired sessions if security is enabled
        if self.security_manager:
            expired_sessions = self.security_manager.cleanup_expired_sessions()
            if expired_sessions > 0:
                self.logger.info(f"Cleaned up {expired_sessions} expired sessions")
        
        self.logger.info("Performance optimization completed")
    
    def run_health_check(self) -> Dict[str, Any]:
        """
        Run comprehensive health check.
        
        Returns:
            Health check results
        """
        health_status = {
            'overall': 'healthy',
            'timestamp': time.time(),
            'components': {}
        }
        
        # Check Discord connection
        health_status['components']['discord'] = {
            'status': 'healthy' if self.connected else 'unhealthy',
            'connected': self.connected,
            'client_id': self.client_id
        }
        
        # Check connection pool
        pool_stats = _enhanced_connection_pool.get_pool_stats()
        health_status['components']['connection_pool'] = {
            'status': 'healthy',
            'active_connections': pool_stats['active_connections'],
            'total_connections': pool_stats['total_connections']
        }
        
        # Check cache
        cache_stats = _predictive_cache.get_stats()
        health_status['components']['cache'] = {
            'status': 'healthy',
            'hit_rate': cache_stats.hit_rate,
            'cache_size': cache_stats.cache_size
        }
        
        # Check plugins
        if self.plugin_manager:
            loaded_plugins = self.plugin_manager.get_loaded_plugins()
            health_status['components']['plugins'] = {
                'status': 'healthy',
                'loaded_count': len(loaded_plugins),
                'enabled_count': len([p for p in loaded_plugins.values() if p.enabled])
            }
        
        # Check dashboard
        if self.dashboard:
            health_status['components']['dashboard'] = {
                'status': 'healthy' if self.dashboard.is_running() else 'unhealthy',
                'running': self.dashboard.is_running(),
                'url': self.dashboard.get_url()
            }
        
        # Check security
        if self.security_manager:
            security_status = self.security_manager.get_security_status()
            health_status['components']['security'] = {
                'status': 'healthy',
                'encryption_enabled': security_status['encryption']['encryption_initialized'],
                'active_sessions': security_status['sessions']['active_sessions']
            }
        
        # Determine overall status
        component_statuses = [comp['status'] for comp in health_status['components'].values()]
        if any(status == 'unhealthy' for status in component_statuses):
            health_status['overall'] = 'unhealthy'
        elif any(status == 'warning' for status in component_statuses):
            health_status['overall'] = 'warning'
        
        return health_status
    
    def get_dashboard_url(self) -> Optional[str]:
        """Get the dashboard URL if enabled."""
        if self.dashboard:
            return self.dashboard.get_url()
        return None
    
    def load_plugin(self, plugin_path: str) -> bool:
        """
        Load a plugin.
        
        Args:
            plugin_path: Path to plugin file
            
        Returns:
            True if loaded successfully
        """
        if not self.plugin_manager:
            self.logger.error("Plugin system is not enabled")
            return False
        
        return self.plugin_manager.load_plugin(plugin_path)
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """
        Enable a plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            True if enabled successfully
        """
        if not self.plugin_manager:
            self.logger.error("Plugin system is not enabled")
            return False
        
        return self.plugin_manager.enable_plugin(plugin_name)
    
    def disable_plugin(self, plugin_name: str):
        """Disable a plugin."""
        if self.plugin_manager:
            self.plugin_manager.disable_plugin(plugin_name)
    
    def get_loaded_plugins(self) -> Dict[str, Any]:
        """Get information about loaded plugins."""
        if not self.plugin_manager:
            return {}
        
        loaded = self.plugin_manager.get_loaded_plugins()
        return {name: asdict(plugin.metadata) for name, plugin in loaded.items()}
    
    def store_secure_data(self, key: str, value: Any):
        """Store data securely."""
        if self.security_manager:
            self.security_manager.store_sensitive_data(key, value)
        else:
            self.logger.warning("Security is not enabled")
    
    def retrieve_secure_data(self, key: str, default: Any = None) -> Any:
        """Retrieve secure data."""
        if self.security_manager:
            return self.security_manager.retrieve_sensitive_data(key, default=default)
        else:
            self.logger.warning("Security is not enabled")
            return default
    
    def disconnect(self):
        """Enhanced disconnect with cleanup."""
        # Emit disconnect event
        if self.plugin_manager:
            self.plugin_manager.emit_event("connection_lost", {
                'client_id': self.client_id,
                'timestamp': time.time(),
                'reason': 'manual_disconnect'
            })
        
        # Call parent disconnect
        super().disconnect()
        
        self.logger.info("Enhanced AIFrameworkRPC disconnected")
    
    def shutdown(self):
        """Shutdown all enhanced components."""
        self.logger.info("Shutting down Enhanced AIFrameworkRPC...")
        
        # Disconnect first
        self.disconnect()
        
        # Shutdown plugin manager
        if self.plugin_manager:
            self.plugin_manager.shutdown()
        
        # Stop dashboard
        if self.dashboard:
            stop_dashboard()
        
        # Shutdown security
        if self.security_manager:
            from .security import shutdown_security
            shutdown_security()
        
        # Shutdown configuration
        from .config_v2 import shutdown_config
        shutdown_config()
        
        self.logger.info("Enhanced AIFrameworkRPC shutdown completed")
    
    def __enter__(self):
        """Enhanced context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Enhanced context manager exit."""
        self.shutdown()


# Convenience function for quick initialization
def create_enhanced_rpc(discord_client_id: str, config_file: Optional[str] = None,
                       **kwargs) -> EnhancedAIFrameworkRPC:
    """
    Create an Enhanced AIFrameworkRPC instance.
    
    Args:
        discord_client_id: Discord application client ID
        config_file: Path to configuration file
        **kwargs: Additional arguments for EnhancedAIFrameworkRPC
        
    Returns:
        EnhancedAIFrameworkRPC instance
    """
    return EnhancedAIFrameworkRPC(discord_client_id, config_file, **kwargs)


# Quick start function
def quick_start(discord_client_id: str, status: str = "Working with AI tools") -> EnhancedAIFrameworkRPC:
    """
    Quick start with sensible defaults.
    
    Args:
        discord_client_id: Discord application client ID
        status: Default status message
        
    Returns:
        Enhanced AIFrameworkRPC instance
    """
    rpc = EnhancedAIFrameworkRPC(discord_client_id)
    rpc.connect()
    rpc.update_status(activity=status)
    return rpc
