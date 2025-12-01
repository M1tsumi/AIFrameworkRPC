"""
Core AIFrameworkRPC implementation for Discord Rich Presence integration.
"""

import time
import logging
import threading
import weakref
from typing import Optional, Dict, Any, Callable
from threading import Thread, Event, Lock
from concurrent.futures import ThreadPoolExecutor
import json
import os
from functools import lru_cache
import asyncio

try:
    from pypresence import Presence
    PYPRESENCE_AVAILABLE = True
except ImportError:
    PYPRESENCE_AVAILABLE = False
    logging.warning("pypresence not available. Install with: pip install pypresence")

try:
    import discord
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    logging.warning("discord.py not available. Install with: pip install discord.py")


class ConnectionPool:
    """Thread-safe connection pool for Discord Rich Presence."""
    
    def __init__(self, max_connections: int = 5):
        self.max_connections = max_connections
        self._pool: Dict[str, Presence] = {}
        self._locks: Dict[str, Lock] = {}
        self._usage_count: Dict[str, int] = {}
        self._lock = Lock()
        
    def get_connection(self, client_id: str) -> Optional[Presence]:
        """Get or create a connection from the pool."""
        with self._lock:
            if client_id not in self._pool:
                if len(self._pool) >= self.max_connections:
                    # Remove least used connection
                    least_used = min(self._usage_count.items(), key=lambda x: x[1])
                    self._remove_connection(least_used[0])
                
                try:
                    connection = Presence(client_id)
                    self._pool[client_id] = connection
                    self._locks[client_id] = Lock()
                    self._usage_count[client_id] = 0
                except Exception as e:
                    logging.error(f"Failed to create connection for {client_id}: {e}")
                    return None
            
            self._usage_count[client_id] += 1
            return self._pool[client_id]
    
    def remove_connection(self, client_id: str):
        """Remove a connection from the pool."""
        with self._lock:
            self._remove_connection(client_id)
    
    def _remove_connection(self, client_id: str):
        """Internal method to remove connection."""
        if client_id in self._pool:
            try:
                self._pool[client_id].close()
            except:
                pass
            del self._pool[client_id]
            del self._locks[client_id]
            del self._usage_count[client_id]
    
    def get_lock(self, client_id: str) -> Lock:
        """Get the lock for a specific connection."""
        return self._locks.get(client_id, Lock())
    
    def clear_all(self):
        """Clear all connections."""
        with self._lock:
            for client_id in list(self._pool.keys()):
                self._remove_connection(client_id)


# Global connection pool
_connection_pool = ConnectionPool()


class PerformanceMonitor:
    """Performance monitoring and metrics collection."""
    
    def __init__(self):
        self._metrics = {
            'connection_attempts': 0,
            'successful_connections': 0,
            'failed_connections': 0,
            'status_updates': 0,
            'reconnections': 0,
            'avg_response_time': 0.0,
            'total_response_time': 0.0
        }
        self._lock = Lock()
        self._start_time = time.time()
    
    def record_connection_attempt(self):
        """Record a connection attempt."""
        with self._lock:
            self._metrics['connection_attempts'] += 1
    
    def record_successful_connection(self, response_time: float):
        """Record a successful connection."""
        with self._lock:
            self._metrics['successful_connections'] += 1
            self._metrics['total_response_time'] += response_time
            total = self._metrics['successful_connections']
            self._metrics['avg_response_time'] = self._metrics['total_response_time'] / total
    
    def record_failed_connection(self):
        """Record a failed connection."""
        with self._lock:
            self._metrics['failed_connections'] += 1
    
    def record_status_update(self):
        """Record a status update."""
        with self._lock:
            self._metrics['status_updates'] += 1
    
    def record_reconnection(self):
        """Record a reconnection."""
        with self._lock:
            self._metrics['reconnections'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        with self._lock:
            uptime = time.time() - self._start_time
            metrics = self._metrics.copy()
            metrics['uptime_seconds'] = uptime
            metrics['updates_per_second'] = self._metrics['status_updates'] / max(uptime, 1)
            return metrics
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            for key in self._metrics:
                if key != 'avg_response_time' and key != 'total_response_time':
                    self._metrics[key] = 0
            self._metrics['total_response_time'] = 0.0
            self._start_time = time.time()


# Global performance monitor
_performance_monitor = PerformanceMonitor()


class AIFrameworkRPC:
    """
    Main class for integrating Discord Rich Presence with AI tools.
    """
    
    def __init__(self, discord_client_id: str, default_status: str = "Working with AI tools",
                 auto_reconnect: bool = True, reconnect_delay: float = 5.0, 
                 max_reconnect_attempts: int = 5):
        """
        Initialize the RPC client.
        
        Args:
            discord_client_id: Discord application client ID
            default_status: Default status message
            auto_reconnect: Enable automatic reconnection
            reconnect_delay: Delay between reconnection attempts (seconds)
            max_reconnect_attempts: Maximum reconnection attempts
        """
        if not PYPRESENCE_AVAILABLE:
            raise ImportError("pypresence is required. Install with: pip install pypresence")
            
        self.client_id = discord_client_id
        self.default_status = default_status
        self.auto_reconnect = auto_reconnect
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_attempts = 0
        
        # Use connection pool
        self.rpc: Optional[Presence] = None
        self.connected = False
        self.stop_event = Event()
        self.update_thread: Optional[Thread] = None
        self.reconnect_thread: Optional[Thread] = None
        
        # Thread safety
        self._lock = Lock()
        
        # Performance optimization: cached status
        self._last_status = {}
        self._status_cache_timeout = 1.0  # seconds
        self._last_status_update = 0.0
        
        # Event handlers
        self.event_handlers: Dict[str, list] = {}
        
        # Current status
        self.current_activity = default_status
        self.current_details = ""
        self.current_state = ""
        self.start_time = time.time()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Performance monitoring
        self.performance_monitor = _performance_monitor
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rpc-")
        
    def connect(self) -> bool:
        """
        Connect to Discord Rich Presence with performance optimizations and auto-reconnect.
        
        Returns:
            True if connection successful, False otherwise
        """
        with self._lock:
            self.performance_monitor.record_connection_attempt()
            start_time = time.time()
            
            try:
                # Get connection from pool
                self.rpc = _connection_pool.get_connection(self.client_id)
                if not self.rpc:
                    self.performance_monitor.record_failed_connection()
                    return False
                
                # Connect with timeout
                connection_lock = _connection_pool.get_lock(self.client_id)
                with connection_lock:
                    self.rpc.connect()
                
                self.connected = True
                self.start_time = time.time()
                self._reconnect_attempts = 0
                
                # Set initial status
                self.update_status(activity=self.default_status)
                
                # Start monitoring thread for auto-reconnect
                if self.auto_reconnect:
                    self._start_connection_monitor()
                
                response_time = time.time() - start_time
                self.performance_monitor.record_successful_connection(response_time)
                
                self.logger.info(f"Connected to Discord Rich Presence with client ID {self.client_id} in {response_time:.3f}s")
                return True
                
            except Exception as e:
                self.performance_monitor.record_failed_connection()
                self.logger.error(f"Failed to connect to Discord Rich Presence: {e}")
                self.connected = False
                
                # Start auto-reconnect if enabled
                if self.auto_reconnect and self._reconnect_attempts < self.max_reconnect_attempts:
                    self._schedule_reconnect()
                
                return False
    
    def _start_connection_monitor(self):
        """Start a background thread to monitor connection health."""
        if self.reconnect_thread and self.reconnect_thread.is_alive():
            return
        
        self.reconnect_thread = Thread(target=self._monitor_connection, daemon=True)
        self.reconnect_thread.start()
    
    def _monitor_connection(self):
        """Monitor connection health and attempt reconnection if needed."""
        while not self.stop_event.is_set() and self.connected:
            try:
                # Test connection with a lightweight operation
                if self.rpc:
                    self.rpc.update(self.current_activity, self.current_details, self.current_state)
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.warning(f"Connection health check failed: {e}")
                if self.auto_reconnect:
                    self._schedule_reconnect()
                break
    
    def _schedule_reconnect(self):
        """Schedule a reconnection attempt."""
        if self._reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error("Max reconnection attempts reached")
            return
        
        self._reconnect_attempts += 1
        self.performance_monitor.record_reconnection()
        
        self.logger.info(f"Scheduling reconnection attempt {self._reconnect_attempts}/{self.max_reconnect_attempts} in {self.reconnect_delay}s")
        
        def reconnect_delayed():
            time.sleep(self.reconnect_delay)
            if not self.stop_event.is_set():
                self.connect()
        
        # Use thread pool to avoid blocking
        self.executor.submit(reconnect_delayed)
    
    def disconnect(self):
        """Disconnect from Discord Rich Presence with cleanup."""
        self.stop_event.set()
        
        # Stop reconnection thread
        if self.reconnect_thread and self.reconnect_thread.is_alive():
            self.reconnect_thread.join(timeout=1)
        
        # Stop update thread
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1)
        
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)
        
        # Return connection to pool (don't close it)
        if self.rpc:
            try:
                # Connection remains in pool for reuse
                pass
            except:
                pass
        
        self.connected = False
        self.logger.info("Disconnected from Discord Rich Presence")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for this instance.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = self.performance_monitor.get_metrics()
        metrics.update({
            'client_id': self.client_id,
            'connected': self.connected,
            'reconnect_attempts': self._reconnect_attempts,
            'last_status_update': self._last_status_update,
            'cache_hit_ratio': getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_requests', 1), 1)
        })
        return metrics
    
    def reset_performance_metrics(self):
        """Reset performance metrics for this instance."""
        self.performance_monitor.reset()
        self._reconnect_attempts = 0
        self._last_status_update = 0.0
        self._last_status = {}
        self._cache_hits = 0
        self._cache_requests = 0
    
    def optimize_performance(self, cache_timeout: float = 1.0, max_workers: int = 2):
        """
        Optimize performance settings.
        
        Args:
            cache_timeout: Status cache timeout in seconds
            max_workers: Maximum number of worker threads
        """
        self._status_cache_timeout = cache_timeout
        
        # Recreate executor with new settings
        if self.executor:
            self.executor.shutdown(wait=True)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rpc-")
        
        self.logger.info(f"Performance optimized: cache_timeout={cache_timeout}s, max_workers={max_workers}")
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get detailed connection information.
        
        Returns:
            Dictionary with connection details
        """
        return {
            'client_id': self.client_id,
            'connected': self.connected,
            'auto_reconnect': self.auto_reconnect,
            'reconnect_delay': self.reconnect_delay,
            'max_reconnect_attempts': self.max_reconnect_attempts,
            'current_reconnect_attempts': self._reconnect_attempts,
            'uptime': time.time() - self.start_time if self.connected else 0,
            'current_activity': self.current_activity,
            'pool_size': len(_connection_pool._pool),
            'pool_connections': list(_connection_pool._pool.keys())
        }
    
    def update_status(self, activity: str, details: str = "", state: str = "", 
                     large_image_key: str = "", small_image_key: str = "",
                     large_image_text: str = "", small_image_text: str = "", force_update: bool = False):
        """
        Update Discord Rich Presence status with performance optimizations.
        
        Args:
            activity: Main activity text
            details: Details text (second line)
            state: State text (third line)
            large_image_key: Key for large image
            small_image_key: Key for small image
            large_image_text: Hover text for large image
            small_image_text: Hover text for small image
            force_update: Force update even if status hasn't changed
        """
        # Performance optimization: skip if status hasn't changed and not forced
        current_status = {
            'activity': activity,
            'details': details,
            'state': state,
            'large_image_key': large_image_key,
            'small_image_key': small_image_key
        }
        
        current_time = time.time()
        
        if not force_update and current_status == self._last_status:
            # Check if enough time has passed for cache timeout
            if current_time - self._last_status_update < self._status_cache_timeout:
                return
        
        with self._lock:
            if not self.connected or not self.rpc:
                self.logger.warning("Not connected to Discord Rich Presence")
                return
                
            self.current_activity = activity
            self.current_details = details
            self.current_state = state
            
            try:
                presence_data = {
                    "activity": {
                        "details": details,
                        "state": state,
                        "timestamps": {
                            "start": int(self.start_time)
                        }
                    }
                }
                
                # Add images if provided
                if large_image_key:
                    presence_data["activity"]["assets"] = {
                        "large_image": large_image_key,
                        "large_text": large_image_text
                    }
                    if small_image_key:
                        presence_data["activity"]["assets"]["small_image"] = small_image_key
                        presence_data["activity"]["assets"]["small_text"] = small_image_text
                
                # Use thread pool for async update to avoid blocking
                def update_async():
                    try:
                        self.rpc.update(**presence_data["activity"])
                        self.performance_monitor.record_status_update()
                    except Exception as e:
                        self.logger.error(f"Failed to update Discord status: {e}")
                        # Try to reconnect if update fails
                        if self.auto_reconnect:
                            self._schedule_reconnect()
                
                self.executor.submit(update_async)
                
                # Update cache
                self._last_status = current_status.copy()
                self._last_status_update = current_time
                
                self.logger.debug(f"Updated Discord status: {activity}")
                
            except Exception as e:
                self.logger.error(f"Failed to prepare Discord status update: {e}")
    
    @lru_cache(maxsize=128)
    def _format_status_text(self, text: str, max_length: int = 128) -> str:
        """
        Format and cache status text to avoid repeated processing.
        
        Args:
            text: Text to format
            max_length: Maximum length
            
        Returns:
            Formatted text
        """
        if len(text) > max_length:
            return text[:max_length-3] + "..."
        return text
    
    def clear_status(self):
        """Clear Discord Rich Presence status."""
        if self.connected and self.rpc:
            try:
                self.rpc.clear()
                self.logger.info("Cleared Discord status")
            except Exception as e:
                self.logger.error(f"Failed to clear Discord status: {e}")
    
    def on_event(self, event_name: str) -> Callable:
        """
        Decorator for registering event handlers.
        
        Args:
            event_name: Name of the event to handle
            
        Returns:
            Decorator function
        """
        def decorator(func):
            if event_name not in self.event_handlers:
                self.event_handlers[event_name] = []
            self.event_handlers[event_name].append(func)
            return func
        return decorator
    
    def emit_event(self, event_name: str, *args, **kwargs):
        """
        Emit an event to all registered handlers.
        
        Args:
            event_name: Name of the event
            *args: Arguments to pass to handlers
            **kwargs: Keyword arguments to pass to handlers
        """
        if event_name in self.event_handlers:
            for handler in self.event_handlers[event_name]:
                try:
                    handler(*args, **kwargs)
                except Exception as e:
                    self.logger.error(f"Error in event handler for {event_name}: {e}")
    
    def share_to_channel(self, content: str, channel_id: str = None, 
                        image_path: str = None, bot_token: str = None):
        """
        Share content to Discord channel (requires bot token).
        
        Args:
            content: Text content to share
            channel_id: Discord channel ID
            image_path: Path to image file (optional)
            bot_token: Discord bot token (if not in environment)
        """
        if not DISCORD_AVAILABLE:
            self.logger.error("discord.py not available. Install with: pip install discord.py")
            return
            
        bot_token = bot_token or os.getenv("DISCORD_BOT_TOKEN")
        if not bot_token:
            self.logger.error("Discord bot token required for sharing to channels")
            return
            
        if not channel_id:
            self.logger.error("Channel ID required for sharing")
            return
        
        # This would require async implementation
        # For now, just log the intent
        self.logger.info(f"Would share to channel {channel_id}: {content}")
        if image_path:
            self.logger.info(f"With image: {image_path}")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.disconnect()
