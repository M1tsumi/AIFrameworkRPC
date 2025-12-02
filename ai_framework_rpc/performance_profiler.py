"""
Real-time performance profiling and bottleneck detection for AIFrameworkRPC v0.2.0
"""

import time
import threading
import psutil
import gc
import traceback
import statistics
from typing import Dict, Any, List, Optional, Callable, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import logging
import inspect
import weakref


@dataclass
class PerformanceMetric:
    """A single performance metric measurement."""
    name: str
    value: float
    unit: str
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp,
            'tags': self.tags
        }


@dataclass
class FunctionProfile:
    """Profile information for a function."""
    function_name: str
    call_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    last_called: float = 0.0
    error_count: int = 0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    
    def update(self, duration: float, memory_delta: float = 0, cpu_delta: float = 0):
        """Update profile with new measurement."""
        self.call_count += 1
        self.total_time += duration
        self.avg_time = self.total_time / self.call_count
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.last_called = time.time()
        self.memory_usage += memory_delta
        self.cpu_usage += cpu_delta
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'function_name': self.function_name,
            'call_count': self.call_count,
            'total_time': self.total_time,
            'avg_time': self.avg_time,
            'min_time': self.min_time if self.min_time != float('inf') else 0,
            'max_time': self.max_time,
            'last_called': self.last_called,
            'error_count': self.error_count,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage
        }


@dataclass
class BottleneckInfo:
    """Information about a detected bottleneck."""
    type: str  # 'memory', 'cpu', 'io', 'network', 'function'
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: float
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'type': self.type,
            'severity': self.severity,
            'description': self.description,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'timestamp': self.timestamp,
            'suggestions': self.suggestions
        }


class SystemMonitor:
    """Monitors system resources (CPU, memory, etc.)."""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.process = psutil.Process()
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Metrics storage
        self.cpu_history: deque = deque(maxlen=300)  # 5 minutes at 1s intervals
        self.memory_history: deque = deque(maxlen=300)
        self.io_history: deque = deque(maxlen=300)
        self.network_history: deque = deque(maxlen=300)
        
        self.lock = threading.Lock()
    
    def start_monitoring(self):
        """Start system monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="SystemMonitor"
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        self.stop_event.set()
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            try:
                timestamp = time.time()
                
                # CPU usage
                cpu_percent = self.process.cpu_percent()
                self.cpu_history.append((timestamp, cpu_percent))
                
                # Memory usage
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.memory_history.append((timestamp, memory_mb))
                
                # I/O usage
                try:
                    io_counters = self.process.io_counters()
                    io_mb = (io_counters.read_bytes + io_counters.write_bytes) / 1024 / 1024
                    self.io_history.append((timestamp, io_mb))
                except (AttributeError, OSError):
                    self.io_history.append((timestamp, 0))
                
                # Network usage (if available)
                try:
                    network_io = psutil.net_io_counters()
                    network_mb = (network_io.bytes_sent + network_io.bytes_recv) / 1024 / 1024
                    self.network_history.append((timestamp, network_mb))
                except (AttributeError, OSError):
                    self.network_history.append((timestamp, 0))
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logging.error(f"System monitor error: {e}")
                time.sleep(self.update_interval)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        with self.lock:
            cpu = self.cpu_history[-1][1] if self.cpu_history else 0
            memory = self.memory_history[-1][1] if self.memory_history else 0
            io = self.io_history[-1][1] if self.io_history else 0
            network = self.network_history[-1][1] if self.network_history else 0
            
            return {
                'cpu_percent': cpu,
                'memory_mb': memory,
                'io_mb': io,
                'network_mb': network
            }
    
    def get_average_metrics(self, duration: float = 60.0) -> Dict[str, float]:
        """Get average metrics over the specified duration."""
        cutoff_time = time.time() - duration
        
        with self.lock:
            # Filter recent metrics
            recent_cpu = [value for timestamp, value in self.cpu_history if timestamp > cutoff_time]
            recent_memory = [value for timestamp, value in self.memory_history if timestamp > cutoff_time]
            recent_io = [value for timestamp, value in self.io_history if timestamp > cutoff_time]
            recent_network = [value for timestamp, value in self.network_history if timestamp > cutoff_time]
            
            return {
                'avg_cpu_percent': statistics.mean(recent_cpu) if recent_cpu else 0,
                'avg_memory_mb': statistics.mean(recent_memory) if recent_memory else 0,
                'avg_io_mb': statistics.mean(recent_io) if recent_io else 0,
                'avg_network_mb': statistics.mean(recent_network) if recent_network else 0
            }


class PerformanceProfiler:
    """
    Real-time performance profiler with bottleneck detection.
    
    Features:
    - Function-level profiling
    - System resource monitoring
    - Automatic bottleneck detection
    - Performance trend analysis
    - Optimization suggestions
    - Real-time alerts
    """
    
    def __init__(self, enable_system_monitor: bool = True,
                 profile_functions: bool = True,
                 bottleneck_detection: bool = True):
        self.enable_system_monitor = enable_system_monitor
        self.profile_functions = profile_functions
        self.bottleneck_detection = bottleneck_detection
        
        # System monitoring
        self.system_monitor = SystemMonitor() if enable_system_monitor else None
        
        # Function profiling
        self.function_profiles: Dict[str, FunctionProfile] = {}
        self.profile_lock = threading.Lock()
        
        # Bottleneck detection
        self.bottlenecks: deque = deque(maxlen=100)
        self.bottleneck_thresholds = {
            'cpu_percent': 80.0,
            'memory_mb': 500.0,
            'function_avg_time': 1.0,
            'function_call_count': 1000,
            'error_rate': 0.1
        }
        
        # Metrics collection
        self.metrics: deque = deque(maxlen=10000)
        self.metrics_lock = threading.Lock()
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[BottleneckInfo], None]] = []
        
        # Background processing
        self.analysis_thread: Optional[threading.Thread] = None
        self.stop_analysis = threading.Event()
        
        # Start background services
        if self.system_monitor:
            self.system_monitor.start_monitoring()
        
        if self.bottleneck_detection:
            self._start_analysis_thread()
    
    def profile_function(self, name: Optional[str] = None):
        """
        Decorator for profiling function performance.
        
        Args:
            name: Custom name for the function profile
        """
        def decorator(func):
            profile_name = name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.profile_functions:
                    return func(*args, **kwargs)
                
                start_time = time.time()
                start_memory = 0
                
                # Get initial memory if system monitoring is available
                if self.system_monitor:
                    start_memory = self.system_monitor.process.memory_info().rss
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    # Record error
                    with self.profile_lock:
                        if profile_name not in self.function_profiles:
                            self.function_profiles[profile_name] = FunctionProfile(profile_name)
                        self.function_profiles[profile_name].error_count += 1
                    raise
                finally:
                    # Calculate performance metrics
                    duration = time.time() - start_time
                    
                    if self.system_monitor:
                        end_memory = self.system_monitor.process.memory_info().rss
                        memory_delta = (end_memory - start_memory) / 1024 / 1024  # MB
                    else:
                        memory_delta = 0
                    
                    # Update function profile
                    with self.profile_lock:
                        if profile_name not in self.function_profiles:
                            self.function_profiles[profile_name] = FunctionProfile(profile_name)
                        
                        self.function_profiles[profile_name].update(duration, memory_delta)
                    
                    # Record metric
                    self.record_metric(
                        f"function_duration_{profile_name}",
                        duration,
                        "seconds",
                        tags={"function": profile_name}
                    )
            
            return wrapper
        return decorator
    
    def record_metric(self, name: str, value: float, unit: str, 
                     tags: Dict[str, str] = None):
        """
        Record a custom performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            tags: Additional tags for categorization
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            tags=tags or {}
        )
        
        with self.metrics_lock:
            self.metrics.append(metric)
    
    def _start_analysis_thread(self):
        """Start the bottleneck analysis thread."""
        self.analysis_thread = threading.Thread(
            target=self._analysis_loop,
            daemon=True,
            name="PerformanceAnalyzer"
        )
        self.analysis_thread.start()
    
    def _analysis_loop(self):
        """Main analysis loop for bottleneck detection."""
        while not self.stop_analysis.is_set():
            try:
                self._detect_bottlenecks()
                time.sleep(5)  # Analyze every 5 seconds
            except Exception as e:
                logging.error(f"Performance analysis error: {e}")
                time.sleep(10)
    
    def _detect_bottlenecks(self):
        """Detect performance bottlenecks."""
        current_time = time.time()
        
        # Check system metrics
        if self.system_monitor:
            system_metrics = self.system_monitor.get_current_metrics()
            
            # CPU bottleneck
            if system_metrics['cpu_percent'] > self.bottleneck_thresholds['cpu_percent']:
                bottleneck = BottleneckInfo(
                    type='cpu',
                    severity='high' if system_metrics['cpu_percent'] > 95 else 'medium',
                    description=f"High CPU usage detected: {system_metrics['cpu_percent']:.1f}%",
                    metric_name='cpu_percent',
                    current_value=system_metrics['cpu_percent'],
                    threshold=self.bottleneck_thresholds['cpu_percent'],
                    timestamp=current_time,
                    suggestions=[
                        "Consider optimizing expensive operations",
                        "Check for infinite loops or blocking calls",
                        "Reduce concurrent operations"
                    ]
                )
                self._add_bottleneck(bottleneck)
            
            # Memory bottleneck
            if system_metrics['memory_mb'] > self.bottleneck_thresholds['memory_mb']:
                bottleneck = BottleneckInfo(
                    type='memory',
                    severity='high' if system_metrics['memory_mb'] > 1000 else 'medium',
                    description=f"High memory usage detected: {system_metrics['memory_mb']:.1f}MB",
                    metric_name='memory_mb',
                    current_value=system_metrics['memory_mb'],
                    threshold=self.bottleneck_thresholds['memory_mb'],
                    timestamp=current_time,
                    suggestions=[
                        "Check for memory leaks",
                        "Implement better caching strategies",
                        "Reduce memory-intensive operations"
                    ]
                )
                self._add_bottleneck(bottleneck)
        
        # Check function performance
        with self.profile_lock:
            for profile in self.function_profiles.values():
                # Slow function bottleneck
                if (profile.avg_time > self.bottleneck_thresholds['function_avg_time'] and 
                    profile.call_count > 10):  # Only consider functions with sufficient calls
                    bottleneck = BottleneckInfo(
                        type='function',
                        severity='high' if profile.avg_time > 5.0 else 'medium',
                        description=f"Slow function detected: {profile.function_name} averaging {profile.avg_time:.3f}s",
                        metric_name='function_avg_time',
                        current_value=profile.avg_time,
                        threshold=self.bottleneck_thresholds['function_avg_time'],
                        timestamp=current_time,
                        suggestions=[
                            "Optimize algorithm or data structures",
                            "Add caching for expensive operations",
                            "Consider async implementation",
                            "Profile with more detailed tools"
                        ]
                    )
                    self._add_bottleneck(bottleneck)
                
                # High call count bottleneck
                if profile.call_count > self.bottleneck_thresholds['function_call_count']:
                    bottleneck = BottleneckInfo(
                        type='function',
                        severity='medium',
                        description=f"High call count function: {profile.function_name} called {profile.call_count} times",
                        metric_name='function_call_count',
                        current_value=profile.call_count,
                        threshold=self.bottleneck_thresholds['function_call_count'],
                        timestamp=current_time,
                        suggestions=[
                            "Consider batching operations",
                            "Implement rate limiting",
                            "Check for unnecessary repeated calls"
                        ]
                    )
                    self._add_bottleneck(bottleneck)
                
                # High error rate
                if profile.call_count > 100:
                    error_rate = profile.error_count / profile.call_count
                    if error_rate > self.bottleneck_thresholds['error_rate']:
                        bottleneck = BottleneckInfo(
                            type='function',
                            severity='high' if error_rate > 0.5 else 'medium',
                            description=f"High error rate in {profile.function_name}: {error_rate:.1%}",
                            metric_name='error_rate',
                            current_value=error_rate,
                            threshold=self.bottleneck_thresholds['error_rate'],
                            timestamp=current_time,
                            suggestions=[
                                "Review error handling logic",
                                "Check input validation",
                                "Improve error recovery mechanisms"
                            ]
                        )
                        self._add_bottleneck(bottleneck)
    
    def _add_bottleneck(self, bottleneck: BottleneckInfo):
        """Add a bottleneck and trigger alerts."""
        # Avoid duplicate bottlenecks of the same type in a short time window
        recent_similar = [
            b for b in self.bottlenecks
            if (b.type == bottleneck.type and 
                b.metric_name == bottleneck.metric_name and
                time.time() - b.timestamp < 60)  # Within last minute
        ]
        
        if not recent_similar:
            self.bottlenecks.append(bottleneck)
            
            # Trigger alerts
            for callback in self.alert_callbacks:
                try:
                    callback(bottleneck)
                except Exception as e:
                    logging.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback: Callable[[BottleneckInfo], None]):
        """Add a callback for bottleneck alerts."""
        self.alert_callbacks.append(callback)
    
    def get_function_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get all function profiles."""
        with self.profile_lock:
            return {name: profile.to_dict() for name, profile in self.function_profiles.items()}
    
    def get_bottlenecks(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get detected bottlenecks.
        
        Args:
            severity: Filter by severity level
            
        Returns:
            List of bottleneck dictionaries
        """
        bottlenecks = list(self.bottlenecks)
        
        if severity:
            bottlenecks = [b for b in bottlenecks if b.severity == severity]
        
        # Sort by timestamp (most recent first)
        bottlenecks.sort(key=lambda x: x.timestamp, reverse=True)
        
        return [b.to_dict() for b in bottlenecks]
    
    def get_metrics_summary(self, duration: float = 300.0) -> Dict[str, Any]:
        """
        Get a summary of performance metrics.
        
        Args:
            duration: Time window in seconds
            
        Returns:
            Metrics summary dictionary
        """
        cutoff_time = time.time() - duration
        
        with self.metrics_lock:
            recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
        
        # Group metrics by name
        metric_groups = defaultdict(list)
        for metric in recent_metrics:
            metric_groups[metric.name].append(metric.value)
        
        # Calculate statistics
        summary = {}
        for name, values in metric_groups.items():
            if values:
                summary[name] = {
                    'count': len(values),
                    'avg': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'latest': values[-1] if values else 0
                }
        
        # Add system metrics if available
        if self.system_monitor:
            summary.update(self.system_monitor.get_average_metrics(duration))
        
        return summary
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        current_time = time.time()
        
        report = {
            'timestamp': current_time,
            'uptime': current_time - (self.system_monitor.cpu_history[0][0] if self.system_monitor and self.system_monitor.cpu_history else current_time),
            'system_metrics': self.system_monitor.get_current_metrics() if self.system_monitor else {},
            'function_profiles': self.get_function_profiles(),
            'bottlenecks': self.get_bottlenecks(),
            'metrics_summary': self.get_metrics_summary(),
            'total_metrics_recorded': len(self.metrics),
            'total_functions_profiled': len(self.function_profiles)
        }
        
        return report
    
    def clear_profiles(self):
        """Clear all function profiles."""
        with self.profile_lock:
            self.function_profiles.clear()
    
    def clear_bottlenecks(self):
        """Clear all detected bottlenecks."""
        self.bottlenecks.clear()
    
    def set_bottleneck_threshold(self, metric_name: str, threshold: float):
        """Set a custom bottleneck threshold."""
        self.bottleneck_thresholds[metric_name] = threshold
    
    def optimize_suggestions(self) -> List[str]:
        """Get optimization suggestions based on current performance."""
        suggestions = []
        bottlenecks = self.get_bottlenecks()
        
        # Group bottlenecks by type
        by_type = defaultdict(list)
        for bottleneck in bottlenecks:
            by_type[bottleneck['type']].append(bottleneck)
        
        # Generate suggestions based on bottleneck patterns
        if 'cpu' in by_type and len(by_type['cpu']) > 2:
            suggestions.append("Consider implementing CPU-intensive operations in separate threads or processes")
        
        if 'memory' in by_type:
            suggestions.append("Implement memory pooling or reduce memory allocations")
        
        if 'function' in by_type:
            slow_functions = [b for b in by_type['function'] if 'avg_time' in b['metric_name']]
            if len(slow_functions) > 3:
                suggestions.append("Multiple functions are running slow - consider system-wide optimization")
        
        # Check function call patterns
        with self.profile_lock:
            if self.function_profiles:
                total_calls = sum(p.call_count for p in self.function_profiles.values())
                if total_calls > 10000:
                    suggestions.append("High function call volume - consider implementing function call caching")
        
        return suggestions
    
    def export_metrics(self, filename: str, format: str = 'json'):
        """
        Export performance metrics to file.
        
        Args:
            filename: Output filename
            format: Export format ('json' or 'csv')
        """
        if format.lower() == 'json':
            import json
            report = self.get_performance_report()
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        elif format.lower() == 'csv':
            import csv
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'name', 'value', 'unit', 'tags'])
                
                with self.metrics_lock:
                    for metric in self.metrics:
                        writer.writerow([
                            metric.timestamp,
                            metric.name,
                            metric.value,
                            metric.unit,
                            json.dumps(metric.tags)
                        ])
    
    def shutdown(self):
        """Shutdown the profiler and cleanup resources."""
        self.stop_analysis.set()
        
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=2)
        
        if self.system_monitor:
            self.system_monitor.stop_monitoring()


# Global profiler instance
_global_profiler = PerformanceProfiler()


# Convenience decorators
def profile_function(name: Optional[str] = None):
    """Convenience decorator for function profiling."""
    return _global_profiler.profile_function(name)


def record_performance_metric(name: str, value: float, unit: str, 
                            tags: Dict[str, str] = None):
    """Record a performance metric using the global profiler."""
    _global_profiler.record_metric(name, value, unit, tags)
