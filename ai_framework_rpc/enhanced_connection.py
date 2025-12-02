"""
Enhanced connection management with intelligent load balancing for AIFrameworkRPC v0.2.0
"""

import time
import threading
import weakref
import heapq
import statistics
from typing import Optional, Dict, Any, List, Tuple
from threading import Thread, Event, Lock
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

try:
    from pypresence import Presence
    PYPRESENCE_AVAILABLE = True
except ImportError:
    PYPRESENCE_AVAILABLE = False
    logging.warning("pypresence not available. Install with: pip install pypresence")


@dataclass
class ConnectionMetrics:
    """Metrics for tracking connection performance."""
    client_id: str
    created_at: float
    last_used: float
    usage_count: int = 0
    avg_response_time: float = 0.0
    response_times: deque = field(default_factory=lambda: deque(maxlen=50))
    error_count: int = 0
    success_count: int = 0
    health_score: float = 100.0
    
    def update_response_time(self, response_time: float):
        """Update response time metrics."""
        self.response_times.append(response_time)
        self.avg_response_time = statistics.mean(self.response_times) if self.response_times else 0.0
        self.last_used = time.time()
        
    def update_health_score(self):
        """Calculate health score based on performance metrics."""
        if self.success_count + self.error_count == 0:
            self.health_score = 100.0
            return
            
        success_rate = self.success_count / (self.success_count + self.error_count)
        response_factor = max(0, 1 - (self.avg_response_time / 2.0))  # 2s is considered slow
        age_factor = max(0, 1 - ((time.time() - self.created_at) / 3600))  # 1 hour age penalty
        
        self.health_score = (success_rate * 0.5 + response_factor * 0.3 + age_factor * 0.2) * 100


@dataclass
class LoadBalancedConnection:
    """A connection with load balancing metadata."""
    connection: Presence
    metrics: ConnectionMetrics
    lock: Lock
    in_use: bool = False
    last_health_check: float = field(default_factory=time.time)


class IntelligentConnectionPool:
    """
    Intelligent connection pool with load balancing and predictive caching.
    
    Features:
    - Smart load balancing based on connection health
    - Predictive connection pre-warming
    - Automatic health monitoring
    - Dynamic pool sizing
    - Connection affinity for better performance
    """
    
    def __init__(self, max_connections: int = 10, min_connections: int = 2,
                 health_check_interval: float = 30.0, predictive_warmup: bool = True):
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.health_check_interval = health_check_interval
        self.predictive_warmup = predictive_warmup
        
        # Connection storage
        self._connections: Dict[str, LoadBalancedConnection] = {}
        self._connection_queue: List[Tuple[float, str]] = []  # (health_score, client_id) min-heap
        self._client_affinity: Dict[str, str] = {}  # preferred client_id for threads
        
        # Load balancing
        self._load_balancer_lock = Lock()
        self._usage_patterns: Dict[str, List[float]] = defaultdict(list)
        self._prediction_model = SimpleUsagePredictor()
        
        # Health monitoring
        self._health_monitor_thread: Optional[Thread] = None
        self._stop_health_monitor = Event()
        
        # Statistics
        self._stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'predictive_warms': 0,
            'health_checks': 0,
            'connections_created': 0,
            'connections_destroyed': 0
        }
        
        # Start health monitoring
        self._start_health_monitor()
        
        # Pre-warm minimum connections
        self._ensure_minimum_connections()
    
    def get_connection(self, client_id: str, thread_id: Optional[str] = None) -> Optional[Presence]:
        """
        Get or create a connection with intelligent load balancing.
        
        Args:
            client_id: Discord client ID
            thread_id: Thread ID for connection affinity
            
        Returns:
            Presence connection or None if failed
        """
        with self._load_balancer_lock:
            self._stats['total_requests'] += 1
            
            # Check for existing connection
            if client_id in self._connections:
                conn = self._connections[client_id]
                if not conn.in_use:
                    conn.in_use = True
                    conn.metrics.usage_count += 1
                    self._stats['cache_hits'] += 1
                    return conn.connection
            
            # Check if we can create a new connection
            if len(self._connections) >= self.max_connections:
                # Remove least healthy connection
                self._remove_least_healthy_connection()
            
            # Create new connection
            connection = self._create_connection(client_id)
            if connection:
                self._stats['connections_created'] += 1
                return connection
        
        return None
    
    def _create_connection(self, client_id: str) -> Optional[Presence]:
        """Create a new connection with metrics tracking."""
        try:
            start_time = time.time()
            presence = Presence(client_id)
            creation_time = time.time() - start_time
            
            metrics = ConnectionMetrics(
                client_id=client_id,
                created_at=time.time(),
                last_used=time.time(),
                avg_response_time=creation_time
            )
            metrics.response_times.append(creation_time)
            
            conn = LoadBalancedConnection(
                connection=presence,
                metrics=metrics,
                lock=Lock(),
                in_use=True
            )
            
            self._connections[client_id] = conn
            heapq.heappush(self._connection_queue, (conn.metrics.health_score, client_id))
            
            # Record usage pattern
            self._record_usage_pattern(client_id)
            
            return presence
            
        except Exception as e:
            logging.error(f"Failed to create connection for {client_id}: {e}")
            return None
    
    def release_connection(self, client_id: str):
        """Release a connection back to the pool."""
        with self._load_balancer_lock:
            if client_id in self._connections:
                self._connections[client_id].in_use = False
                # Update heap with new health score
                self._update_connection_health(client_id)
    
    def _update_connection_health(self, client_id: str):
        """Update connection health and rebalance if needed."""
        if client_id not in self._connections:
            return
            
        conn = self._connections[client_id]
        conn.metrics.update_health_score()
        
        # Update heap
        # Remove old entry and add new one (lazy deletion for simplicity)
        heapq.heappush(self._connection_queue, (conn.metrics.health_score, client_id))
    
    def _remove_least_healthy_connection(self):
        """Remove the least healthy connection from the pool."""
        while self._connection_queue:
            health_score, client_id = heapq.heappop(self._connection_queue)
            if client_id in self._connections:
                conn = self._connections[client_id]
                if not conn.in_use:
                    self._destroy_connection(client_id)
                    break
                # If in use, skip and continue (lazy deletion)
    
    def _destroy_connection(self, client_id: str):
        """Destroy a connection and clean up resources."""
        if client_id in self._connections:
            conn = self._connections[client_id]
            try:
                conn.connection.close()
            except:
                pass
            del self._connections[client_id]
            self._stats['connections_destroyed'] += 1
    
    def _record_usage_pattern(self, client_id: str):
        """Record usage pattern for predictive analysis."""
        current_time = time.time()
        self._usage_patterns[client_id].append(current_time)
        
        # Keep only recent patterns (last hour)
        cutoff = current_time - 3600
        self._usage_patterns[client_id] = [
            t for t in self._usage_patterns[client_id] if t > cutoff
        ]
    
    def _start_health_monitor(self):
        """Start the health monitoring thread."""
        if self._health_monitor_thread and self._health_monitor_thread.is_alive():
            return
            
        self._health_monitor_thread = Thread(
            target=self._health_monitor_loop,
            daemon=True,
            name="ConnectionHealthMonitor"
        )
        self._health_monitor_thread.start()
    
    def _health_monitor_loop(self):
        """Main health monitoring loop."""
        while not self._stop_health_monitor.is_set():
            try:
                self._perform_health_checks()
                self._predictive_warmup()
                self._cleanup_idle_connections()
                time.sleep(self.health_check_interval)
            except Exception as e:
                logging.error(f"Health monitor error: {e}")
                time.sleep(5)  # Quick retry on error
    
    def _perform_health_checks(self):
        """Perform health checks on all connections."""
        with self._load_balancer_lock:
            for client_id, conn in list(self._connections.items()):
                try:
                    # Lightweight health check
                    if hasattr(conn.connection, 'ping'):
                        start_time = time.time()
                        conn.connection.ping()
                        response_time = time.time() - start_time
                        conn.metrics.update_response_time(response_time)
                        conn.metrics.success_count += 1
                    else:
                        # Fallback: try a minimal update
                        conn.connection.update("", "", "")
                        conn.metrics.success_count += 1
                    
                    conn.last_health_check = time.time()
                    self._stats['health_checks'] += 1
                    
                except Exception as e:
                    conn.metrics.error_count += 1
                    logging.warning(f"Health check failed for {client_id}: {e}")
                
                # Update health score
                conn.metrics.update_health_score()
    
    def _predictive_warmup(self):
        """Predictively warm up connections based on usage patterns."""
        if not self.predictive_warmup:
            return
            
        predictions = self._prediction_model.predict_next_usage(self._usage_patterns)
        
        with self._load_balancer_lock:
            for client_id, probability in predictions:
                if probability > 0.7 and client_id not in self._connections:
                    if len(self._connections) < self.max_connections:
                        self._create_connection(client_id)
                        self._stats['predictive_warms'] += 1
                        logging.info(f"Predictively warmed up connection for {client_id}")
    
    def _cleanup_idle_connections(self):
        """Clean up connections that have been idle too long."""
        current_time = time.time()
        idle_timeout = 1800  # 30 minutes
        
        with self._load_balancer_lock:
            # Ensure we don't go below minimum connections
            if len(self._connections) <= self.min_connections:
                return
                
            for client_id, conn in list(self._connections.items()):
                if (current_time - conn.metrics.last_used > idle_timeout and 
                    not conn.in_use and len(self._connections) > self.min_connections):
                    self._destroy_connection(client_id)
                    logging.info(f"Cleaned up idle connection for {client_id}")
    
    def _ensure_minimum_connections(self):
        """Ensure minimum number of connections are available."""
        # This would need to know which client_ids to pre-warm
        # For now, we'll skip this as it requires configuration
        pass
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        with self._load_balancer_lock:
            active_connections = sum(1 for conn in self._connections.values() if conn.in_use)
            
            stats = {
                'total_connections': len(self._connections),
                'active_connections': active_connections,
                'idle_connections': len(self._connections) - active_connections,
                'pool_utilization': active_connections / max(self.max_connections, 1),
                'health_scores': {
                    client_id: conn.metrics.health_score 
                    for client_id, conn in self._connections.items()
                },
                'avg_response_time': statistics.mean([
                    conn.metrics.avg_response_time for conn in self._connections.values()
                ]) if self._connections else 0.0,
                **self._stats
            }
            
            return stats
    
    def optimize_pool(self, target_size: Optional[int] = None):
        """Optimize pool size based on current usage patterns."""
        if target_size is None:
            # Calculate optimal size based on recent usage
            recent_usage = sum(len(patterns) for patterns in self._usage_patterns.values())
            target_size = max(self.min_connections, 
                            min(self.max_connections, recent_usage // 10 + 2))
        
        with self._load_balancer_lock:
            current_size = len(self._connections)
            
            if target_size > current_size:
                # Need more connections (predictive)
                logging.info(f"Expanding pool from {current_size} to {target_size}")
            elif target_size < current_size:
                # Need fewer connections
                excess = current_size - target_size
                logging.info(f"Contracting pool from {current_size} to {target_size}")
                
                # Remove least healthy connections
                for _ in range(excess):
                    self._remove_least_healthy_connection()
    
    def shutdown(self):
        """Shutdown the connection pool and clean up all resources."""
        self._stop_health_monitor.set()
        
        if self._health_monitor_thread and self._health_monitor_thread.is_alive():
            self._health_monitor_thread.join(timeout=5)
        
        with self._load_balancer_lock:
            for client_id in list(self._connections.keys()):
                self._destroy_connection(client_id)
    
    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()


class SimpleUsagePredictor:
    """Simple usage pattern predictor for connection warmup."""
    
    def predict_next_usage(self, usage_patterns: Dict[str, List[float]]) -> List[Tuple[str, float]]:
        """
        Predict which client_ids will be used next with probabilities.
        
        Args:
            usage_patterns: Dictionary of client_id -> list of usage timestamps
            
        Returns:
            List of (client_id, probability) tuples sorted by probability
        """
        predictions = []
        current_time = time.time()
        
        for client_id, timestamps in usage_patterns.items():
            if len(timestamps) < 2:
                continue
            
            # Calculate usage frequency and recency
            time_diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            avg_interval = statistics.mean(time_diffs) if time_diffs else 3600
            
            last_usage = timestamps[-1]
            time_since_last = current_time - last_usage
            
            # Simple probability based on expected next usage
            if avg_interval > 0:
                expected_next = last_usage + avg_interval
                time_to_expected = expected_next - current_time
                
                # Higher probability if we're close to expected usage time
                if -300 < time_to_expected < 300:  # Within 5 minutes of expected
                    probability = max(0, 1 - abs(time_to_expected) / 300)
                    predictions.append((client_id, probability))
        
        # Sort by probability and return top candidates
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:5]  # Top 5 predictions


# Global enhanced connection pool
_enhanced_connection_pool = IntelligentConnectionPool()
