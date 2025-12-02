"""
Predictive caching system with smart pre-loading for AIFrameworkRPC v0.2.0
"""

import time
import threading
import hashlib
import json
import pickle
import statistics
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging
import weakref
import gc


@dataclass
class CacheEntry:
    """A cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: float = 3600.0  # Time to live in seconds
    priority: float = 1.0  # Priority for eviction
    prediction_score: float = 0.0  # Likelihood of future access
    
    def is_expired(self) -> bool:
        """Check if the cache entry is expired."""
        return time.time() - self.created_at > self.ttl
    
    def update_access(self):
        """Update access information."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def get_age(self) -> float:
        """Get age of the entry in seconds."""
        return time.time() - self.created_at
    
    def get_time_since_last_access(self) -> float:
        """Get time since last access in seconds."""
        return time.time() - self.last_accessed


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    predictive_hits: int = 0
    total_requests: int = 0
    avg_access_time: float = 0.0
    memory_usage_bytes: int = 0
    cache_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
    
    @property
    def predictive_hit_rate(self) -> float:
        """Calculate predictive cache hit rate."""
        if self.hits == 0:
            return 0.0
        return self.predictive_hits / self.hits


class UsagePatternAnalyzer:
    """Analyzes usage patterns for predictive caching."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.access_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.pattern_lock = threading.Lock()
        
        # Pattern analysis cache
        self._pattern_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamp = 0
        self._cache_ttl = 300  # 5 minutes
    
    def record_access(self, key: str, context: Dict[str, Any] = None):
        """Record an access pattern."""
        with self.pattern_lock:
            timestamp = time.time()
            self.access_patterns[key].append({
                'timestamp': timestamp,
                'context': context or {}
            })
            
            # Invalidate pattern cache
            if timestamp - self._cache_timestamp > self._cache_ttl:
                self._pattern_cache.clear()
                self._cache_timestamp = timestamp
    
    def predict_next_accesses(self, current_context: Dict[str, Any] = None, 
                            top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Predict which keys are likely to be accessed next.
        
        Args:
            current_context: Current context information
            top_k: Number of top predictions to return
            
        Returns:
            List of (key, probability) tuples
        """
        predictions = []
        current_time = time.time()
        
        with self.pattern_lock:
            for key, accesses in self.access_patterns.items():
                if len(accesses) < 2:
                    continue
                
                # Calculate access frequency and recency
                recent_accesses = [a for a in accesses if current_time - a['timestamp'] < 3600]
                if not recent_accesses:
                    continue
                
                # Time-based prediction
                time_diffs = []
                for i in range(1, len(recent_accesses)):
                    diff = recent_accesses[i]['timestamp'] - recent_accesses[i-1]['timestamp']
                    time_diffs.append(diff)
                
                if time_diffs:
                    avg_interval = statistics.mean(time_diffs)
                    last_access = recent_accesses[-1]['timestamp']
                    next_expected = last_access + avg_interval
                    
                    # Probability based on proximity to expected time
                    time_to_expected = next_expected - current_time
                    if -300 < time_to_expected < 600:  # Within 10 minutes window
                        time_prob = max(0, 1 - abs(time_to_expected) / 600)
                        
                        # Frequency factor
                        freq_prob = min(1, len(recent_accesses) / 10)
                        
                        # Context similarity (if available)
                        context_prob = self._calculate_context_similarity(
                            current_context, recent_accesses[-1]['context']
                        )
                        
                        # Combined probability
                        combined_prob = (time_prob * 0.4 + freq_prob * 0.4 + context_prob * 0.2)
                        predictions.append((key, combined_prob))
        
        # Sort by probability and return top_k
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], 
                                    context2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts."""
        if not context1 or not context2:
            return 0.0
        
        # Simple similarity based on common keys and values
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        matches = sum(1 for key in common_keys if context1.get(key) == context2.get(key))
        return matches / len(common_keys)


class PredictiveCache:
    """
    Advanced predictive caching system with smart pre-loading.
    
    Features:
    - Multi-tier caching (memory + disk)
    - Predictive pre-loading based on usage patterns
    - Intelligent eviction policies
    - Context-aware caching
    - Automatic size management
    - Performance monitoring
    """
    
    def __init__(self, max_memory_size: int = 100 * 1024 * 1024,  # 100MB
                 max_disk_size: int = 500 * 1024 * 1024,  # 500MB
                 cache_dir: str = "cache",
                 enable_prediction: bool = True,
                 prediction_threshold: float = 0.7):
        self.max_memory_size = max_memory_size
        self.max_disk_size = max_disk_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.enable_prediction = enable_prediction
        self.prediction_threshold = prediction_threshold
        
        # Memory cache (LRU with priority)
        self._memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._memory_lock = threading.RLock()
        
        # Disk cache index
        self._disk_index: Dict[str, Dict[str, Any]] = {}
        self._disk_lock = threading.RLock()
        
        # Usage pattern analyzer
        self.pattern_analyzer = UsagePatternAnalyzer()
        
        # Statistics
        self.stats = CacheStats()
        self._stats_lock = threading.Lock()
        
        # Pre-loading system
        self._preload_executor = ThreadPoolExecutor(max_workers=2, 
                                                   thread_name_prefix="cache-preload")
        self._preload_queue: deque = deque(maxlen=50)
        self._preload_thread: Optional[threading.Thread] = None
        self._stop_preloading = threading.Event()
        
        # Cleanup thread
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        
        # Start background threads
        self._start_background_threads()
        
        # Load disk index
        self._load_disk_index()
    
    def get(self, key: str, context: Dict[str, Any] = None) -> Optional[Any]:
        """
        Get a value from cache with predictive analytics.
        
        Args:
            key: Cache key
            context: Current context information
            
        Returns:
            Cached value or None if not found
        """
        start_time = time.time()
        
        with self._stats_lock:
            self.stats.total_requests += 1
        
        # Record access pattern
        self.pattern_analyzer.record_access(key, context)
        
        # Try memory cache first
        value = self._get_from_memory(key)
        if value is not None:
            self._record_hit(True, False, time.time() - start_time)
            return value
        
        # Try disk cache
        value = self._get_from_disk(key)
        if value is not None:
            # Promote to memory cache
            self._put_to_memory(key, value, ttl=3600, priority=1.0)
            self._record_hit(False, False, time.time() - start_time)
            return value
        
        # Cache miss - trigger predictive pre-loading
        self._record_hit(False, True, time.time() - start_time)
        self._trigger_predictive_preload(context)
        
        return None
    
    def put(self, key: str, value: Any, ttl: float = 3600, 
            priority: float = 1.0, context: Dict[str, Any] = None):
        """
        Put a value into cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            priority: Priority for eviction (higher = less likely to evict)
            context: Context information
        """
        # Record access pattern
        self.pattern_analyzer.record_access(key, context)
        
        # Try to put in memory first
        if not self._put_to_memory(key, value, ttl, priority):
            # If memory is full, spill to disk
            self._put_to_disk(key, value, ttl, priority)
    
    def _get_from_memory(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        with self._memory_lock:
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                
                # Check if expired
                if entry.is_expired():
                    del self._memory_cache[key]
                    return None
                
                # Move to end (LRU)
                self._memory_cache.move_to_end(key)
                entry.update_access()
                return entry.value
        
        return None
    
    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        with self._disk_lock:
            if key not in self._disk_index:
                return None
            
            entry_info = self._disk_index[key]
            
            # Check if expired
            if time.time() - entry_info['created_at'] > entry_info['ttl']:
                self._remove_from_disk(key)
                return None
            
            try:
                cache_file = self.cache_dir / f"{entry_info['hash']}.cache"
                if cache_file.exists():
                    with open(cache_file, 'rb') as f:
                        value = pickle.load(f)
                    
                    # Update access info
                    entry_info['last_accessed'] = time.time()
                    entry_info['access_count'] += 1
                    self._save_disk_index()
                    
                    return value
                    
            except Exception as e:
                logging.error(f"Failed to load from disk cache: {e}")
                self._remove_from_disk(key)
        
        return None
    
    def _put_to_memory(self, key: str, value: Any, ttl: float, 
                      priority: float) -> bool:
        """
        Put value into memory cache.
        
        Returns:
            True if successful, False if cache is full
        """
        try:
            # Calculate size
            size = len(pickle.dumps(value))
            
            with self._memory_lock:
                # Check if we need to evict
                while (self._get_memory_usage() + size > self.max_memory_size and 
                       len(self._memory_cache) > 0):
                    if not self._evict_from_memory():
                        break  # Can't evict more
                
                # Still too big?
                if self._get_memory_usage() + size > self.max_memory_size:
                    return False
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    size_bytes=size,
                    ttl=ttl,
                    priority=priority
                )
                
                self._memory_cache[key] = entry
                self._memory_cache.move_to_end(key)
                
                # Update stats
                with self._stats_lock:
                    self.stats.memory_usage_bytes = self._get_memory_usage()
                    self.stats.cache_size = len(self._memory_cache)
                
                return True
                
        except Exception as e:
            logging.error(f"Failed to put to memory cache: {e}")
            return False
    
    def _put_to_disk(self, key: str, value: Any, ttl: float, priority: float):
        """Put value into disk cache."""
        try:
            # Calculate hash for filename
            key_hash = hashlib.md5(key.encode()).hexdigest()
            cache_file = self.cache_dir / f"{key_hash}.cache"
            
            # Save to disk
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            
            # Update index
            with self._disk_lock:
                self._disk_index[key] = {
                    'hash': key_hash,
                    'created_at': time.time(),
                    'last_accessed': time.time(),
                    'access_count': 1,
                    'size_bytes': cache_file.stat().st_size,
                    'ttl': ttl,
                    'priority': priority
                }
                self._save_disk_index()
            
            # Check disk usage and cleanup if needed
            self._cleanup_disk_cache()
            
        except Exception as e:
            logging.error(f"Failed to put to disk cache: {e}")
    
    def _evict_from_memory(self) -> bool:
        """
        Evict least valuable entry from memory cache.
        
        Returns:
            True if evicted something, False if cache is empty
        """
        if not self._memory_cache:
            return False
        
        # Find entry with lowest priority score
        # Priority score = priority * (1 / access_count) * age
        current_time = time.time()
        min_score = float('inf')
        evict_key = None
        
        for key, entry in self._memory_cache.items():
            if entry.is_expired():
                evict_key = key
                break
            
            # Calculate priority score (lower = more likely to evict)
            access_factor = 1 / max(entry.access_count, 1)
            age_factor = (current_time - entry.last_accessed) / 3600  # Hours since last access
            score = entry.priority * access_factor * age_factor
            
            if score < min_score:
                min_score = score
                evict_key = key
        
        if evict_key:
            # Move to disk before evicting if valuable enough
            entry = self._memory_cache[evict_key]
            if entry.priority > 0.5 and not entry.is_expired():
                self._put_to_disk(evict_key, entry.value, entry.ttl, entry.priority)
            
            del self._memory_cache[evict_key]
            
            with self._stats_lock:
                self.stats.evictions += 1
                self.stats.memory_usage_bytes = self._get_memory_usage()
                self.stats.cache_size = len(self._memory_cache)
            
            return True
        
        return False
    
    def _remove_from_disk(self, key: str):
        """Remove entry from disk cache."""
        if key in self._disk_index:
            entry_info = self._disk_index[key]
            cache_file = self.cache_dir / f"{entry_info['hash']}.cache"
            
            try:
                if cache_file.exists():
                    cache_file.unlink()
            except:
                pass
            
            del self._disk_index[key]
            self._save_disk_index()
    
    def _cleanup_disk_cache(self):
        """Clean up disk cache if it exceeds size limit."""
        current_size = sum(info['size_bytes'] for info in self._disk_index.values())
        
        if current_size > self.max_disk_size:
            # Sort by priority and evict least valuable
            sorted_entries = sorted(
                self._disk_index.items(),
                key=lambda x: (x[1]['priority'], x[1]['last_accessed'])
            )
            
            for key, info in sorted_entries:
                if current_size <= self.max_disk_size * 0.8:  # Leave 20% headroom
                    break
                
                self._remove_from_disk(key)
                current_size -= info['size_bytes']
                
                with self._stats_lock:
                    self.stats.evictions += 1
    
    def _trigger_predictive_preload(self, context: Dict[str, Any] = None):
        """Trigger predictive pre-loading based on usage patterns."""
        if not self.enable_prediction:
            return
        
        predictions = self.pattern_analyzer.predict_next_accesses(context, top_k=5)
        
        for key, probability in predictions:
            if probability > self.prediction_threshold:
                self._preload_queue.append((key, probability))
    
    def _preload_worker(self):
        """Background worker for predictive pre-loading."""
        while not self._stop_preloading.is_set():
            try:
                if self._preload_queue:
                    key, probability = self._preload_queue.popleft()
                    
                    # Check if already in memory
                    if key not in self._memory_cache:
                        # Try to load from disk
                        value = self._get_from_disk(key)
                        if value is not None:
                            self._put_to_memory(key, value, ttl=3600, priority=probability)
                            
                            with self._stats_lock:
                                self.stats.predictive_hits += 1
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logging.error(f"Preload worker error: {e}")
                time.sleep(1)
    
    def _cleanup_worker(self):
        """Background worker for cache cleanup."""
        while not self._stop_cleanup.is_set():
            try:
                # Clean up expired entries
                self._cleanup_expired_entries()
                
                # Optimize memory usage
                if self._get_memory_usage() > self.max_memory_size * 0.9:
                    self._evict_from_memory()
                
                time.sleep(60)  # Run cleanup every minute
                
            except Exception as e:
                logging.error(f"Cleanup worker error: {e}")
                time.sleep(30)
    
    def _cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        
        # Clean memory cache
        with self._memory_lock:
            expired_keys = [
                key for key, entry in self._memory_cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self._memory_cache[key]
        
        # Clean disk cache
        with self._disk_lock:
            expired_keys = [
                key for key, info in self._disk_index.items()
                if current_time - info['created_at'] > info['ttl']
            ]
            
            for key in expired_keys:
                self._remove_from_disk(key)
    
    def _start_background_threads(self):
        """Start background threads for pre-loading and cleanup."""
        self._preload_thread = threading.Thread(
            target=self._preload_worker,
            daemon=True,
            name="PredictiveCache-Preloader"
        )
        self._preload_thread.start()
        
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="PredictiveCache-Cleaner"
        )
        self._cleanup_thread.start()
    
    def _get_memory_usage(self) -> int:
        """Calculate current memory usage."""
        return sum(entry.size_bytes for entry in self._memory_cache.values())
    
    def _load_disk_index(self):
        """Load disk cache index from file."""
        index_file = self.cache_dir / "disk_index.json"
        try:
            if index_file.exists():
                with open(index_file, 'r') as f:
                    self._disk_index = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load disk index: {e}")
            self._disk_index = {}
    
    def _save_disk_index(self):
        """Save disk cache index to file."""
        index_file = self.cache_dir / "disk_index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self._disk_index, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save disk index: {e}")
    
    def _record_hit(self, memory_hit: bool, predictive_hit: bool, access_time: float):
        """Record cache hit statistics."""
        with self._stats_lock:
            if memory_hit or predictive_hit:
                self.stats.hits += 1
                if predictive_hit:
                    self.stats.predictive_hits += 1
            else:
                self.stats.misses += 1
            
            # Update average access time
            total_hits = self.stats.hits
            if total_hits > 0:
                self.stats.avg_access_time = (
                    (self.stats.avg_access_time * (total_hits - 1) + access_time) / total_hits
                )
    
    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        with self._stats_lock:
            stats = CacheStats(
                hits=self.stats.hits,
                misses=self.stats.misses,
                evictions=self.stats.evictions,
                predictive_hits=self.stats.predictive_hits,
                total_requests=self.stats.total_requests,
                avg_access_time=self.stats.avg_access_time,
                memory_usage_bytes=self._get_memory_usage(),
                cache_size=len(self._memory_cache)
            )
            return stats
    
    def clear(self):
        """Clear all cache entries."""
        with self._memory_lock:
            self._memory_cache.clear()
        
        with self._disk_lock:
            for key in list(self._disk_index.keys()):
                self._remove_from_disk(key)
        
        with self._stats_lock:
            self.stats = CacheStats()
    
    def optimize(self, target_memory_usage: float = 0.8, target_hit_rate: float = 0.8):
        """
        Optimize cache performance.
        
        Args:
            target_memory_usage: Target memory usage as fraction of max (0.0-1.0)
            target_hit_rate: Target hit rate (0.0-1.0)
        """
        current_hit_rate = self.stats.hit_rate
        
        if current_hit_rate < target_hit_rate:
            # Increase cache size if hit rate is low
            new_memory_size = int(self.max_memory_size * 1.2)
            if new_memory_size <= 200 * 1024 * 1024:  # Max 200MB
                self.max_memory_size = new_memory_size
                logging.info(f"Increased memory cache size to {new_memory_size} bytes")
        
        elif current_hit_rate > target_hit_rate * 1.1:
            # Decrease cache size if hit rate is very high
            new_memory_size = int(self.max_memory_size * 0.9)
            if new_memory_size >= 50 * 1024 * 1024:  # Min 50MB
                self.max_memory_size = new_memory_size
                logging.info(f"Decreased memory cache size to {new_memory_size} bytes")
        
        # Evict to target memory usage
        target_bytes = int(self.max_memory_size * target_memory_usage)
        while self._get_memory_usage() > target_bytes and self._memory_cache:
            self._evict_from_memory()
    
    def shutdown(self):
        """Shutdown cache and cleanup resources."""
        self._stop_preloading.set()
        self._stop_cleanup.set()
        
        if self._preload_thread and self._preload_thread.is_alive():
            self._preload_thread.join(timeout=5)
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        
        self._preload_executor.shutdown(wait=True)
        
        # Save final disk index
        self._save_disk_index()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()


# Global predictive cache instance
_predictive_cache = PredictiveCache()
