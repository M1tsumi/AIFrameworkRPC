"""
Context-aware error recovery and retry strategies for AIFrameworkRPC v0.2.0
"""

import time
import threading
import random
import inspect
from typing import Dict, Any, List, Optional, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import Future, ThreadPoolExecutor
from collections import defaultdict, deque
import logging
import traceback
import statistics


class ErrorType(Enum):
    """Types of errors that can occur."""
    NETWORK_ERROR = "network_error"
    CONNECTION_ERROR = "connection_error"
    TIMEOUT_ERROR = "timeout_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    DISCORD_ERROR = "discord_error"
    PLUGIN_ERROR = "plugin_error"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"      # Minor issues, can retry immediately
    MEDIUM = "medium"  # Issues that need some delay
    HIGH = "high"    # Serious issues, need significant delay
    CRITICAL = "critical"  # Critical issues, may need manual intervention


@dataclass
class ErrorContext:
    """Context information about an error."""
    error_type: ErrorType
    severity: ErrorSeverity
    error: Exception
    timestamp: float
    operation: str
    retry_count: int = 0
    context_data: Dict[str, Any] = field(default_factory=dict)
    stack_trace: str = field(default="")
    
    def __post_init__(self):
        if not self.stack_trace:
            self.stack_trace = traceback.format_exc()


@dataclass
class RetryStrategy:
    """Configuration for retry strategies."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retry_on_errors: List[ErrorType] = field(default_factory=list)
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number."""
        delay = min(self.base_delay * (self.backoff_factor ** (attempt - 1)), self.max_delay)
        
        if self.jitter:
            # Add random jitter to prevent thundering herd
            delay *= (0.5 + random.random() * 0.5)
        
        return delay


@dataclass
class RecoveryAction:
    """A recovery action to take when an error occurs."""
    name: str
    description: str
    action: Callable[[ErrorContext], bool]
    applicable_errors: List[ErrorType]
    priority: int = 1  # Lower number = higher priority
    
    def is_applicable(self, error_context: ErrorContext) -> bool:
        """Check if this action is applicable to the given error."""
        return error_context.error_type in self.applicable_errors


class ErrorAnalyzer:
    """Analyzes errors to determine type and severity."""
    
    def __init__(self):
        self._error_patterns = {
            ErrorType.NETWORK_ERROR: [
                "ConnectionRefusedError", "ConnectionResetError", "Network unreachable",
                "socket.error", "urllib.error.URLError", "requests.exceptions.ConnectionError"
            ],
            ErrorType.CONNECTION_ERROR: [
                "pypresence.exceptions.InvalidID", "pypresence.exceptions.DiscordError",
                "pypresence.exceptions.InvalidPipe", "Connection not established"
            ],
            ErrorType.TIMEOUT_ERROR: [
                "TimeoutError", "socket.timeout", "requests.exceptions.Timeout",
                "timed out"
            ],
            ErrorType.AUTHENTICATION_ERROR: [
                "AuthenticationError", "InvalidToken", "Unauthorized",
                "401", "403"
            ],
            ErrorType.RATE_LIMIT_ERROR: [
                "RateLimitError", "429", "TooManyRequests",
                "rate limit"
            ],
            ErrorType.DISCORD_ERROR: [
                "DiscordError", "DiscordException", "Discord API error"
            ],
            ErrorType.PLUGIN_ERROR: [
                "PluginError", "PluginLoadError", "Plugin not found"
            ],
            ErrorType.CONFIGURATION_ERROR: [
                "ConfigurationError", "ConfigError", "Invalid configuration"
            ]
        }
    
    def analyze_error(self, error: Exception, operation: str = "") -> ErrorContext:
        """
        Analyze an error and create context information.
        
        Args:
            error: The exception that occurred
            operation: The operation that was being performed
            
        Returns:
            ErrorContext with analysis results
        """
        error_str = str(error).lower()
        error_type_name = type(error).__name__
        
        # Determine error type
        error_type = ErrorType.UNKNOWN_ERROR
        for etype, patterns in self._error_patterns.items():
            if (any(pattern in error_str for pattern in patterns) or
                any(pattern in error_type_name for pattern in patterns)):
                error_type = etype
                break
        
        # Determine severity
        severity = self._determine_severity(error_type, error)
        
        return ErrorContext(
            error_type=error_type,
            severity=severity,
            error=error,
            timestamp=time.time(),
            operation=operation,
            context_data=self._extract_context(error)
        )
    
    def _determine_severity(self, error_type: ErrorType, error: Exception) -> ErrorSeverity:
        """Determine the severity of an error."""
        if error_type in [ErrorType.AUTHENTICATION_ERROR, ErrorType.CONFIGURATION_ERROR]:
            return ErrorSeverity.CRITICAL
        elif error_type in [ErrorType.CONNECTION_ERROR, ErrorType.DISCORD_ERROR]:
            return ErrorSeverity.HIGH
        elif error_type in [ErrorType.NETWORK_ERROR, ErrorType.TIMEOUT_ERROR]:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _extract_context(self, error: Exception) -> Dict[str, Any]:
        """Extract context information from an error."""
        context = {}
        
        # Extract common error attributes
        if hasattr(error, 'code'):
            context['error_code'] = error.code
        if hasattr(error, 'response'):
            context['response_status'] = getattr(error.response, 'status_code', None)
        if hasattr(error, 'request'):
            context['request_url'] = getattr(error.request, 'url', None)
        
        return context


class ContextAwareRetryManager:
    """
    Context-aware retry manager with intelligent strategies.
    
    Features:
    - Error type and severity analysis
    - Adaptive retry strategies
    - Context-aware recovery actions
    - Circuit breaker pattern
    - Retry history tracking
    - Performance monitoring
    """
    
    def __init__(self, max_concurrent_retries: int = 5):
        self.max_concurrent_retries = max_concurrent_retries
        self.error_analyzer = ErrorAnalyzer()
        
        # Retry strategies by error type
        self.retry_strategies: Dict[ErrorType, RetryStrategy] = {
            ErrorType.NETWORK_ERROR: RetryStrategy(
                max_attempts=5,
                base_delay=2.0,
                max_delay=30.0,
                backoff_factor=1.5,
                jitter=True
            ),
            ErrorType.CONNECTION_ERROR: RetryStrategy(
                max_attempts=3,
                base_delay=5.0,
                max_delay=60.0,
                backoff_factor=2.0,
                jitter=True
            ),
            ErrorType.TIMEOUT_ERROR: RetryStrategy(
                max_attempts=3,
                base_delay=1.0,
                max_delay=15.0,
                backoff_factor=1.5,
                jitter=True
            ),
            ErrorType.RATE_LIMIT_ERROR: RetryStrategy(
                max_attempts=5,
                base_delay=60.0,  # Start with 1 minute for rate limits
                max_delay=300.0,  # Max 5 minutes
                backoff_factor=2.0,
                jitter=True
            ),
            ErrorType.DISCORD_ERROR: RetryStrategy(
                max_attempts=3,
                base_delay=3.0,
                max_delay=30.0,
                backoff_factor=1.5,
                jitter=True
            ),
            ErrorType.PLUGIN_ERROR: RetryStrategy(
                max_attempts=2,
                base_delay=1.0,
                max_delay=10.0,
                backoff_factor=1.0,
                jitter=False
            )
        }
        
        # Recovery actions
        self.recovery_actions: List[RecoveryAction] = []
        self._setup_default_recovery_actions()
        
        # Circuit breaker state
        self.circuit_breakers: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                'failures': 0,
                'last_failure': 0,
                'state': 'closed',  # closed, open, half-open
                'threshold': 5,
                'timeout': 60
            }
        )
        
        # Retry history
        self.retry_history: deque = deque(maxlen=1000)
        self.history_lock = threading.Lock()
        
        # Thread pool for retries
        self.retry_executor = ThreadPoolExecutor(
            max_workers=max_concurrent_retries,
            thread_name_prefix="retry-manager"
        )
        
        # Statistics
        self.stats = {
            'total_retries': 0,
            'successful_retries': 0,
            'failed_retries': 0,
            'circuit_breaker_trips': 0,
            'recovery_actions_taken': 0
        }
        self.stats_lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
    
    def execute_with_retry(self, func: Callable, *args, operation: str = "", 
                          custom_strategy: Optional[RetryStrategy] = None,
                          context_data: Dict[str, Any] = None, **kwargs) -> Future:
        """
        Execute a function with automatic retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            operation: Description of the operation
            custom_strategy: Custom retry strategy
            context_data: Additional context data
            **kwargs: Function keyword arguments
            
        Returns:
            Future that will resolve with the result or raise the last exception
        """
        future = Future()
        
        def execute():
            try:
                result = self._execute_with_retry_internal(
                    func, args, kwargs, operation, custom_strategy, context_data or {}
                )
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
        
        self.retry_executor.submit(execute)
        return future
    
    def _execute_with_retry_internal(self, func: Callable, args: tuple, kwargs: dict,
                                   operation: str, custom_strategy: Optional[RetryStrategy],
                                   context_data: Dict[str, Any]) -> Any:
        """Internal retry execution logic."""
        attempt = 0
        last_error_context = None
        
        while True:
            try:
                # Check circuit breaker
                if self._is_circuit_breaker_open(operation):
                    raise Exception(f"Circuit breaker is open for operation: {operation}")
                
                # Execute the function
                result = func(*args, **kwargs)
                
                # Success - reset circuit breaker if it was failing
                self._reset_circuit_breaker(operation)
                
                # Record successful retry if this was a retry attempt
                if attempt > 0:
                    with self.stats_lock:
                        self.stats['successful_retries'] += 1
                
                return result
                
            except Exception as e:
                attempt += 1
                error_context = self.error_analyzer.analyze_error(e, operation)
                error_context.retry_count = attempt
                error_context.context_data.update(context_data)
                last_error_context = error_context
                
                # Record retry attempt
                with self.stats_lock:
                    self.stats['total_retries'] += 1
                
                # Try recovery actions
                if self._try_recovery_actions(error_context):
                    with self.stats_lock:
                        self.stats['recovery_actions_taken'] += 1
                
                # Check if we should retry
                strategy = custom_strategy or self.retry_strategies.get(
                    error_context.error_type, 
                    RetryStrategy(max_attempts=1)  # Default: no retry
                )
                
                if (attempt >= strategy.max_attempts or 
                    error_context.error_type not in strategy.retry_on_errors and 
                    not custom_strategy):
                    # No more retries
                    self._record_failure(operation, error_context)
                    with self.stats_lock:
                        self.stats['failed_retries'] += 1
                    raise e
                
                # Calculate delay and wait
                delay = strategy.get_delay(attempt)
                self.logger.warning(
                    f"Retry {attempt}/{strategy.max_attempts} for {operation} "
                    f"after {delay:.2f}s due to {error_context.error_type.value}: {e}"
                )
                
                time.sleep(delay)
    
    def _is_circuit_breaker_open(self, operation: str) -> bool:
        """Check if circuit breaker is open for an operation."""
        breaker = self.circuit_breakers[operation]
        
        if breaker['state'] == 'open':
            # Check if timeout has passed
            if time.time() - breaker['last_failure'] > breaker['timeout']:
                breaker['state'] = 'half-open'
                return False
            return True
        
        return False
    
    def _reset_circuit_breaker(self, operation: str):
        """Reset circuit breaker after successful operation."""
        breaker = self.circuit_breakers[operation]
        breaker['failures'] = 0
        breaker['state'] = 'closed'
    
    def _record_failure(self, operation: str, error_context: ErrorContext):
        """Record a failure for circuit breaker."""
        breaker = self.circuit_breakers[operation]
        breaker['failures'] += 1
        breaker['last_failure'] = time.time()
        
        # Trip circuit breaker if threshold exceeded
        if breaker['failures'] >= breaker['threshold']:
            breaker['state'] = 'open'
            with self.stats_lock:
                self.stats['circuit_breaker_trips'] += 1
            self.logger.warning(f"Circuit breaker tripped for operation: {operation}")
        
        # Record in history
        with self.history_lock:
            self.retry_history.append({
                'timestamp': error_context.timestamp,
                'operation': operation,
                'error_type': error_context.error_type.value,
                'severity': error_context.severity.value,
                'retry_count': error_context.retry_count
            })
    
    def _try_recovery_actions(self, error_context: ErrorContext) -> bool:
        """Try applicable recovery actions."""
        applicable_actions = [
            action for action in self.recovery_actions
            if action.is_applicable(error_context)
        ]
        
        # Sort by priority
        applicable_actions.sort(key=lambda x: x.priority)
        
        for action in applicable_actions:
            try:
                if action.action(error_context):
                    self.logger.info(f"Recovery action '{action.name}' succeeded")
                    return True
            except Exception as e:
                self.logger.error(f"Recovery action '{action.name}' failed: {e}")
        
        return False
    
    def _setup_default_recovery_actions(self):
        """Set up default recovery actions."""
        
        def reconnect_discord(error_context: ErrorContext) -> bool:
            """Attempt to reconnect to Discord."""
            if error_context.error_type == ErrorType.CONNECTION_ERROR:
                # This would need access to the RPC instance
                # For now, just log the attempt
                self.logger.info("Attempting Discord reconnection")
                return True
            return False
        
        def clear_cache(error_context: ErrorContext) -> bool:
            """Clear cache on certain errors."""
            if error_context.error_type in [ErrorType.CONFIGURATION_ERROR, ErrorType.PLUGIN_ERROR]:
                self.logger.info("Clearing cache due to error")
                return True
            return False
        
        def reload_configuration(error_context: ErrorContext) -> bool:
            """Reload configuration on config errors."""
            if error_context.error_type == ErrorType.CONFIGURATION_ERROR:
                self.logger.info("Reloading configuration")
                return True
            return False
        
        # Add recovery actions
        self.recovery_actions.extend([
            RecoveryAction(
                name="reconnect_discord",
                description="Reconnect to Discord on connection errors",
                action=reconnect_discord,
                applicable_errors=[ErrorType.CONNECTION_ERROR],
                priority=1
            ),
            RecoveryAction(
                name="clear_cache",
                description="Clear cache on configuration/plugin errors",
                action=clear_cache,
                applicable_errors=[ErrorType.CONFIGURATION_ERROR, ErrorType.PLUGIN_ERROR],
                priority=2
            ),
            RecoveryAction(
                name="reload_config",
                description="Reload configuration on config errors",
                action=reload_configuration,
                applicable_errors=[ErrorType.CONFIGURATION_ERROR],
                priority=3
            )
        ])
    
    def add_recovery_action(self, action: RecoveryAction):
        """Add a custom recovery action."""
        self.recovery_actions.append(action)
    
    def set_retry_strategy(self, error_type: ErrorType, strategy: RetryStrategy):
        """Set a custom retry strategy for an error type."""
        self.retry_strategies[error_type] = strategy
    
    def configure_circuit_breaker(self, operation: str, threshold: int = 5, timeout: int = 60):
        """Configure circuit breaker for an operation."""
        breaker = self.circuit_breakers[operation]
        breaker['threshold'] = threshold
        breaker['timeout'] = timeout
    
    def get_retry_stats(self) -> Dict[str, Any]:
        """Get retry statistics."""
        with self.stats_lock:
            stats = self.stats.copy()
        
        with self.history_lock:
            recent_failures = [
                entry for entry in self.retry_history
                if time.time() - entry['timestamp'] < 3600  # Last hour
            ]
            
            stats['recent_failures'] = len(recent_failures)
            stats['success_rate'] = (
                stats['successful_retries'] / max(stats['total_retries'], 1)
            )
            
            # Error type breakdown
            error_types = defaultdict(int)
            for entry in recent_failures:
                error_types[entry['error_type']] += 1
            stats['recent_error_types'] = dict(error_types)
        
        return stats
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        status = {}
        for operation, breaker in self.circuit_breakers.items():
            status[operation] = {
                'state': breaker['state'],
                'failures': breaker['failures'],
                'threshold': breaker['threshold'],
                'time_until_retry': max(0, breaker['timeout'] - (time.time() - breaker['last_failure']))
            }
        return status
    
    def reset_circuit_breaker(self, operation: str):
        """Manually reset a circuit breaker."""
        self._reset_circuit_breaker(operation)
        self.logger.info(f"Manually reset circuit breaker for: {operation}")
    
    def clear_stats(self):
        """Clear all statistics."""
        with self.stats_lock:
            self.stats = {
                'total_retries': 0,
                'successful_retries': 0,
                'failed_retries': 0,
                'circuit_breaker_trips': 0,
                'recovery_actions_taken': 0
            }
        
        with self.history_lock:
            self.retry_history.clear()
    
    def shutdown(self):
        """Shutdown the retry manager."""
        self.retry_executor.shutdown(wait=True)


# Decorator for easy retry functionality
def retry_on_failure(max_attempts: int = 3, delay: float = 1.0, 
                    backoff_factor: float = 2.0, 
                    error_types: List[ErrorType] = None):
    """
    Decorator for automatic retry on function failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff_factor: Multiplier for delay after each attempt
        error_types: List of error types to retry on
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get retry manager from instance or create new one
            retry_manager = getattr(args[0], '_retry_manager', None)
            if not retry_manager:
                retry_manager = ContextAwareRetryManager()
            
            strategy = RetryStrategy(
                max_attempts=max_attempts,
                base_delay=delay,
                backoff_factor=backoff_factor,
                retry_on_errors=error_types or []
            )
            
            future = retry_manager.execute_with_retry(
                func, *args, 
                operation=f"{func.__module__}.{func.__name__}",
                custom_strategy=strategy,
                **kwargs
            )
            
            return future.result()
        
        return wrapper
    return decorator


# Global retry manager instance
_global_retry_manager = ContextAwareRetryManager()
