"""
Web dashboard foundation with real-time monitoring for AIFrameworkRPC v0.2.0
"""

import json
import time
import asyncio
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    WEB_FRAMEWORK_AVAILABLE = True
except ImportError:
    WEB_FRAMEWORK_AVAILABLE = False
    logging.warning("FastAPI not available. Install with: pip install fastapi uvicorn")

from .performance_profiler import _global_profiler
from .enhanced_connection import _enhanced_connection_pool
from .predictive_cache import _predictive_cache
from .error_recovery import _global_retry_manager


@dataclass
class DashboardConfig:
    """Configuration for the web dashboard."""
    host: str = "localhost"
    port: int = 8080
    enable_auth: bool = False
    auth_token: str = ""
    enable_cors: bool = True
    static_dir: str = "static"
    update_interval: float = 1.0
    max_websocket_connections: int = 100


class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_lock = threading.Lock()
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        
        with self.connection_lock:
            self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        with self.connection_lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific WebSocket."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logging.error(f"Failed to send WebSocket message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected WebSockets."""
        if not self.active_connections:
            return
        
        message_text = json.dumps(message)
        disconnected = []
        
        with self.connection_lock:
            connections = self.active_connections.copy()
        
        for connection in connections:
            try:
                await connection.send_text(message_text)
            except Exception as e:
                logging.error(f"Failed to broadcast to WebSocket: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        with self.connection_lock:
            return len(self.active_connections)


class DataProvider:
    """Provides data for the dashboard."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_overview_data(self) -> Dict[str, Any]:
        """Get overview data for the dashboard."""
        try:
            # Get performance stats
            perf_stats = _global_profiler.get_performance_report()
            
            # Get connection pool stats
            conn_stats = _enhanced_connection_pool.get_pool_stats()
            
            # Get cache stats
            cache_stats = _predictive_cache.get_stats()
            
            # Get retry stats
            retry_stats = _global_retry_manager.get_retry_stats()
            
            return {
                'timestamp': time.time(),
                'performance': perf_stats,
                'connections': conn_stats,
                'cache': asdict(cache_stats),
                'retry': retry_stats,
                'status': 'healthy'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get overview data: {e}")
            return {
                'timestamp': time.time(),
                'status': 'error',
                'error': str(e)
            }
    
    def get_metrics_data(self, time_range: str = "1h") -> Dict[str, Any]:
        """Get metrics data for a specific time range."""
        try:
            # Convert time range to seconds
            range_seconds = {
                "1h": 3600,
                "6h": 21600,
                "24h": 86400,
                "7d": 604800
            }.get(time_range, 3600)
            
            metrics_summary = _global_profiler.get_metrics_summary(range_seconds)
            
            return {
                'time_range': time_range,
                'metrics': metrics_summary,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics data: {e}")
            return {'error': str(e), 'timestamp': time.time()}
    
    def get_bottlenecks_data(self) -> Dict[str, Any]:
        """Get bottlenecks data."""
        try:
            bottlenecks = _global_profiler.get_bottlenecks()
            suggestions = _global_profiler.optimize_suggestions()
            
            return {
                'bottlenecks': bottlenecks,
                'suggestions': suggestions,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get bottlenecks data: {e}")
            return {'error': str(e), 'timestamp': time.time()}
    
    def get_functions_data(self) -> Dict[str, Any]:
        """Get function profiling data."""
        try:
            profiles = _global_profiler.get_function_profiles()
            
            # Sort by average time (slowest first)
            sorted_profiles = sorted(
                profiles.items(),
                key=lambda x: x[1]['avg_time'],
                reverse=True
            )
            
            return {
                'functions': dict(sorted_profiles),
                'total_functions': len(profiles),
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get functions data: {e}")
            return {'error': str(e), 'timestamp': time.time()}
    
    def get_system_data(self) -> Dict[str, Any]:
        """Get system monitoring data."""
        try:
            # Get system metrics from profiler
            system_metrics = {}
            
            if hasattr(_global_profiler, 'system_monitor') and _global_profiler.system_monitor:
                system_metrics = _global_profiler.system_monitor.get_current_metrics()
                system_metrics.update(_global_profiler.system_monitor.get_average_metrics(300))  # 5 minutes
            
            return {
                'system': system_metrics,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system data: {e}")
            return {'error': str(e), 'timestamp': time.time()}


class WebDashboard:
    """
    Web dashboard for real-time monitoring of AIFrameworkRPC.
    
    Features:
    - Real-time WebSocket updates
    - Performance metrics visualization
    - System monitoring
    - Bottleneck detection and alerts
    - Function profiling
    - Configuration management
    - Dark mode support
    - Mobile responsive design
    """
    
    def __init__(self, config: DashboardConfig = None):
        if not WEB_FRAMEWORK_AVAILABLE:
            raise ImportError("FastAPI and uvicorn are required for the web dashboard")
        
        self.config = config or DashboardConfig()
        self.app = FastAPI(
            title="AIFrameworkRPC Dashboard",
            description="Real-time monitoring dashboard for AIFrameworkRPC",
            version="0.2.0"
        )
        
        # Components
        self.websocket_manager = WebSocketManager()
        self.data_provider = DataProvider()
        
        # Background tasks
        self.update_thread: Optional[threading.Thread] = None
        self.stop_updates = threading.Event()
        
        # Setup routes and middleware
        self._setup_middleware()
        self._setup_routes()
        
        # Create static directory
        self.static_dir = Path(self.config.static_dir)
        self.static_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        if self.config.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Authentication middleware (if enabled)
        if self.config.enable_auth:
            @self.app.middleware("http")
            async def auth_middleware(request, call_next):
                auth_header = request.headers.get("Authorization")
                if not auth_header or auth_header != f"Bearer {self.config.auth_token}":
                    raise HTTPException(status_code=401, detail="Invalid authentication")
                return await call_next(request)
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """Serve the main dashboard page."""
            return self._get_dashboard_html()
        
        @self.app.get("/api/overview")
        async def get_overview():
            """Get overview data."""
            return JSONResponse(self.data_provider.get_overview_data())
        
        @self.app.get("/api/metrics")
        async def get_metrics(time_range: str = "1h"):
            """Get metrics data."""
            return JSONResponse(self.data_provider.get_metrics_data(time_range))
        
        @self.app.get("/api/bottlenecks")
        async def get_bottlenecks():
            """Get bottlenecks data."""
            return JSONResponse(self.data_provider.get_bottlenecks_data())
        
        @self.app.get("/api/functions")
        async def get_functions():
            """Get function profiling data."""
            return JSONResponse(self.data_provider.get_functions_data())
        
        @self.app.get("/api/system")
        async def get_system():
            """Get system monitoring data."""
            return JSONResponse(self.data_provider.get_system_data())
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await self.websocket_manager.connect(websocket)
            
            try:
                while True:
                    # Send real-time updates
                    data = {
                        'type': 'update',
                        'data': self.data_provider.get_overview_data()
                    }
                    await self.websocket_manager.send_personal_message(data, websocket)
                    
                    # Wait for next update
                    await asyncio.sleep(self.config.update_interval)
                    
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket)
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                self.websocket_manager.disconnect(websocket)
        
        @self.app.post("/api/actions/optimize")
        async def optimize_performance():
            """Trigger performance optimization."""
            try:
                _predictive_cache.optimize()
                _enhanced_connection_pool.optimize_pool()
                
                return JSONResponse({
                    'status': 'success',
                    'message': 'Performance optimization triggered'
                })
            except Exception as e:
                return JSONResponse({
                    'status': 'error',
                    'message': str(e)
                }, status_code=500)
        
        @self.app.post("/api/actions/clear-cache")
        async def clear_cache():
            """Clear all caches."""
            try:
                _predictive_cache.clear()
                return JSONResponse({
                    'status': 'success',
                    'message': 'Cache cleared successfully'
                })
            except Exception as e:
                return JSONResponse({
                    'status': 'error',
                    'message': str(e)
                }, status_code=500)
        
        @self.app.post("/api/actions/reset-stats")
        async def reset_stats():
            """Reset all statistics."""
            try:
                _global_profiler.clear_profiles()
                _global_retry_manager.clear_stats()
                return JSONResponse({
                    'status': 'success',
                    'message': 'Statistics reset successfully'
                })
            except Exception as e:
                return JSONResponse({
                    'status': 'error',
                    'message': str(e)
                }, status_code=500)
    
    def _get_dashboard_html(self) -> str:
        """Generate the dashboard HTML."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIFrameworkRPC Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .dark { background-color: #1a202c; color: #e2e8f0; }
        .dark .bg-white { background-color: #2d3748; }
        .dark .text-gray-900 { color: #e2e8f0; }
        .dark .text-gray-600 { color: #cbd5e0; }
        .dark .border-gray-200 { border-color: #4a5568; }
        .metric-card { transition: transform 0.2s; }
        .metric-card:hover { transform: translateY(-2px); }
    </style>
</head>
<body class="bg-gray-100 text-gray-900">
    <div class="container mx-auto px-4 py-8">
        <!-- Header -->
        <div class="flex justify-between items-center mb-8">
            <h1 class="text-3xl font-bold">AIFrameworkRPC Dashboard</h1>
            <div class="flex space-x-4">
                <button onclick="toggleDarkMode()" class="px-4 py-2 bg-gray-800 text-white rounded hover:bg-gray-700">
                    🌙 Dark Mode
                </button>
                <button onclick="refreshData()" class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
                    🔄 Refresh
                </button>
            </div>
        </div>

        <!-- Overview Cards -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div class="metric-card bg-white p-6 rounded-lg shadow-md">
                <h3 class="text-lg font-semibold mb-2">Status</h3>
                <p class="text-3xl font-bold text-green-600" id="status">Healthy</p>
                <p class="text-sm text-gray-600">System Status</p>
            </div>
            <div class="metric-card bg-white p-6 rounded-lg shadow-md">
                <h3 class="text-lg font-semibold mb-2">Connections</h3>
                <p class="text-3xl font-bold text-blue-600" id="connections">0</p>
                <p class="text-sm text-gray-600">Active Connections</p>
            </div>
            <div class="metric-card bg-white p-6 rounded-lg shadow-md">
                <h3 class="text-lg font-semibold mb-2">Cache Hit Rate</h3>
                <p class="text-3xl font-bold text-purple-600" id="cache-hit-rate">0%</p>
                <p class="text-sm text-gray-600">Performance</p>
            </div>
            <div class="metric-card bg-white p-6 rounded-lg shadow-md">
                <h3 class="text-lg font-semibold mb-2">Bottlenecks</h3>
                <p class="text-3xl font-bold text-red-600" id="bottlenecks">0</p>
                <p class="text-sm text-gray-600">Issues Detected</p>
            </div>
        </div>

        <!-- Charts Section -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <div class="bg-white p-6 rounded-lg shadow-md">
                <h3 class="text-lg font-semibold mb-4">Performance Metrics</h3>
                <canvas id="performanceChart"></canvas>
            </div>
            <div class="bg-white p-6 rounded-lg shadow-md">
                <h3 class="text-lg font-semibold mb-4">System Resources</h3>
                <canvas id="systemChart"></canvas>
            </div>
        </div>

        <!-- Bottlenecks Section -->
        <div class="bg-white p-6 rounded-lg shadow-md mb-8">
            <h3 class="text-lg font-semibold mb-4">Bottlenecks & Issues</h3>
            <div id="bottlenecks-list" class="space-y-2">
                <p class="text-gray-600">No bottlenecks detected</p>
            </div>
        </div>

        <!-- Actions Section -->
        <div class="bg-white p-6 rounded-lg shadow-md">
            <h3 class="text-lg font-semibold mb-4">Actions</h3>
            <div class="flex space-x-4">
                <button onclick="optimizePerformance()" class="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700">
                    ⚡ Optimize Performance
                </button>
                <button onclick="clearCache()" class="px-4 py-2 bg-yellow-600 text-white rounded hover:bg-yellow-700">
                    🗑️ Clear Cache
                </button>
                <button onclick="resetStats()" class="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700">
                    📊 Reset Stats
                </button>
            </div>
        </div>
    </div>

    <script>
        let performanceChart, systemChart;
        let darkMode = false;

        // Initialize charts
        function initCharts() {
            const performanceCtx = document.getElementById('performanceChart').getContext('2d');
            performanceChart = new Chart(performanceCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Response Time (ms)',
                        data: [],
                        borderColor: 'rgb(59, 130, 246)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: { beginAtZero: true }
                    }
                }
            });

            const systemCtx = document.getElementById('systemChart').getContext('2d');
            systemChart = new Chart(systemCtx, {
                type: 'doughnut',
                data: {
                    labels: ['CPU', 'Memory', 'Available'],
                    datasets: [{
                        data: [0, 0, 100],
                        backgroundColor: [
                            'rgb(239, 68, 68)',
                            'rgb(245, 158, 11)',
                            'rgb(34, 197, 94)'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
        }

        // Update dashboard data
        async function updateData() {
            try {
                const response = await fetch('/api/overview');
                const data = await response.json();
                
                // Update overview cards
                document.getElementById('status').textContent = data.status || 'Unknown';
                document.getElementById('connections').textContent = data.connections?.total_connections || 0;
                document.getElementById('cache-hit-rate').textContent = 
                    Math.round((data.cache?.hit_rate || 0) * 100) + '%';
                document.getElementById('bottlenecks').textContent = 
                    data.performance?.bottlenecks?.length || 0;

                // Update bottlenecks list
                const bottlenecksList = document.getElementById('bottlenecks-list');
                const bottlenecks = data.performance?.bottlenecks || [];
                
                if (bottlenecks.length > 0) {
                    bottlenecksList.innerHTML = bottlenecks.map(b => `
                        <div class="border-l-4 border-red-500 pl-4 py-2">
                            <p class="font-semibold">${b.description}</p>
                            <p class="text-sm text-gray-600">Type: ${b.type}, Severity: ${b.severity}</p>
                        </div>
                    `).join('');
                } else {
                    bottlenecksList.innerHTML = '<p class="text-gray-600">No bottlenecks detected</p>';
                }

            } catch (error) {
                console.error('Failed to update data:', error);
            }
        }

        // WebSocket connection
        function connectWebSocket() {
            const ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onmessage = function(event) {
                const message = JSON.parse(event.data);
                if (message.type === 'update') {
                    updateData();
                }
            };
            
            ws.onclose = function() {
                // Reconnect after 5 seconds
                setTimeout(connectWebSocket, 5000);
            };
        }

        // Action functions
        async function optimizePerformance() {
            try {
                const response = await fetch('/api/actions/optimize', { method: 'POST' });
                const result = await response.json();
                alert(result.message);
                updateData();
            } catch (error) {
                alert('Failed to optimize performance: ' + error.message);
            }
        }

        async function clearCache() {
            if (confirm('Are you sure you want to clear all caches?')) {
                try {
                    const response = await fetch('/api/actions/clear-cache', { method: 'POST' });
                    const result = await response.json();
                    alert(result.message);
                    updateData();
                } catch (error) {
                    alert('Failed to clear cache: ' + error.message);
                }
            }
        }

        async function resetStats() {
            if (confirm('Are you sure you want to reset all statistics?')) {
                try {
                    const response = await fetch('/api/actions/reset-stats', { method: 'POST' });
                    const result = await response.json();
                    alert(result.message);
                    updateData();
                } catch (error) {
                    alert('Failed to reset stats: ' + error.message);
                }
            }
        }

        function refreshData() {
            updateData();
        }

        function toggleDarkMode() {
            darkMode = !darkMode;
            document.body.classList.toggle('dark');
        }

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            updateData();
            connectWebSocket();
            
            // Update data every 5 seconds
            setInterval(updateData, 5000);
        });
    </script>
</body>
</html>
        """
    
    def start(self):
        """Start the web dashboard server."""
        if not WEB_FRAMEWORK_AVAILABLE:
            raise ImportError("FastAPI and uvicorn are required for the web dashboard")
        
        # Start background updates
        self._start_background_updates()
        
        # Start the server
        self.logger.info(f"Starting dashboard on {self.config.host}:{self.config.port}")
        
        # Run in a separate thread
        def run_server():
            uvicorn.run(
                self.app,
                host=self.config.host,
                port=self.config.port,
                log_level="warning"  # Reduce uvicorn logging
            )
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        return server_thread
    
    def _start_background_updates(self):
        """Start background data updates."""
        def update_loop():
            while not self.stop_updates.is_set():
                try:
                    # Broadcast updates to all WebSocket connections
                    if self.websocket_manager.get_connection_count() > 0:
                        data = {
                            'type': 'update',
                            'data': self.data_provider.get_overview_data()
                        }
                        asyncio.run(self.websocket_manager.broadcast(data))
                    
                    time.sleep(self.config.update_interval)
                    
                except Exception as e:
                    self.logger.error(f"Background update error: {e}")
                    time.sleep(5)
        
        self.update_thread = threading.Thread(
            target=update_loop,
            daemon=True,
            name="DashboardUpdater"
        )
        self.update_thread.start()
    
    def stop(self):
        """Stop the web dashboard."""
        self.stop_updates.set()
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2)
        
        self.logger.info("Dashboard stopped")
    
    def get_url(self) -> str:
        """Get the dashboard URL."""
        return f"http://{self.config.host}:{self.config.port}"
    
    def is_running(self) -> bool:
        """Check if the dashboard is running."""
        return (self.update_thread is not None and 
                self.update_thread.is_alive())


# Global dashboard instance
_global_dashboard: Optional[WebDashboard] = None


def start_dashboard(config: DashboardConfig = None) -> WebDashboard:
    """
    Start the global web dashboard.
    
    Args:
        config: Dashboard configuration
        
    Returns:
        WebDashboard instance
    """
    global _global_dashboard
    
    if _global_dashboard is not None:
        logging.warning("Dashboard is already running")
        return _global_dashboard
    
    _global_dashboard = WebDashboard(config)
    _global_dashboard.start()
    
    return _global_dashboard


def stop_dashboard():
    """Stop the global web dashboard."""
    global _global_dashboard
    
    if _global_dashboard is not None:
        _global_dashboard.stop()
        _global_dashboard = None


def get_dashboard() -> Optional[WebDashboard]:
    """Get the global dashboard instance."""
    return _global_dashboard
