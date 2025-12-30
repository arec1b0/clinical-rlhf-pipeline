"""
Memory Pressure Monitoring Module

Production-grade memory monitoring with:
- GPU/CPU memory tracking
- Proactive OOM prevention
- Automatic garbage collection
- Memory leak detection
- Alert callbacks

Author: Dani (MLOps Lead)
"""

import gc
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple
from enum import Enum

import torch

logger = logging.getLogger(__name__)


class MemoryPressureLevel(Enum):
    """Memory pressure severity levels."""
    NORMAL = "normal"           # < 70% usage
    WARNING = "warning"         # 70-85% usage
    CRITICAL = "critical"       # 85-95% usage
    EMERGENCY = "emergency"     # > 95% usage


@dataclass
class MemorySnapshot:
    """Point-in-time memory state."""
    timestamp: datetime
    
    # GPU memory (bytes)
    gpu_allocated: int = 0
    gpu_reserved: int = 0
    gpu_total: int = 0
    gpu_free: int = 0
    
    # CPU memory (bytes)
    cpu_used: int = 0
    cpu_total: int = 0
    cpu_available: int = 0
    
    # Derived metrics
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    pressure_level: MemoryPressureLevel = MemoryPressureLevel.NORMAL
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "gpu_allocated_gb": self.gpu_allocated / (1024**3),
            "gpu_reserved_gb": self.gpu_reserved / (1024**3),
            "gpu_total_gb": self.gpu_total / (1024**3),
            "gpu_utilization": self.gpu_utilization,
            "cpu_used_gb": self.cpu_used / (1024**3),
            "cpu_total_gb": self.cpu_total / (1024**3),
            "cpu_utilization": self.cpu_utilization,
            "pressure_level": self.pressure_level.value,
        }


@dataclass
class MemoryConfig:
    """Memory monitoring configuration."""
    
    # Thresholds (as fraction of total)
    warning_threshold: float = 0.70
    critical_threshold: float = 0.85
    emergency_threshold: float = 0.95
    
    # Monitoring settings
    check_interval_seconds: float = 5.0
    history_size: int = 100
    
    # Automatic actions
    auto_gc_on_warning: bool = True
    auto_clear_cache_on_critical: bool = True
    raise_on_emergency: bool = True
    
    # Leak detection
    leak_detection_enabled: bool = True
    leak_growth_threshold: float = 0.1  # 10% growth over window
    leak_window_size: int = 20  # Number of samples


class MemoryLeakDetector:
    """Detects potential memory leaks using trend analysis."""
    
    def __init__(self, window_size: int = 20, growth_threshold: float = 0.1):
        self.window_size = window_size
        self.growth_threshold = growth_threshold
        self.samples: List[int] = []
    
    def add_sample(self, memory_bytes: int) -> Optional[str]:
        """
        Add memory sample and check for leak.
        
        Returns warning message if leak detected, None otherwise.
        """
        self.samples.append(memory_bytes)
        
        if len(self.samples) > self.window_size:
            self.samples.pop(0)
        
        if len(self.samples) < self.window_size:
            return None
        
        # Check for consistent growth
        start_avg = sum(self.samples[:5]) / 5
        end_avg = sum(self.samples[-5:]) / 5
        
        if start_avg > 0:
            growth_rate = (end_avg - start_avg) / start_avg
            
            if growth_rate > self.growth_threshold:
                return (
                    f"Potential memory leak detected: "
                    f"{growth_rate*100:.1f}% growth over {self.window_size} samples "
                    f"({start_avg/(1024**3):.2f}GB → {end_avg/(1024**3):.2f}GB)"
                )
        
        return None
    
    def reset(self):
        """Reset leak detection state."""
        self.samples.clear()


class MemoryMonitor:
    """
    Production memory monitoring with proactive OOM prevention.
    
    Features:
    - Real-time GPU/CPU memory tracking
    - Configurable pressure thresholds
    - Automatic garbage collection
    - Memory leak detection
    - Alert callbacks
    
    Usage:
        monitor = MemoryMonitor(config=MemoryConfig())
        monitor.add_alert_callback(lambda snap: print(f"Alert: {snap.pressure_level}"))
        monitor.start()
        
        # During training
        monitor.check()  # Manual check
        
        monitor.stop()
    """
    
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        device: str = "auto",
    ):
        self.config = config or MemoryConfig()
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.has_gpu = self.device == "cuda" and torch.cuda.is_available()
        
        # State
        self.history: List[MemorySnapshot] = []
        self.alert_callbacks: List[Callable[[MemorySnapshot], None]] = []
        self.current_pressure = MemoryPressureLevel.NORMAL
        
        # Leak detection
        self.gpu_leak_detector = MemoryLeakDetector(
            window_size=self.config.leak_window_size,
            growth_threshold=self.config.leak_growth_threshold,
        )
        self.cpu_leak_detector = MemoryLeakDetector(
            window_size=self.config.leak_window_size,
            growth_threshold=self.config.leak_growth_threshold,
        )
        
        # Background monitoring thread
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Statistics
        self.stats = {
            "gc_triggered": 0,
            "cache_cleared": 0,
            "warnings_issued": 0,
            "critical_events": 0,
            "emergency_events": 0,
            "leaks_detected": 0,
        }
        
        logger.info(f"MemoryMonitor initialized (device={self.device}, has_gpu={self.has_gpu})")
    
    def get_memory_snapshot(self) -> MemorySnapshot:
        """Capture current memory state."""
        snapshot = MemorySnapshot(timestamp=datetime.now())
        
        # GPU memory
        if self.has_gpu:
            try:
                snapshot.gpu_allocated = torch.cuda.memory_allocated()
                snapshot.gpu_reserved = torch.cuda.memory_reserved()
                props = torch.cuda.get_device_properties(0)
                snapshot.gpu_total = props.total_memory
                snapshot.gpu_free = snapshot.gpu_total - snapshot.gpu_reserved
                
                if snapshot.gpu_total > 0:
                    snapshot.gpu_utilization = snapshot.gpu_reserved / snapshot.gpu_total
            except Exception as e:
                logger.warning(f"Failed to get GPU memory: {e}")
        
        # CPU memory
        try:
            import psutil
            mem = psutil.virtual_memory()
            snapshot.cpu_used = mem.used
            snapshot.cpu_total = mem.total
            snapshot.cpu_available = mem.available
            snapshot.cpu_utilization = mem.percent / 100.0
        except ImportError:
            # psutil not available, use basic method
            pass
        except Exception as e:
            logger.warning(f"Failed to get CPU memory: {e}")
        
        # Determine pressure level (use GPU if available, else CPU)
        utilization = snapshot.gpu_utilization if self.has_gpu else snapshot.cpu_utilization
        
        if utilization >= self.config.emergency_threshold:
            snapshot.pressure_level = MemoryPressureLevel.EMERGENCY
        elif utilization >= self.config.critical_threshold:
            snapshot.pressure_level = MemoryPressureLevel.CRITICAL
        elif utilization >= self.config.warning_threshold:
            snapshot.pressure_level = MemoryPressureLevel.WARNING
        else:
            snapshot.pressure_level = MemoryPressureLevel.NORMAL
        
        return snapshot
    
    def check(self) -> MemorySnapshot:
        """
        Perform memory check and take automatic actions if needed.
        
        Returns current memory snapshot.
        """
        snapshot = self.get_memory_snapshot()
        
        # Add to history
        self.history.append(snapshot)
        if len(self.history) > self.config.history_size:
            self.history.pop(0)
        
        # Check for leaks
        if self.config.leak_detection_enabled:
            self._check_leaks(snapshot)
        
        # Handle pressure levels
        previous_pressure = self.current_pressure
        self.current_pressure = snapshot.pressure_level
        
        if snapshot.pressure_level == MemoryPressureLevel.WARNING:
            self.stats["warnings_issued"] += 1
            
            if self.config.auto_gc_on_warning:
                self._trigger_gc()
            
            if previous_pressure != MemoryPressureLevel.WARNING:
                logger.warning(
                    f"Memory pressure WARNING: "
                    f"GPU={snapshot.gpu_utilization*100:.1f}%, "
                    f"CPU={snapshot.cpu_utilization*100:.1f}%"
                )
                self._notify_callbacks(snapshot)
        
        elif snapshot.pressure_level == MemoryPressureLevel.CRITICAL:
            self.stats["critical_events"] += 1
            
            if self.config.auto_gc_on_warning:
                self._trigger_gc()
            
            if self.config.auto_clear_cache_on_critical:
                self._clear_cuda_cache()
            
            if previous_pressure != MemoryPressureLevel.CRITICAL:
                logger.error(
                    f"Memory pressure CRITICAL: "
                    f"GPU={snapshot.gpu_utilization*100:.1f}%, "
                    f"CPU={snapshot.cpu_utilization*100:.1f}%"
                )
                self._notify_callbacks(snapshot)
        
        elif snapshot.pressure_level == MemoryPressureLevel.EMERGENCY:
            self.stats["emergency_events"] += 1
            
            # Aggressive cleanup
            self._trigger_gc()
            self._clear_cuda_cache()
            
            logger.critical(
                f"Memory pressure EMERGENCY: "
                f"GPU={snapshot.gpu_utilization*100:.1f}%, "
                f"CPU={snapshot.cpu_utilization*100:.1f}%"
            )
            self._notify_callbacks(snapshot)
            
            if self.config.raise_on_emergency:
                raise MemoryError(
                    f"Memory emergency: GPU {snapshot.gpu_utilization*100:.1f}% used. "
                    f"Consider reducing batch size or enabling gradient checkpointing."
                )
        
        return snapshot
    
    def _check_leaks(self, snapshot: MemorySnapshot):
        """Check for memory leaks."""
        # GPU leak check
        if self.has_gpu:
            leak_warning = self.gpu_leak_detector.add_sample(snapshot.gpu_allocated)
            if leak_warning:
                self.stats["leaks_detected"] += 1
                logger.warning(f"GPU {leak_warning}")
        
        # CPU leak check
        leak_warning = self.cpu_leak_detector.add_sample(snapshot.cpu_used)
        if leak_warning:
            self.stats["leaks_detected"] += 1
            logger.warning(f"CPU {leak_warning}")
    
    def _trigger_gc(self):
        """Trigger garbage collection."""
        gc.collect()
        self.stats["gc_triggered"] += 1
        logger.debug("Garbage collection triggered")
    
    def _clear_cuda_cache(self):
        """Clear CUDA memory cache."""
        if self.has_gpu:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.stats["cache_cleared"] += 1
            logger.debug("CUDA cache cleared")
    
    def _notify_callbacks(self, snapshot: MemorySnapshot):
        """Notify all registered callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def add_alert_callback(self, callback: Callable[[MemorySnapshot], None]):
        """Register callback for memory alerts."""
        self.alert_callbacks.append(callback)
    
    def start(self):
        """Start background memory monitoring."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            logger.warning("Monitor already running")
            return
        
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="MemoryMonitor",
        )
        self._monitor_thread.start()
        logger.info("Background memory monitoring started")
    
    def stop(self):
        """Stop background memory monitoring."""
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
        logger.info("Background memory monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self.check()
            except MemoryError:
                # Don't crash the monitoring thread on emergency
                pass
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
            
            self._stop_event.wait(self.config.check_interval_seconds)
    
    def get_summary(self) -> Dict:
        """Get monitoring summary."""
        current = self.get_memory_snapshot()
        
        return {
            "current": current.to_dict(),
            "pressure_level": current.pressure_level.value,
            "stats": self.stats.copy(),
            "history_size": len(self.history),
            "config": {
                "warning_threshold": self.config.warning_threshold,
                "critical_threshold": self.config.critical_threshold,
                "emergency_threshold": self.config.emergency_threshold,
            },
        }
    
    def get_recommendation(self) -> str:
        """Get recommendation based on current memory state."""
        snapshot = self.get_memory_snapshot()
        
        if snapshot.pressure_level == MemoryPressureLevel.NORMAL:
            return "Memory usage is healthy. No action needed."
        
        recommendations = []
        
        if snapshot.pressure_level in (MemoryPressureLevel.WARNING, MemoryPressureLevel.CRITICAL):
            recommendations.append("Consider reducing batch_size")
            recommendations.append("Enable gradient_checkpointing: true")
            
            if self.has_gpu and snapshot.gpu_utilization > 0.8:
                recommendations.append("Enable 4-bit quantization: load_in_4bit: true")
                recommendations.append("Consider using share_value_base: true")
        
        if snapshot.pressure_level == MemoryPressureLevel.EMERGENCY:
            recommendations.append("URGENT: Reduce batch_size immediately")
            recommendations.append("Switch to smaller model (e.g., biogpt)")
            recommendations.append("Enable all memory optimizations")
        
        if self.stats["leaks_detected"] > 0:
            recommendations.append("Memory leak detected - check for tensor accumulation in loops")
        
        return "\n".join(f"• {r}" for r in recommendations)


def create_memory_monitor(config: dict) -> MemoryMonitor:
    """Create memory monitor from config dictionary."""
    mem_config = config.get("memory_monitoring", {})
    
    return MemoryMonitor(
        config=MemoryConfig(
            warning_threshold=mem_config.get("warning_threshold", 0.70),
            critical_threshold=mem_config.get("critical_threshold", 0.85),
            emergency_threshold=mem_config.get("emergency_threshold", 0.95),
            check_interval_seconds=mem_config.get("check_interval_seconds", 5.0),
            auto_gc_on_warning=mem_config.get("auto_gc_on_warning", True),
            auto_clear_cache_on_critical=mem_config.get("auto_clear_cache_on_critical", True),
            raise_on_emergency=mem_config.get("raise_on_emergency", True),
            leak_detection_enabled=mem_config.get("leak_detection_enabled", True),
        ),
        device=config.get("model", {}).get("device", "auto"),
    )
