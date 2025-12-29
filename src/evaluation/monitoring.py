"""
Evaluation and Monitoring for Clinical RLHF

Production monitoring with:
- Reward drift detection
- Safety metric tracking
- Performance degradation alerts
- Prometheus metrics export

Author: Dani (MLOps Lead)
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import logging
import json
from pathlib import Path

# Prometheus client (optional)
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of drift that can be detected."""
    REWARD_DRIFT = "reward_drift"
    SAFETY_DRIFT = "safety_drift"
    DISTRIBUTION_SHIFT = "distribution_shift"
    PERFORMANCE_DEGRADATION = "performance_degradation"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    """Alert generated when drift is detected."""
    alert_id: str
    drift_type: DriftType
    severity: AlertSeverity
    metric_name: str
    baseline_value: float
    current_value: float
    threshold: float
    p_value: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "drift_type": self.drift_type.value,
            "severity": self.severity.value,
            "metric_name": self.metric_name,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "p_value": self.p_value,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
        }


class StatisticalTests:
    """Statistical tests for drift detection."""
    
    @staticmethod
    def kolmogorov_smirnov(
        baseline: np.ndarray,
        current: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[float, bool]:
        """
        Two-sample Kolmogorov-Smirnov test.
        
        Returns:
            (statistic, is_significant)
        """
        statistic, p_value = stats.ks_2samp(baseline, current)
        return statistic, p_value < alpha
    
    @staticmethod
    def population_stability_index(
        baseline: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Compute Population Stability Index (PSI).
        
        PSI < 0.1: No significant shift
        0.1 <= PSI < 0.25: Moderate shift
        PSI >= 0.25: Significant shift
        """
        # Compute histograms
        min_val = min(baseline.min(), current.min())
        max_val = max(baseline.max(), current.max())
        
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        baseline_hist, _ = np.histogram(baseline, bins=bin_edges, density=True)
        current_hist, _ = np.histogram(current, bins=bin_edges, density=True)
        
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        baseline_hist = baseline_hist + eps
        current_hist = current_hist + eps
        
        # Normalize
        baseline_hist = baseline_hist / baseline_hist.sum()
        current_hist = current_hist / current_hist.sum()
        
        # PSI
        psi = np.sum(
            (current_hist - baseline_hist) * np.log(current_hist / baseline_hist)
        )
        
        return psi
    
    @staticmethod
    def mann_whitney_u(
        baseline: np.ndarray,
        current: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[float, bool]:
        """
        Mann-Whitney U test for distribution shift.
        
        Non-parametric test suitable for non-normal distributions.
        """
        statistic, p_value = stats.mannwhitneyu(
            baseline, current, alternative='two-sided'
        )
        return statistic, p_value < alpha
    
    @staticmethod
    def cusum(
        values: np.ndarray,
        target: float,
        std: float,
        k: float = 0.5,
        h: float = 4.0
    ) -> Tuple[np.ndarray, List[int]]:
        """
        CUSUM (Cumulative Sum) control chart for detecting mean shift.
        
        Args:
            values: Time series data
            target: Target mean
            std: Standard deviation
            k: Slack parameter (typically 0.5)
            h: Decision interval (typically 4-5)
            
        Returns:
            (cusum_values, change_points)
        """
        n = len(values)
        cusum_pos = np.zeros(n)
        cusum_neg = np.zeros(n)
        change_points = []
        
        for i in range(1, n):
            z = (values[i] - target) / std
            cusum_pos[i] = max(0, cusum_pos[i-1] + z - k)
            cusum_neg[i] = max(0, cusum_neg[i-1] - z - k)
            
            if cusum_pos[i] > h or cusum_neg[i] > h:
                change_points.append(i)
                # Reset after detection
                cusum_pos[i] = 0
                cusum_neg[i] = 0
        
        return cusum_pos + cusum_neg, change_points


@dataclass
class MetricWindow:
    """Sliding window for metric tracking."""
    name: str
    window_size: int
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Baseline statistics (computed once, updated periodically)
    baseline_mean: Optional[float] = None
    baseline_std: Optional[float] = None
    baseline_values: Optional[np.ndarray] = None
    
    def add(self, value: float, timestamp: Optional[datetime] = None):
        """Add a value to the window."""
        self.values.append(value)
        self.timestamps.append(timestamp or datetime.utcnow())
    
    def get_recent(self, n: Optional[int] = None) -> np.ndarray:
        """Get recent values."""
        n = n or len(self.values)
        return np.array(list(self.values)[-n:])
    
    def get_statistics(self) -> Dict[str, float]:
        """Compute current statistics."""
        if not self.values:
            return {}
        
        arr = np.array(self.values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "count": len(arr),
        }
    
    def set_baseline(self, n: Optional[int] = None):
        """Set current values as baseline."""
        values = self.get_recent(n)
        if len(values) > 0:
            self.baseline_values = values
            self.baseline_mean = float(np.mean(values))
            self.baseline_std = float(np.std(values))


class ClinicalRLHFMonitor:
    """
    Production monitoring for Clinical RLHF.
    
    Tracks:
    - Reward distribution and drift
    - Safety metrics
    - Response quality metrics
    - Latency and throughput
    """
    
    def __init__(
        self,
        drift_threshold_psi: float = 0.25,
        safety_alert_threshold: float = 0.1,
        prometheus_port: Optional[int] = None,
        alert_callback: Optional[Callable[[DriftAlert], None]] = None
    ):
        self.drift_threshold_psi = drift_threshold_psi
        self.safety_alert_threshold = safety_alert_threshold
        self.alert_callback = alert_callback
        
        # Metric windows
        self.metrics: Dict[str, MetricWindow] = {
            "total_reward": MetricWindow("total_reward", 1000),
            "safety_score": MetricWindow("safety_score", 1000),
            "uncertainty_score": MetricWindow("uncertainty_score", 1000),
            "guideline_adherence": MetricWindow("guideline_adherence", 1000),
            "response_latency_ms": MetricWindow("response_latency_ms", 1000),
        }
        
        # Alert history
        self.alerts: List[DriftAlert] = []
        
        # Statistical test utilities
        self.stat_tests = StatisticalTests()
        
        # Prometheus setup
        self.prometheus_metrics = {}
        if PROMETHEUS_AVAILABLE and prometheus_port:
            self._setup_prometheus(prometheus_port)
    
    def _setup_prometheus(self, port: int):
        """Setup Prometheus metrics."""
        self.prometheus_metrics = {
            "reward_mean": Gauge(
                "clinical_rlhf_reward_mean",
                "Mean reward over recent samples"
            ),
            "safety_score_mean": Gauge(
                "clinical_rlhf_safety_mean",
                "Mean safety score"
            ),
            "unsafe_rate": Gauge(
                "clinical_rlhf_unsafe_rate",
                "Rate of unsafe responses"
            ),
            "requests_total": Counter(
                "clinical_rlhf_requests_total",
                "Total number of requests processed"
            ),
            "latency": Histogram(
                "clinical_rlhf_latency_seconds",
                "Response latency distribution",
                buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
            ),
            "drift_detected": Counter(
                "clinical_rlhf_drift_detected_total",
                "Number of drift alerts",
                ["drift_type", "severity"]
            ),
        }
        
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")
    
    def record_inference(
        self,
        reward_breakdown: Dict[str, float],
        latency_ms: float,
        safety_score: float,
        is_safe: bool
    ):
        """
        Record metrics from an inference.
        
        Called after each model response.
        """
        now = datetime.utcnow()
        
        # Record metrics
        self.metrics["total_reward"].add(
            reward_breakdown.get("total_reward", 0), now
        )
        self.metrics["safety_score"].add(safety_score, now)
        self.metrics["response_latency_ms"].add(latency_ms, now)
        
        if "uncertainty_quantification" in reward_breakdown:
            self.metrics["uncertainty_score"].add(
                reward_breakdown["uncertainty_quantification"], now
            )
        
        if "guideline_adherence" in reward_breakdown:
            self.metrics["guideline_adherence"].add(
                reward_breakdown["guideline_adherence"], now
            )
        
        # Update Prometheus metrics
        if self.prometheus_metrics:
            self.prometheus_metrics["requests_total"].inc()
            self.prometheus_metrics["latency"].observe(latency_ms / 1000)
            
            # Update gauges periodically
            if len(self.metrics["total_reward"].values) % 100 == 0:
                self._update_prometheus_gauges()
    
    def _update_prometheus_gauges(self):
        """Update Prometheus gauge metrics."""
        if not self.prometheus_metrics:
            return
        
        stats = self.metrics["total_reward"].get_statistics()
        if stats:
            self.prometheus_metrics["reward_mean"].set(stats["mean"])
        
        safety_stats = self.metrics["safety_score"].get_statistics()
        if safety_stats:
            self.prometheus_metrics["safety_score_mean"].set(safety_stats["mean"])
    
    def set_baselines(self, n_samples: int = 500):
        """
        Set baseline statistics from recent data.
        
        Should be called after initial model deployment
        with representative traffic.
        """
        for name, window in self.metrics.items():
            if len(window.values) >= n_samples:
                window.set_baseline(n_samples)
                logger.info(
                    f"Baseline set for {name}: "
                    f"mean={window.baseline_mean:.4f}, "
                    f"std={window.baseline_std:.4f}"
                )
    
    def check_drift(self) -> List[DriftAlert]:
        """
        Check for drift in all metrics.
        
        Returns list of any new alerts.
        """
        new_alerts = []
        
        for name, window in self.metrics.items():
            if window.baseline_values is None:
                continue
            
            current = window.get_recent(100)
            if len(current) < 50:
                continue
            
            # PSI test
            psi = self.stat_tests.population_stability_index(
                window.baseline_values, current
            )
            
            if psi >= self.drift_threshold_psi:
                severity = AlertSeverity.CRITICAL if psi > 0.5 else AlertSeverity.WARNING
                
                alert = DriftAlert(
                    alert_id=f"{name}_{datetime.utcnow().timestamp()}",
                    drift_type=DriftType.DISTRIBUTION_SHIFT,
                    severity=severity,
                    metric_name=name,
                    baseline_value=window.baseline_mean or 0,
                    current_value=float(np.mean(current)),
                    threshold=self.drift_threshold_psi,
                    p_value=psi,
                    message=f"Distribution shift detected in {name}: PSI={psi:.4f}"
                )
                
                new_alerts.append(alert)
                self.alerts.append(alert)
                
                if self.alert_callback:
                    self.alert_callback(alert)
                
                logger.warning(alert.message)
                
                # Prometheus counter
                if self.prometheus_metrics:
                    self.prometheus_metrics["drift_detected"].labels(
                        drift_type="distribution_shift",
                        severity=severity.value
                    ).inc()
        
        # Special check for safety degradation
        safety_alert = self._check_safety_degradation()
        if safety_alert:
            new_alerts.append(safety_alert)
        
        return new_alerts
    
    def _check_safety_degradation(self) -> Optional[DriftAlert]:
        """Check specifically for safety metric degradation."""
        safety_window = self.metrics["safety_score"]
        
        if len(safety_window.values) < 100:
            return None
        
        recent = safety_window.get_recent(100)
        unsafe_rate = (recent < 0.5).mean()
        
        if unsafe_rate > self.safety_alert_threshold:
            alert = DriftAlert(
                alert_id=f"safety_degradation_{datetime.utcnow().timestamp()}",
                drift_type=DriftType.SAFETY_DRIFT,
                severity=AlertSeverity.CRITICAL,
                metric_name="unsafe_rate",
                baseline_value=0.0,
                current_value=unsafe_rate,
                threshold=self.safety_alert_threshold,
                message=f"CRITICAL: High unsafe response rate: {unsafe_rate:.2%}"
            )
            
            self.alerts.append(alert)
            
            if self.alert_callback:
                self.alert_callback(alert)
            
            logger.critical(alert.message)
            return alert
        
        return None
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        dashboard = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {},
            "recent_alerts": [],
        }
        
        for name, window in self.metrics.items():
            stats = window.get_statistics()
            if stats:
                dashboard["metrics"][name] = {
                    **stats,
                    "baseline_mean": window.baseline_mean,
                    "baseline_std": window.baseline_std,
                }
        
        # Recent alerts
        recent_alerts = [
            a for a in self.alerts
            if a.timestamp > datetime.utcnow() - timedelta(hours=24)
        ]
        dashboard["recent_alerts"] = [a.to_dict() for a in recent_alerts[-10:]]
        
        return dashboard
    
    def export_metrics(self, filepath: Path):
        """Export metrics history to file."""
        export_data = {
            "exported_at": datetime.utcnow().isoformat(),
            "metrics": {},
        }
        
        for name, window in self.metrics.items():
            export_data["metrics"][name] = {
                "values": list(window.values),
                "timestamps": [t.isoformat() for t in window.timestamps],
                "statistics": window.get_statistics(),
            }
        
        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")


class ModelComparator:
    """
    Compare two model versions for A/B testing or model updates.
    
    Ensures new model maintains safety while improving other metrics.
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.stat_tests = StatisticalTests()
    
    def compare(
        self,
        baseline_metrics: Dict[str, List[float]],
        candidate_metrics: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """
        Compare baseline and candidate model metrics.
        
        Returns comparison results with statistical tests.
        """
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "comparisons": {},
            "recommendation": None,
            "safety_check": None,
        }
        
        for metric_name in baseline_metrics.keys():
            if metric_name not in candidate_metrics:
                continue
            
            baseline = np.array(baseline_metrics[metric_name])
            candidate = np.array(candidate_metrics[metric_name])
            
            # Basic statistics
            baseline_mean = np.mean(baseline)
            candidate_mean = np.mean(candidate)
            improvement = (candidate_mean - baseline_mean) / (abs(baseline_mean) + 1e-8)
            
            # Statistical tests
            ks_stat, ks_significant = self.stat_tests.kolmogorov_smirnov(
                baseline, candidate, self.significance_level
            )
            
            mw_stat, mw_significant = self.stat_tests.mann_whitney_u(
                baseline, candidate, self.significance_level
            )
            
            psi = self.stat_tests.population_stability_index(baseline, candidate)
            
            results["comparisons"][metric_name] = {
                "baseline_mean": float(baseline_mean),
                "candidate_mean": float(candidate_mean),
                "improvement_pct": float(improvement * 100),
                "ks_statistic": float(ks_stat),
                "ks_significant": ks_significant,
                "mw_significant": mw_significant,
                "psi": float(psi),
            }
        
        # Safety check - candidate must not be worse on safety
        if "safety_score" in results["comparisons"]:
            safety_comp = results["comparisons"]["safety_score"]
            
            if safety_comp["candidate_mean"] < safety_comp["baseline_mean"] * 0.95:
                results["safety_check"] = "FAILED"
                results["recommendation"] = "REJECT - Safety degradation"
            else:
                results["safety_check"] = "PASSED"
        
        # Overall recommendation
        if results["recommendation"] is None:
            total_improvement = sum(
                c["improvement_pct"]
                for c in results["comparisons"].values()
            )
            
            if total_improvement > 5 and results.get("safety_check") == "PASSED":
                results["recommendation"] = "APPROVE - Overall improvement"
            elif total_improvement < -5:
                results["recommendation"] = "REJECT - Overall degradation"
            else:
                results["recommendation"] = "NEUTRAL - No significant change"
        
        return results
