"""
Tests for Monitoring and Drift Detection Module
Tests for production medical AI monitoring and alerting
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.monitoring import (
    ClinicalRLHFMonitor,
    DriftAlert,
    AlertSeverity,
    ModelComparator,
    ComparisonResult,
)


class TestClinicalRLHFMonitor:
    """Test suite for clinical RLHF monitoring."""
    
    def setup_method(self):
        self.monitor = ClinicalRLHFMonitor(
            window_size=100,
            drift_threshold_psi=0.1,
            drift_threshold_ks=0.05,
        )
    
    def test_records_metrics(self):
        """Should record incoming metrics."""
        self.monitor.record(
            reward=0.8,
            safety_score=0.9,
            latency_ms=150,
            is_safe=True
        )
        
        stats = self.monitor.get_current_stats()
        
        assert stats["sample_count"] == 1
        assert stats["mean_reward"] == 0.8
        assert stats["mean_safety"] == 0.9
    
    def test_sliding_window(self):
        """Should maintain sliding window of samples."""
        # Fill window
        for i in range(150):
            self.monitor.record(
                reward=0.5 + (i * 0.001),
                safety_score=0.9,
                latency_ms=100,
                is_safe=True
            )
        
        stats = self.monitor.get_current_stats()
        
        # Should only have last 100 samples
        assert stats["sample_count"] == 100
    
    def test_calculates_psi(self):
        """Should calculate Population Stability Index correctly."""
        # Create baseline distribution
        baseline = np.random.normal(0.7, 0.1, 1000)
        
        # Create current distribution (same)
        current_same = np.random.normal(0.7, 0.1, 100)
        
        # Create current distribution (shifted)
        current_shifted = np.random.normal(0.5, 0.1, 100)
        
        psi_same = self.monitor._calculate_psi(baseline, current_same)
        psi_shifted = self.monitor._calculate_psi(baseline, current_shifted)
        
        # Same distribution should have low PSI
        assert psi_same < 0.1
        
        # Shifted distribution should have high PSI
        assert psi_shifted > 0.1
    
    def test_detects_reward_drift(self):
        """Should detect drift in reward distribution."""
        # Establish baseline with high rewards
        for _ in range(100):
            self.monitor.record(
                reward=np.random.normal(0.8, 0.05),
                safety_score=0.9,
                latency_ms=100,
                is_safe=True
            )
        
        self.monitor.establish_baseline()
        
        # Simulate degradation
        for _ in range(100):
            self.monitor.record(
                reward=np.random.normal(0.4, 0.05),  # Much lower
                safety_score=0.9,
                latency_ms=100,
                is_safe=True
            )
        
        alerts = self.monitor.check_for_drift()
        
        # Should detect reward drift
        reward_alerts = [a for a in alerts if "reward" in a.metric.lower()]
        assert len(reward_alerts) > 0
    
    def test_detects_safety_degradation(self):
        """Should alert on safety score degradation."""
        # Establish baseline with good safety
        for _ in range(100):
            self.monitor.record(
                reward=0.7,
                safety_score=0.95,
                latency_ms=100,
                is_safe=True
            )
        
        self.monitor.establish_baseline()
        
        # Simulate safety degradation
        for _ in range(100):
            self.monitor.record(
                reward=0.7,
                safety_score=0.6,  # Degraded safety
                latency_ms=100,
                is_safe=np.random.random() > 0.3  # 30% unsafe
            )
        
        alerts = self.monitor.check_for_drift()
        
        # Should have safety-related alerts
        safety_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        assert len(safety_alerts) > 0
    
    def test_alert_callback(self):
        """Should invoke callback on alert."""
        callback_invoked = []
        
        def on_alert(alert: DriftAlert):
            callback_invoked.append(alert)
        
        self.monitor.register_alert_callback(on_alert)
        
        # Force an alert condition
        for _ in range(100):
            self.monitor.record(reward=0.9, safety_score=0.95, latency_ms=100, is_safe=True)
        
        self.monitor.establish_baseline()
        
        for _ in range(100):
            self.monitor.record(reward=0.2, safety_score=0.4, latency_ms=500, is_safe=False)
        
        self.monitor.check_for_drift()
        
        assert len(callback_invoked) > 0


class TestDriftAlert:
    """Test suite for DriftAlert dataclass."""
    
    def test_alert_creation(self):
        """Should create valid alert."""
        alert = DriftAlert(
            metric="reward_distribution",
            severity=AlertSeverity.WARNING,
            message="Reward distribution shift detected",
            current_value=0.5,
            baseline_value=0.8,
            statistical_test="PSI",
            p_value=0.03,
            timestamp=datetime.now()
        )
        
        assert alert.metric == "reward_distribution"
        assert alert.severity == AlertSeverity.WARNING
    
    def test_alert_severity_ordering(self):
        """Should order severities correctly."""
        assert AlertSeverity.INFO.value < AlertSeverity.WARNING.value
        assert AlertSeverity.WARNING.value < AlertSeverity.CRITICAL.value


class TestKolmogorovSmirnovTest:
    """Test suite for KS drift detection."""
    
    def setup_method(self):
        self.monitor = ClinicalRLHFMonitor(
            window_size=100,
            drift_threshold_ks=0.05
        )
    
    def test_detects_distribution_shift(self):
        """KS test should detect distribution shift."""
        # Identical distributions
        dist1 = np.random.normal(0, 1, 100)
        dist2 = np.random.normal(0, 1, 100)
        
        ks_stat_same, p_same = self.monitor._ks_test(dist1, dist2)
        
        # Different distributions
        dist3 = np.random.normal(0, 1, 100)
        dist4 = np.random.normal(2, 1, 100)  # Shifted mean
        
        ks_stat_diff, p_diff = self.monitor._ks_test(dist3, dist4)
        
        assert p_same > p_diff
        assert ks_stat_diff > ks_stat_same


class TestCUSUM:
    """Test suite for CUSUM change detection."""
    
    def setup_method(self):
        self.monitor = ClinicalRLHFMonitor(window_size=100)
    
    def test_detects_mean_shift(self):
        """CUSUM should detect sustained mean shift."""
        # Stable period
        stable = [0.7] * 50
        
        # Shift period
        shifted = [0.5] * 50
        
        combined = stable + shifted
        
        cusum_pos, cusum_neg = self.monitor._cusum(combined, target=0.7)
        
        # CUSUM should show deviation after shift
        assert max(cusum_neg[50:]) > max(cusum_neg[:50])


class TestModelComparator:
    """Test suite for A/B model comparison."""
    
    def setup_method(self):
        self.comparator = ModelComparator(
            min_samples=50,
            significance_level=0.05,
            safety_margin=0.05
        )
    
    def test_approves_better_model(self):
        """Should approve model with significantly better performance."""
        # Baseline: mediocre performance
        baseline_rewards = np.random.normal(0.6, 0.1, 100)
        baseline_safety = np.random.normal(0.85, 0.05, 100)
        
        # Candidate: better performance
        candidate_rewards = np.random.normal(0.75, 0.1, 100)
        candidate_safety = np.random.normal(0.9, 0.05, 100)
        
        result = self.comparator.compare(
            baseline_rewards=baseline_rewards,
            baseline_safety=baseline_safety,
            candidate_rewards=candidate_rewards,
            candidate_safety=candidate_safety
        )
        
        assert result.recommendation in ["APPROVE", "NEUTRAL"]
        assert result.reward_improvement > 0
    
    def test_rejects_safety_degradation(self):
        """Should reject model with safety degradation even if rewards improve."""
        # Baseline
        baseline_rewards = np.random.normal(0.6, 0.1, 100)
        baseline_safety = np.random.normal(0.9, 0.05, 100)
        
        # Candidate: better rewards but worse safety
        candidate_rewards = np.random.normal(0.8, 0.1, 100)
        candidate_safety = np.random.normal(0.75, 0.1, 100)  # Degraded!
        
        result = self.comparator.compare(
            baseline_rewards=baseline_rewards,
            baseline_safety=baseline_safety,
            candidate_rewards=candidate_rewards,
            candidate_safety=candidate_safety
        )
        
        # Should reject due to safety
        assert result.recommendation == "REJECT"
        assert result.safety_passed == False
    
    def test_neutral_on_similar_performance(self):
        """Should return NEUTRAL when differences not significant."""
        # Nearly identical distributions
        baseline_rewards = np.random.normal(0.7, 0.1, 100)
        baseline_safety = np.random.normal(0.9, 0.05, 100)
        
        candidate_rewards = np.random.normal(0.71, 0.1, 100)
        candidate_safety = np.random.normal(0.9, 0.05, 100)
        
        result = self.comparator.compare(
            baseline_rewards=baseline_rewards,
            baseline_safety=baseline_safety,
            candidate_rewards=candidate_rewards,
            candidate_safety=candidate_safety
        )
        
        assert result.recommendation == "NEUTRAL"
    
    def test_requires_min_samples(self):
        """Should require minimum samples for comparison."""
        small_baseline = np.random.normal(0.7, 0.1, 10)
        small_candidate = np.random.normal(0.8, 0.1, 10)
        
        with pytest.raises(ValueError, match="samples"):
            self.comparator.compare(
                baseline_rewards=small_baseline,
                baseline_safety=small_baseline,
                candidate_rewards=small_candidate,
                candidate_safety=small_candidate
            )


class TestComparisonResult:
    """Test suite for ComparisonResult dataclass."""
    
    def test_creation(self):
        """Should create valid result."""
        result = ComparisonResult(
            recommendation="APPROVE",
            reward_improvement=0.15,
            safety_improvement=0.02,
            reward_significant=True,
            safety_significant=False,
            safety_passed=True,
            p_value_reward=0.01,
            p_value_safety=0.35,
            details={"test": "t-test"}
        )
        
        assert result.recommendation == "APPROVE"
        assert result.safety_passed


class TestPrometheusMetrics:
    """Test suite for Prometheus metrics export (optional)."""
    
    def test_metrics_initialization(self):
        """Should initialize Prometheus metrics if available."""
        monitor = ClinicalRLHFMonitor(
            window_size=100,
            enable_prometheus=True
        )
        
        # Should not crash even if prometheus not installed
        monitor.record(reward=0.8, safety_score=0.9, latency_ms=100, is_safe=True)
    
    def test_metrics_export(self):
        """Should export metrics in Prometheus format."""
        monitor = ClinicalRLHFMonitor(
            window_size=100,
            enable_prometheus=True
        )
        
        for i in range(10):
            monitor.record(
                reward=0.7 + (i * 0.01),
                safety_score=0.9,
                latency_ms=100 + i,
                is_safe=True
            )
        
        # Get metrics (may be empty dict if prometheus not installed)
        metrics = monitor.get_prometheus_metrics()
        assert isinstance(metrics, dict)


# Fixtures for pytest
@pytest.fixture
def clinical_monitor():
    """Provide a ClinicalRLHFMonitor instance for tests."""
    return ClinicalRLHFMonitor(window_size=100)


@pytest.fixture
def model_comparator():
    """Provide a ModelComparator instance for tests."""
    return ModelComparator(min_samples=50)


# Run with: pytest tests/test_monitoring.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
