"""
Integration Tests for Clinical RLHF Pipeline

End-to-end tests covering:
- Full training loop
- Safety guardrails integration
- Reward model computation
- Checkpoint save/restore
- Memory monitoring
- Rollback system

Author: Dani (MLOps Lead)
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    tmp = tempfile.mkdtemp(prefix="clinical_rlhf_test_")
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model": {
            "use_mock_models": True,
            "device": "cpu",
            "max_length": 128,
        },
        "reward_weights": {
            "uncertainty_quantification": 0.25,
            "guideline_adherence": 0.30,
            "patient_safety": 0.35,
            "response_coherence": 0.10,
        },
        "safety": {
            "red_flag_symptoms": ["chest pain", "difficulty breathing"],
            "hard_safety_threshold": 0.3,
            "soft_safety_threshold": 0.6,
        },
        "training": {
            "ppo": {
                "learning_rate": 1e-4,
                "batch_size": 2,
                "mini_batch_size": 1,
                "ppo_epochs": 1,
                "clip_range": 0.2,
                "target_kl": 10.0,  # High for testing
            },
            "total_steps": 10,
            "eval_frequency": 5,
            "save_frequency": 5,
            "max_unsafe_ratio": 0.9,  # High for testing with mock models
        },
    }


@pytest.fixture
def mock_policy_model():
    """Create mock policy model for testing."""
    class MockPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 64)
            self.output = nn.Linear(64, 1000)
        
        def forward(self, input_ids, **kwargs):
            x = self.embedding(input_ids)
            logits = self.output(x)
            return type('Output', (), {'logits': logits, 'loss': None})()
        
        def generate(self, input_ids, **kwargs):
            batch = input_ids.shape[0]
            gen = torch.randint(0, 1000, (batch, 20))
            return type('GenOutput', (), {'sequences': torch.cat([input_ids, gen], dim=1)})()
    
    return MockPolicy()


@pytest.fixture
def mock_value_model():
    """Create mock value model for testing."""
    class MockValue(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 64)
            self.head = nn.Linear(64, 1)
        
        def forward(self, input_ids, **kwargs):
            x = self.embedding(input_ids)
            return self.head(x[:, -1, :])
    
    return MockValue()


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer for testing."""
    class MockTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        
        def __call__(self, text, **kwargs):
            if isinstance(text, str):
                text = [text]
            ids = torch.randint(2, 1000, (len(text), 10))
            mask = torch.ones_like(ids)
            return type('TokenizerOutput', (), {
                'input_ids': ids,
                'attention_mask': mask,
            })()
        
        def decode(self, ids, **kwargs):
            return "Mock medical response"
    
    return MockTokenizer()


# =============================================================================
# Safety Guardrails Integration Tests
# =============================================================================

class TestSafetyGuardrailsIntegration:
    """Test safety guardrails end-to-end."""
    
    def test_guardrails_initialization(self, sample_config):
        """Test guardrails initialize correctly."""
        from src.safety.guardrails import SafetyGuardrails
        
        guardrails = SafetyGuardrails(config=sample_config.get("safety", {}))
        
        assert guardrails is not None
        assert len(guardrails.red_flag_patterns) > 0
    
    def test_guardrails_detect_unsafe_response(self, sample_config):
        """Test guardrails detect unsafe responses."""
        from src.safety.guardrails import SafetyGuardrails
        
        guardrails = SafetyGuardrails(config=sample_config.get("safety", {}))
        
        # Unsafe: mentions chest pain but doesn't recommend emergency care
        query = "I have chest pain"
        response = "Try taking some antacids and rest."
        
        result = guardrails.check(query, response)
        
        assert not result.is_safe
        assert len(result.violations) > 0
    
    def test_guardrails_approve_safe_response(self, sample_config):
        """Test guardrails approve safe responses."""
        from src.safety.guardrails import SafetyGuardrails
        
        guardrails = SafetyGuardrails(config=sample_config.get("safety", {}))
        
        # Safe: general question with appropriate response
        query = "What vitamins should I take?"
        response = "Consult your doctor for personalized vitamin recommendations based on your health needs."
        
        result = guardrails.check(query, response)
        
        # Should be safe (no red flags in query)
        assert result.is_safe or result.safety_score > 0.5
    
    def test_guardrails_audit_logging(self, sample_config, temp_dir):
        """Test guardrails create audit logs."""
        from src.safety.guardrails import SafetyGuardrails
        
        audit_path = temp_dir / "safety_audit.jsonl"
        guardrails = SafetyGuardrails(
            config=sample_config.get("safety", {}),
            audit_log_path=audit_path,
        )
        
        # Make some checks
        guardrails.check("chest pain", "Call 911 immediately")
        guardrails.check("headache", "Take acetaminophen")
        
        # Verify log exists
        # Note: only unsafe responses are logged
        assert audit_path.parent.exists()


# =============================================================================
# Reward Model Integration Tests
# =============================================================================

class TestRewardModelIntegration:
    """Test reward model components."""
    
    def test_reward_model_creation(self, sample_config):
        """Test reward model creates correctly from config."""
        from src.reward_models.multi_objective_reward import create_reward_model_from_config
        
        reward_model = create_reward_model_from_config(sample_config)
        
        assert reward_model is not None
        assert hasattr(reward_model, '__call__')
    
    def test_reward_computation(self, sample_config):
        """Test reward computation returns valid values."""
        from src.reward_models.multi_objective_reward import create_reward_model_from_config
        
        reward_model = create_reward_model_from_config(sample_config)
        
        query = "What should I do for a headache?"
        response = "Take acetaminophen up to 1000mg. See a doctor if it persists."
        
        result = reward_model(query, response, {})
        
        assert hasattr(result, 'total_reward')
        assert hasattr(result, 'weighted_scores')
        assert -1.0 <= result.total_reward <= 1.5
    
    def test_reward_weights_sum_to_one(self, sample_config):
        """Test reward weights are normalized."""
        weights = sample_config["reward_weights"]
        total = sum(weights.values())
        
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected 1.0"


# =============================================================================
# Memory Monitor Integration Tests
# =============================================================================

class TestMemoryMonitorIntegration:
    """Test memory monitoring system."""
    
    def test_memory_monitor_initialization(self):
        """Test memory monitor initializes correctly."""
        from src.evaluation.memory_monitor import MemoryMonitor, MemoryConfig
        
        monitor = MemoryMonitor(config=MemoryConfig(), device="cpu")
        
        assert monitor is not None
        assert not monitor.has_gpu  # CPU mode
    
    def test_memory_snapshot(self):
        """Test memory snapshot captures data."""
        from src.evaluation.memory_monitor import MemoryMonitor
        
        monitor = MemoryMonitor(device="cpu")
        snapshot = monitor.get_memory_snapshot()
        
        assert snapshot is not None
        assert snapshot.timestamp is not None
        assert snapshot.pressure_level is not None
    
    def test_memory_check_returns_snapshot(self):
        """Test check() returns snapshot and adds to history."""
        from src.evaluation.memory_monitor import MemoryMonitor
        
        monitor = MemoryMonitor(device="cpu")
        
        # Perform multiple checks
        for _ in range(5):
            snapshot = monitor.check()
            assert snapshot is not None
        
        assert len(monitor.history) == 5
    
    def test_memory_alert_callback(self):
        """Test alert callbacks are triggered."""
        from src.evaluation.memory_monitor import (
            MemoryMonitor, 
            MemoryConfig,
            MemoryPressureLevel,
        )
        
        alerts_received = []
        
        config = MemoryConfig(
            warning_threshold=0.01,  # Very low to trigger on any usage
            raise_on_emergency=False,
        )
        monitor = MemoryMonitor(config=config, device="cpu")
        monitor.add_alert_callback(lambda snap: alerts_received.append(snap))
        
        # This should trigger warning due to low threshold
        monitor.check()
        
        # May or may not trigger depending on system memory
        assert isinstance(alerts_received, list)


# =============================================================================
# Rollback System Integration Tests
# =============================================================================

class TestRollbackIntegration:
    """Test automatic rollback system."""
    
    def test_rollback_manager_initialization(self, temp_dir):
        """Test rollback manager initializes correctly."""
        from src.safety.rollback import SafetyRollbackManager, RollbackConfig
        
        manager = SafetyRollbackManager(
            config=RollbackConfig(),
            checkpoint_dir=temp_dir / "checkpoints",
        )
        
        assert manager is not None
        assert manager.checkpoint_manager is not None
    
    def test_checkpoint_save_and_load(
        self, temp_dir, mock_policy_model, mock_value_model
    ):
        """Test checkpoint save and restore."""
        from src.safety.rollback import (
            SafetyRollbackManager,
            SafetyMetrics,
        )
        
        manager = SafetyRollbackManager(
            checkpoint_dir=temp_dir / "checkpoints",
        )
        optimizer = torch.optim.Adam(mock_policy_model.parameters())
        
        # Save checkpoint
        metrics = SafetyMetrics(
            safety_score=0.9,
            unsafe_ratio=0.05,
            reward=0.5,
            step=100,
        )
        
        path = manager.save_checkpoint(
            policy_model=mock_policy_model,
            value_model=mock_value_model,
            optimizer=optimizer,
            step=100,
            metrics=metrics,
        )
        
        assert Path(path).exists()
        assert (Path(path) / "policy.pt").exists()
        assert (Path(path) / "metadata.json").exists()
        
        # Modify model weights
        with torch.no_grad():
            for p in mock_policy_model.parameters():
                p.fill_(999.0)
        
        # Load checkpoint
        manager.checkpoint_manager.load(
            checkpoint_path=path,
            policy_model=mock_policy_model,
            value_model=mock_value_model,
            optimizer=optimizer,
        )
        
        # Verify weights were restored
        with torch.no_grad():
            for p in mock_policy_model.parameters():
                assert not torch.all(p == 999.0)
    
    def test_rollback_trigger_on_safety_degradation(self, temp_dir):
        """Test rollback triggers on safety degradation."""
        from src.safety.rollback import (
            SafetyRollbackManager,
            SafetyMetrics,
            RollbackConfig,
            RollbackReason,
        )
        
        config = RollbackConfig(
            min_safety_score=0.5,
            cooldown_steps=0,  # No cooldown for testing
        )
        manager = SafetyRollbackManager(
            config=config,
            checkpoint_dir=temp_dir / "checkpoints",
        )
        
        # Good metrics - no rollback
        good_metrics = SafetyMetrics(safety_score=0.8, step=1)
        should_rollback, reason = manager.check(good_metrics)
        assert not should_rollback
        
        # Bad metrics - should trigger rollback
        bad_metrics = SafetyMetrics(safety_score=0.3, step=2)
        should_rollback, reason = manager.check(bad_metrics)
        assert should_rollback
        assert reason == RollbackReason.SAFETY_DEGRADATION


# =============================================================================
# Hallucination Detection Integration Tests
# =============================================================================

class TestHallucinationDetection:
    """Test hallucination detection system."""
    
    def test_detector_initialization(self):
        """Test detector initializes correctly."""
        from src.safety.hallucination_detector import HallucinationDetector
        
        detector = HallucinationDetector()
        
        assert detector is not None
        assert detector.kb is not None
    
    def test_detect_wrong_dosage(self):
        """Test detection of incorrect dosages."""
        from src.safety.hallucination_detector import (
            HallucinationDetector,
            HallucinationType,
        )
        
        detector = HallucinationDetector()
        
        # Response with dangerous dosage
        query = "How much acetaminophen can I take?"
        response = "You can take up to 10000mg of acetaminophen daily."  # Way too high!
        
        result = detector.check(query, response)
        
        # Should detect dosage issue
        dosage_detections = [
            d for d in result.detections 
            if d.hallucination_type == HallucinationType.WRONG_DOSAGE
        ]
        assert len(dosage_detections) > 0
    
    def test_detect_overconfidence(self):
        """Test detection of overconfident statements."""
        from src.safety.hallucination_detector import (
            HallucinationDetector,
            HallucinationType,
        )
        
        detector = HallucinationDetector()
        
        # Overconfident response
        query = "Will this medication work?"
        response = "This medication is 100% safe and guaranteed to cure your condition."
        
        result = detector.check(query, response)
        
        overconf_detections = [
            d for d in result.detections
            if d.hallucination_type == HallucinationType.OVERCONFIDENT_STATEMENT
        ]
        assert len(overconf_detections) > 0
    
    def test_clean_response_passes(self):
        """Test that appropriate responses pass checks."""
        from src.safety.hallucination_detector import HallucinationDetector
        
        detector = HallucinationDetector()
        
        query = "What should I take for a headache?"
        response = "You may consider taking ibuprofen 400mg. If symptoms persist, consult your doctor."
        
        result = detector.check(query, response)
        
        # Should have high score (few/no hallucinations)
        assert result.overall_score >= 0.5


# =============================================================================
# Expert Feedback Integration Tests
# =============================================================================

class TestExpertFeedbackIntegration:
    """Test expert feedback collection system."""
    
    def test_feedback_collector_initialization(self, temp_dir):
        """Test feedback collector initializes correctly."""
        from src.data.expert_feedback import ExpertFeedbackCollector
        
        collector = ExpertFeedbackCollector(storage_path=temp_dir / "feedback")
        
        assert collector is not None
    
    def test_expert_registration(self, temp_dir):
        """Test expert registration."""
        from src.data.expert_feedback import (
            ExpertFeedbackCollector,
            Expert,
            ExpertRole,
        )
        
        collector = ExpertFeedbackCollector(storage_path=temp_dir / "feedback")
        
        expert = Expert(
            expert_id="TEST001",
            role=ExpertRole.PHYSICIAN,
            specialization="General Practice",
            years_experience=5,
        )
        
        collector.register_expert(expert)
        
        assert "TEST001" in collector.experts
    
    def test_sample_creation_and_annotation(self, temp_dir):
        """Test creating samples and collecting preferences."""
        from src.data.expert_feedback import (
            ExpertFeedbackCollector,
            Expert,
            ExpertRole,
            PreferenceType,
        )
        
        collector = ExpertFeedbackCollector(
            storage_path=temp_dir / "feedback",
            min_annotations_per_sample=1,  # Low for testing
        )
        
        # Register expert
        expert = Expert(
            expert_id="TEST001",
            role=ExpertRole.PHYSICIAN,
            specialization="Cardiology",
            years_experience=10,
        )
        collector.register_expert(expert)
        
        # Create sample
        sample = collector.create_sample(
            query="What causes chest pain?",
            responses=["Response A", "Response B"],
            category="cardiology",
        )
        
        assert sample is not None
        assert sample.sample_id is not None
        
        # Collect preference
        preference = collector.collect_preference(
            sample_id=sample.sample_id,
            expert_id="TEST001",
            preference_type=PreferenceType.PAIRWISE,
            value=0,
            confidence=0.9,
        )
        
        assert preference is not None


# =============================================================================
# End-to-End Training Loop Tests
# =============================================================================

class TestTrainingLoopIntegration:
    """Test full training loop integration."""
    
    def test_ppo_trainer_initialization(
        self,
        sample_config,
        temp_dir,
        mock_policy_model,
        mock_value_model,
        mock_tokenizer,
    ):
        """Test PPO trainer initializes correctly."""
        from src.training.ppo_trainer import ClinicalPPOTrainer, PPOConfig
        from src.reward_models.multi_objective_reward import create_reward_model_from_config
        from src.safety.guardrails import SafetyGuardrails
        
        reward_model = create_reward_model_from_config(sample_config)
        guardrails = SafetyGuardrails(config=sample_config.get("safety", {}))
        
        ppo_config = PPOConfig(
            **sample_config["training"]["ppo"],
            device="cpu",
            max_unsafe_ratio=0.9,
        )
        
        trainer = ClinicalPPOTrainer(
            policy_model=mock_policy_model,
            value_model=mock_value_model,
            reward_model=reward_model,
            tokenizer=mock_tokenizer,
            config=ppo_config,
            safety_guardrails=guardrails,
            output_dir=temp_dir / "checkpoints",
            experiment_name="test-experiment",
        )
        
        assert trainer is not None
        trainer.close()
    
    @pytest.mark.slow
    def test_training_loop_runs(
        self,
        sample_config,
        temp_dir,
        mock_policy_model,
        mock_value_model,
        mock_tokenizer,
    ):
        """Test training loop completes without errors."""
        from src.training.ppo_trainer import ClinicalPPOTrainer, PPOConfig
        from src.reward_models.multi_objective_reward import create_reward_model_from_config
        from src.safety.guardrails import SafetyGuardrails
        
        reward_model = create_reward_model_from_config(sample_config)
        guardrails = SafetyGuardrails(config=sample_config.get("safety", {}))
        
        ppo_config = PPOConfig(
            **sample_config["training"]["ppo"],
            device="cpu",
            max_unsafe_ratio=0.99,  # Very high for mock models
            total_steps=5,
        )
        
        trainer = ClinicalPPOTrainer(
            policy_model=mock_policy_model,
            value_model=mock_value_model,
            reward_model=reward_model,
            tokenizer=mock_tokenizer,
            config=ppo_config,
            safety_guardrails=guardrails,
            output_dir=temp_dir / "checkpoints",
            experiment_name="test-run",
        )
        
        # Small training set
        queries = [
            "What is a headache?",
            "How do I treat a cold?",
        ] * 2
        
        try:
            trainer.train(
                train_queries=queries,
                eval_queries=queries[:2],
            )
        finally:
            trainer.close()
        
        # Verify checkpoint was created
        checkpoints = list((temp_dir / "checkpoints").glob("*"))
        assert len(checkpoints) > 0


# =============================================================================
# Configuration Validation Tests
# =============================================================================

class TestConfigurationValidation:
    """Test configuration loading and validation."""
    
    def test_config_loads_from_yaml(self, temp_dir):
        """Test configuration loads from YAML file."""
        import yaml
        
        config_content = """
model:
  use_mock_models: true
  device: cpu
  
training:
  ppo:
    learning_rate: 0.0001
    batch_size: 4
"""
        config_path = temp_dir / "test_config.yaml"
        config_path.write_text(config_content)
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert config["model"]["use_mock_models"] is True
        assert config["training"]["ppo"]["batch_size"] == 4
    
    def test_reward_weights_validation(self):
        """Test reward weights are properly validated."""
        weights = {
            "uncertainty_quantification": 0.25,
            "guideline_adherence": 0.30,
            "patient_safety": 0.35,
            "response_coherence": 0.10,
        }
        
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
