"""
Pytest Configuration and Shared Fixtures

Provides common fixtures for Clinical RLHF Pipeline tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

import torch


@pytest.fixture(scope="session")
def device():
    """Get available device for testing."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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
        "memory_monitoring": {
            "warning_threshold": 0.70,
            "critical_threshold": 0.85,
            "emergency_threshold": 0.95,
            "raise_on_emergency": False,
        },
        "rollback": {
            "min_safety_score": 0.5,
            "max_unsafe_ratio": 0.15,
            "cooldown_steps": 0,
        },
        "hallucination_detection": {
            "data_dir": "data",
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
                "target_kl": 10.0,
            },
            "total_steps": 10,
            "eval_frequency": 5,
            "save_frequency": 5,
            "max_unsafe_ratio": 0.9,
        },
    }


@pytest.fixture
def mock_policy_model():
    """Create mock policy model for testing."""
    import torch.nn as nn
    
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
            device = input_ids.device
            gen = torch.randint(0, 1000, (batch, 20), device=device)
            return type('GenOutput', (), {
                'sequences': torch.cat([input_ids, gen], dim=1)
            })()
    
    return MockPolicy()


@pytest.fixture
def mock_value_model():
    """Create mock value model for testing."""
    import torch.nn as nn
    
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
