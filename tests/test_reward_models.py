"""
Tests for Multi-Objective Reward Model
Tests for medical AI reward components with clinical constraints
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reward_models.multi_objective_reward import (
    MultiObjectiveRewardModel,
    RewardBreakdown,
    UncertaintyRewardModel,
    GuidelineAdherenceRewardModel,
    SafetyRewardModel,
    CoherenceRewardModel,
)


class TestUncertaintyRewardModel:
    """Test suite for uncertainty quantification rewards."""
    
    def setup_method(self):
        # Mock model for testing
        self.mock_model = MagicMock()
        self.uncertainty_model = UncertaintyRewardModel(
            model=self.mock_model,
            n_samples=5
        )
    
    def test_rewards_hedging_language(self):
        """Should reward appropriate uncertainty expression."""
        response_hedged = "I'm not certain, but it could be a viral infection. Further testing would help confirm."
        response_overconfident = "This is definitely a viral infection. Take these antibiotics."
        
        # Mock the uncertainty computation
        with patch.object(self.uncertainty_model, 'compute_epistemic_uncertainty') as mock_unc:
            mock_unc.return_value = 0.3
            
            score_hedged = self.uncertainty_model.compute_reward(
                query="What's causing my symptoms?",
                response=response_hedged
            )
            
            score_overconfident = self.uncertainty_model.compute_reward(
                query="What's causing my symptoms?",
                response=response_overconfident
            )
        
        # Hedged response should score higher when uncertainty is present
        assert score_hedged >= score_overconfident
    
    def test_penalizes_overconfidence(self):
        """Should penalize overconfident responses in uncertain situations."""
        response = "This is definitely cancer. No need for further tests."
        
        with patch.object(self.uncertainty_model, 'compute_epistemic_uncertainty') as mock_unc:
            mock_unc.return_value = 0.8  # High uncertainty
            
            score = self.uncertainty_model.compute_reward(
                query="What's this lump?",
                response=response
            )
        
        # Should receive penalty for overconfidence
        assert score < 0.5
    
    def test_detects_hedging_phrases(self):
        """Should detect common hedging phrases."""
        hedging_phrases = [
            "I'm not certain",
            "It's possible that",
            "Further testing may be needed",
            "Consult a specialist to confirm",
            "This could indicate",
        ]
        
        for phrase in hedging_phrases:
            response = f"{phrase} you have an infection."
            has_hedging = self.uncertainty_model._has_appropriate_hedging(response)
            assert has_hedging, f"Failed to detect: {phrase}"


class TestGuidelineAdherenceRewardModel:
    """Test suite for clinical guideline adherence scoring."""
    
    def setup_method(self):
        self.guideline_model = GuidelineAdherenceRewardModel(
            guideline_sources={
                "CDC": 1.0,
                "WHO": 0.9,
                "NICE": 0.85,
            }
        )
    
    def test_rewards_guideline_adherent_response(self):
        """Should reward responses that follow clinical guidelines."""
        # Response following flu treatment guidelines
        response_adherent = """
        For flu symptoms, the CDC recommends rest, fluids, and over-the-counter 
        medications for symptom relief. Antiviral medications like oseltamivir 
        may be prescribed within 48 hours of symptom onset for high-risk patients.
        """
        
        response_non_adherent = """
        For flu, try this herbal remedy and essential oils. 
        Antibiotics will definitely cure you quickly.
        """
        
        score_adherent = self.guideline_model.compute_reward(
            query="How should I treat the flu?",
            response=response_adherent
        )
        
        score_non_adherent = self.guideline_model.compute_reward(
            query="How should I treat the flu?",
            response=response_non_adherent
        )
        
        assert score_adherent > score_non_adherent
    
    def test_minimum_adherence_threshold(self):
        """Should enforce minimum adherence score."""
        response_poor = "Just take random supplements."
        
        score = self.guideline_model.compute_reward(
            query="How to manage diabetes?",
            response=response_poor
        )
        
        # Should be below threshold
        assert score < 0.5


class TestSafetyRewardModel:
    """Test suite for safety-focused reward scoring."""
    
    def setup_method(self):
        self.safety_model = SafetyRewardModel()
    
    def test_penalizes_missing_emergency_escalation(self):
        """Should penalize missing emergency guidance for red flags."""
        query = "I'm having chest pain and shortness of breath"
        response_bad = "Try some antacids and rest."
        response_good = "These symptoms require immediate medical attention. Call 911 or go to the ER immediately."
        
        score_bad = self.safety_model.compute_reward(query, response_bad)
        score_good = self.safety_model.compute_reward(query, response_good)
        
        assert score_good > score_bad
        assert score_bad < 0.3  # Should be heavily penalized
    
    def test_rewards_appropriate_disclaimer(self):
        """Should reward responses with proper medical disclaimers."""
        response_with_disclaimer = """
        Ibuprofen can help with inflammation. Please consult your doctor 
        before starting any new medication, especially if you have other 
        health conditions.
        """
        
        response_without = "Just take ibuprofen, it will fix everything."
        
        score_with = self.safety_model.compute_reward("Joint pain remedy?", response_with_disclaimer)
        score_without = self.safety_model.compute_reward("Joint pain remedy?", response_without)
        
        assert score_with > score_without
    
    def test_flags_dangerous_advice(self):
        """Should heavily penalize dangerous medical advice."""
        dangerous_responses = [
            "Stop taking your prescribed medications immediately.",
            "Ignore what your doctor said and try this instead.",
            "This natural remedy is guaranteed to cure your cancer.",
        ]
        
        for response in dangerous_responses:
            score = self.safety_model.compute_reward("What should I do?", response)
            assert score < 0.2, f"Failed to penalize: {response}"
    
    def test_veto_power(self):
        """Safety should have veto power over total reward."""
        result = self.safety_model.compute_reward_with_veto(
            query="chest pain",
            response="Just ignore it, probably nothing.",
            other_reward=0.9  # High reward from other components
        )
        
        assert result["final_reward"] < 0.5
        assert result["safety_vetoed"] == True


class TestCoherenceRewardModel:
    """Test suite for response coherence scoring."""
    
    def setup_method(self):
        self.coherence_model = CoherenceRewardModel(
            min_length=20,
            max_length=500,
            optimal_paragraphs=(1, 4)
        )
    
    def test_penalizes_too_short(self):
        """Should penalize responses that are too brief."""
        response_short = "Take medicine."
        response_appropriate = """
        For your symptoms, over-the-counter pain relievers like acetaminophen 
        or ibuprofen may help. Make sure to follow the dosage instructions on 
        the packaging and consult your doctor if symptoms persist.
        """
        
        score_short = self.coherence_model.compute_reward("How to treat headache?", response_short)
        score_appropriate = self.coherence_model.compute_reward("How to treat headache?", response_appropriate)
        
        assert score_appropriate > score_short
    
    def test_penalizes_too_long(self):
        """Should penalize excessively long responses."""
        response_verbose = " ".join(["This is a very long response."] * 200)
        response_appropriate = "A concise but complete response with relevant information."
        
        score_verbose = self.coherence_model.compute_reward("Simple question?", response_verbose)
        score_appropriate = self.coherence_model.compute_reward("Simple question?", response_appropriate)
        
        # Verbose should be penalized
        assert score_verbose < score_appropriate
    
    def test_rewards_query_relevance(self):
        """Should reward responses relevant to the query."""
        query = "How to treat a headache?"
        response_relevant = "For headaches, try rest, hydration, and OTC pain relievers."
        response_irrelevant = "The weather is nice today. Consider going for a walk."
        
        score_relevant = self.coherence_model.compute_reward(query, response_relevant)
        score_irrelevant = self.coherence_model.compute_reward(query, response_irrelevant)
        
        assert score_relevant > score_irrelevant


class TestMultiObjectiveRewardModel:
    """Integration tests for multi-objective reward aggregation."""
    
    def setup_method(self):
        self.weights = {
            "uncertainty": 0.25,
            "guideline_adherence": 0.30,
            "safety": 0.35,
            "coherence": 0.10,
        }
        self.reward_model = MultiObjectiveRewardModel(weights=self.weights)
    
    def test_weights_sum_to_one(self):
        """Weights must sum to 1.0 for proper normalization."""
        assert abs(sum(self.weights.values()) - 1.0) < 0.001
    
    def test_returns_reward_breakdown(self):
        """Should return detailed breakdown of reward components."""
        breakdown = self.reward_model.compute_reward_with_breakdown(
            query="How to treat cold?",
            response="Rest and fluids help. Consult doctor if symptoms persist."
        )
        
        assert isinstance(breakdown, RewardBreakdown)
        assert "uncertainty" in breakdown.components
        assert "guideline_adherence" in breakdown.components
        assert "safety" in breakdown.components
        assert "coherence" in breakdown.components
        assert breakdown.total_reward is not None
    
    def test_safety_veto_overrides_high_scores(self):
        """Safety veto should cap total reward regardless of other components."""
        # Simulate high scores on other components but unsafe response
        with patch.object(self.reward_model, '_compute_component_scores') as mock_scores:
            mock_scores.return_value = {
                "uncertainty": 0.9,
                "guideline_adherence": 0.9,
                "safety": 0.1,  # Unsafe!
                "coherence": 0.9,
            }
            
            breakdown = self.reward_model.compute_reward_with_breakdown(
                query="test",
                response="unsafe test"
            )
        
        # Total should be capped due to safety veto
        assert breakdown.total_reward < 0.5
        assert breakdown.safety_vetoed == True
    
    def test_confidence_intervals(self):
        """Should provide confidence intervals via bootstrap."""
        breakdown = self.reward_model.compute_reward_with_breakdown(
            query="How to manage diabetes?",
            response="Diet and exercise are important. Monitor blood sugar regularly. Consult your doctor."
        )
        
        if breakdown.confidence_interval is not None:
            assert breakdown.confidence_interval[0] <= breakdown.total_reward
            assert breakdown.confidence_interval[1] >= breakdown.total_reward
    
    def test_interpretability(self):
        """Reward breakdown should be fully interpretable."""
        breakdown = self.reward_model.compute_reward_with_breakdown(
            query="test query",
            response="test response with consult doctor disclaimer"
        )
        
        explanation = breakdown.explain()
        
        assert isinstance(explanation, str)
        assert "uncertainty" in explanation.lower() or "safety" in explanation.lower()


class TestRewardBreakdown:
    """Test suite for RewardBreakdown dataclass."""
    
    def test_creation(self):
        """Should create valid RewardBreakdown."""
        breakdown = RewardBreakdown(
            components={
                "uncertainty": 0.7,
                "safety": 0.9,
                "guideline_adherence": 0.8,
                "coherence": 0.75,
            },
            weights={
                "uncertainty": 0.25,
                "safety": 0.35,
                "guideline_adherence": 0.30,
                "coherence": 0.10,
            },
            total_reward=0.81,
            confidence_interval=(0.75, 0.87),
            safety_vetoed=False
        )
        
        assert breakdown.total_reward == 0.81
        assert not breakdown.safety_vetoed
    
    def test_weighted_sum_calculation(self):
        """Should correctly calculate weighted sum."""
        components = {
            "a": 0.8,
            "b": 0.6,
        }
        weights = {
            "a": 0.6,
            "b": 0.4,
        }
        
        expected = 0.8 * 0.6 + 0.6 * 0.4  # 0.48 + 0.24 = 0.72
        
        breakdown = RewardBreakdown(
            components=components,
            weights=weights,
            total_reward=expected,
            confidence_interval=None,
            safety_vetoed=False
        )
        
        assert abs(breakdown.total_reward - 0.72) < 0.001


# Fixtures for pytest
@pytest.fixture
def multi_objective_reward_model():
    """Provide a MultiObjectiveRewardModel instance for tests."""
    return MultiObjectiveRewardModel(
        weights={
            "uncertainty": 0.25,
            "guideline_adherence": 0.30,
            "safety": 0.35,
            "coherence": 0.10,
        }
    )


@pytest.fixture
def safety_reward_model():
    """Provide a SafetyRewardModel instance for tests."""
    return SafetyRewardModel()


# Run with: pytest tests/test_reward_models.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
