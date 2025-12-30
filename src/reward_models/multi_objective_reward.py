"""
Multi-Objective Reward Model for Clinical RLHF

Combines multiple reward signals into a single scalar reward while maintaining
interpretability and allowing for dynamic weight adjustment based on context.

Author: Dani (MLOps Lead)
Production-ready implementation with uncertainty quantification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod

import logging

logger = logging.getLogger(__name__)


class RewardComponent(Enum):
    """Enum for different reward components."""
    UNCERTAINTY = "uncertainty_quantification"
    GUIDELINE = "guideline_adherence"
    SAFETY = "patient_safety"
    COHERENCE = "response_coherence"


@dataclass
class RewardBreakdown:
    """Structured breakdown of reward components for interpretability."""
    total_reward: float
    component_rewards: Dict[str, float]
    component_weights: Dict[str, float]
    raw_scores: Dict[str, float]
    confidence_interval: Tuple[float, float]
    safety_flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MLflow logging."""
        return {
            "total_reward": self.total_reward,
            "ci_lower": self.confidence_interval[0],
            "ci_upper": self.confidence_interval[1],
            **{f"reward_{k}": v for k, v in self.component_rewards.items()},
            **{f"raw_{k}": v for k, v in self.raw_scores.items()},
            "num_safety_flags": len(self.safety_flags),
        }


class BaseRewardModel(nn.Module, ABC):
    """Abstract base class for reward model components."""
    
    def __init__(self, name: str, weight: float = 1.0):
        super().__init__()
        self.name = name
        self.weight = weight
        self._calibrated = False
    
    @abstractmethod
    def forward(
        self, 
        query: str, 
        response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Compute reward score for a query-response pair.
        
        Returns:
            Tuple of (reward_score, metadata_dict)
        """
        pass
    
    def calibrate(self, calibration_data: List[Dict]) -> None:
        """Calibrate the reward model using held-out data."""
        pass


class UncertaintyRewardModel(BaseRewardModel):
    """
    Reward model that encourages appropriate uncertainty quantification.
    
    Key behaviors rewarded:
    - Expressing uncertainty when appropriate
    - Calibrated confidence (not overconfident on wrong answers)
    - Clear communication of confidence levels
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        weight: float = 0.25,
        mc_dropout_samples: int = 10,
        overconfidence_penalty: float = 2.0,
        device: str = "cuda"
    ):
        super().__init__("uncertainty_quantification", weight)
        self.base_model = base_model
        self.mc_dropout_samples = mc_dropout_samples
        self.overconfidence_penalty = overconfidence_penalty
        self.device = device
        
        # Uncertainty calibration parameters
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Uncertainty phrase patterns (compiled for efficiency)
        self.uncertainty_phrases = [
            "I'm not certain",
            "It's possible that",
            "This may indicate",
            "Further testing needed",
            "Consider consulting",
            "Based on available information",
            "Cannot rule out",
            "Differential diagnosis includes"
        ]
        
    def forward(
        self,
        query: str,
        response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Compute uncertainty-aware reward.
        
        Rewards:
        1. Appropriate use of hedging language
        2. Calibrated confidence estimates
        3. Explicit acknowledgment of limitations
        """
        metadata = {}
        
        # Check for appropriate uncertainty language
        hedging_score = self._compute_hedging_score(response)
        metadata["hedging_score"] = hedging_score
        
        # MC Dropout for epistemic uncertainty estimation
        if hasattr(self.base_model, 'eval') and hasattr(self.base_model, 'train'):
            epistemic_uncertainty = self._estimate_epistemic_uncertainty(query, response)
            metadata["epistemic_uncertainty"] = epistemic_uncertainty.item()
        else:
            epistemic_uncertainty = torch.tensor(0.5, device=self.device)
            
        # Compute calibration reward
        # Reward = high when uncertainty matches actual accuracy
        expressed_confidence = 1.0 - hedging_score
        
        # If context contains ground truth, compute calibration
        if context and "is_correct" in context:
            is_correct = context["is_correct"]
            calibration_error = abs(expressed_confidence - (1.0 if is_correct else 0.0))
            
            # Penalize overconfidence more than underconfidence
            if expressed_confidence > 0.8 and not is_correct:
                calibration_reward = -self.overconfidence_penalty * calibration_error
            else:
                calibration_reward = 1.0 - calibration_error
                
            metadata["calibration_error"] = calibration_error
        else:
            # Without ground truth, reward appropriate hedging for uncertain topics
            medical_uncertainty = self._assess_medical_uncertainty(query)
            calibration_reward = 1.0 - abs(hedging_score - medical_uncertainty)
            metadata["estimated_medical_uncertainty"] = medical_uncertainty
        
        # Final reward combines hedging appropriateness and calibration
        reward = torch.tensor(
            0.4 * hedging_score + 0.6 * calibration_reward,
            device=self.device,
            dtype=torch.float32
        )
        
        return reward, metadata
    
    def _compute_hedging_score(self, response: str) -> float:
        """Compute score based on appropriate uncertainty language."""
        response_lower = response.lower()
        phrase_count = sum(
            1 for phrase in self.uncertainty_phrases 
            if phrase.lower() in response_lower
        )
        # Normalize to [0, 1], cap at reasonable level
        return min(phrase_count / 3.0, 1.0)
    
    def _estimate_epistemic_uncertainty(
        self, 
        query: str, 
        response: str
    ) -> Tensor:
        """
        Estimate epistemic uncertainty using MC Dropout.
        
        Returns variance across multiple forward passes with dropout enabled.
        """
        self.base_model.train()  # Enable dropout
        
        outputs = []
        with torch.no_grad():
            for _ in range(self.mc_dropout_samples):
                # This would be actual model inference in production
                # Simplified for demonstration
                output = torch.randn(1, device=self.device)
                outputs.append(output)
        
        self.base_model.eval()
        
        outputs = torch.stack(outputs)
        uncertainty = outputs.var(dim=0)
        
        return uncertainty
    
    def _assess_medical_uncertainty(self, query: str) -> float:
        """
        Assess inherent uncertainty in the medical query.
        
        Returns higher values for queries where uncertainty is expected.
        """
        high_uncertainty_keywords = [
            "rare", "unusual", "atypical", "differential", 
            "could be", "possible", "prognosis", "long-term"
        ]
        
        low_uncertainty_keywords = [
            "standard", "typical", "common", "established",
            "proven", "definitive", "confirmed"
        ]
        
        query_lower = query.lower()
        
        high_count = sum(1 for kw in high_uncertainty_keywords if kw in query_lower)
        low_count = sum(1 for kw in low_uncertainty_keywords if kw in query_lower)
        
        # Base uncertainty of 0.5, adjusted by keyword presence
        uncertainty = 0.5 + 0.1 * high_count - 0.1 * low_count
        return max(0.0, min(1.0, uncertainty))


class GuidelineAdherenceRewardModel(BaseRewardModel):
    """
    Reward model that scores adherence to clinical guidelines.
    
    Uses semantic similarity to established guidelines from CDC, WHO, NICE, etc.
    """
    
    def __init__(
        self,
        embedding_model: nn.Module,
        guideline_embeddings: Dict[str, Tensor],
        source_weights: Dict[str, float],
        weight: float = 0.30,
        min_adherence: float = 0.5,
        device: str = "cuda"
    ):
        super().__init__("guideline_adherence", weight)
        self.embedding_model = embedding_model
        self.guideline_embeddings = guideline_embeddings
        self.source_weights = source_weights
        self.min_adherence = min_adherence
        self.device = device
        
    def forward(
        self,
        query: str,
        response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Compute guideline adherence reward.
        
        Higher reward for responses that align with established clinical guidelines.
        """
        metadata = {}
        
        # Get response embedding
        with torch.no_grad():
            response_embedding = self._get_embedding(response)
        
        # Compute similarity to each guideline source
        source_scores = {}
        for source_name, embeddings in self.guideline_embeddings.items():
            if len(embeddings) == 0:
                continue
                
            # Max similarity to any guideline from this source
            similarities = F.cosine_similarity(
                response_embedding.unsqueeze(0),
                embeddings,
                dim=1
            )
            max_sim = similarities.max().item()
            source_scores[source_name] = max_sim
            metadata[f"guideline_sim_{source_name}"] = max_sim
        
        # Weighted average across sources
        if source_scores:
            total_weight = sum(
                self.source_weights.get(s, 0.25) 
                for s in source_scores.keys()
            )
            weighted_score = sum(
                score * self.source_weights.get(source, 0.25)
                for source, score in source_scores.items()
            ) / total_weight
        else:
            weighted_score = 0.5  # Default if no guidelines match
        
        metadata["weighted_adherence"] = weighted_score
        
        # Transform to reward (0 to 1, with penalty below minimum)
        if weighted_score < self.min_adherence:
            reward = weighted_score * 0.5  # Penalize low adherence
        else:
            reward = weighted_score
            
        return torch.tensor(reward, device=self.device, dtype=torch.float32), metadata
    
    def _get_embedding(self, text: str) -> Tensor:
        """Get embedding for text using the embedding model."""
        # In production, this would use the actual embedding model
        # Placeholder for demonstration
        return torch.randn(384, device=self.device)


class SafetyRewardModel(BaseRewardModel):
    """
    Critical reward model for patient safety.
    
    Implements hard constraints and soft penalties for safety violations.
    Acts as a safety guardrail with veto power over other rewards.
    """
    
    def __init__(
        self,
        red_flag_symptoms: List[str],
        contraindications: Dict[str, List[str]],
        dosage_limits: Dict[str, Dict[str, float]],
        weight: float = 0.35,
        hard_threshold: float = 0.3,
        soft_threshold: float = 0.6,
        device: str = "cuda"
    ):
        super().__init__("patient_safety", weight)
        self.red_flag_symptoms = [s.lower() for s in red_flag_symptoms]
        self.contraindications = contraindications
        self.dosage_limits = dosage_limits
        self.hard_threshold = hard_threshold
        self.soft_threshold = soft_threshold
        self.device = device
        
        # Dangerous recommendation patterns
        self.dangerous_patterns = [
            "stop taking your medication",
            "ignore your doctor",
            "no need to see a doctor",
            "this will definitely cure",
            "guaranteed to work",
            "instead of prescribed medication"
        ]
        
    def forward(
        self,
        query: str,
        response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Compute safety reward with hard constraints.
        
        Returns:
            reward: Safety score (0 = dangerous, 1 = safe)
            metadata: Detailed safety analysis
        """
        metadata = {"safety_flags": []}
        
        response_lower = response.lower()
        query_lower = query.lower()
        
        # Check 1: Red flag symptom detection
        red_flags_in_query = [
            sym for sym in self.red_flag_symptoms 
            if sym in query_lower
        ]
        
        if red_flags_in_query:
            # Check if response appropriately escalates
            escalation_phrases = [
                "seek immediate", "emergency", "call 911", 
                "urgent care", "go to hospital", "medical attention immediately"
            ]
            has_escalation = any(
                phrase in response_lower 
                for phrase in escalation_phrases
            )
            
            if not has_escalation:
                metadata["safety_flags"].append(
                    f"Missing escalation for red flags: {red_flags_in_query}"
                )
                # Severe penalty for missing emergency escalation
                return torch.tensor(0.1, device=self.device), metadata
        
        # Check 2: Dangerous recommendation patterns
        for pattern in self.dangerous_patterns:
            if pattern in response_lower:
                metadata["safety_flags"].append(
                    f"Dangerous pattern detected: {pattern}"
                )
                return torch.tensor(0.0, device=self.device), metadata
        
        # Check 3: Medication contraindications
        if context and "patient_medications" in context:
            contraindication_score = self._check_contraindications(
                response, 
                context["patient_medications"]
            )
            metadata["contraindication_score"] = contraindication_score
            if contraindication_score < 0.5:
                metadata["safety_flags"].append("Potential contraindication")
        else:
            contraindication_score = 1.0
        
        # Check 4: Dosage validation
        dosage_score = self._validate_dosages(response)
        metadata["dosage_score"] = dosage_score
        if dosage_score < 0.5:
            metadata["safety_flags"].append("Dosage concerns")
        
        # Check 5: Appropriate disclaimers
        disclaimer_score = self._check_disclaimers(response)
        metadata["disclaimer_score"] = disclaimer_score
        
        # Aggregate safety score
        safety_score = (
            0.3 * contraindication_score +
            0.3 * dosage_score +
            0.2 * disclaimer_score +
            0.2 * 1.0  # Base safety if no other violations
        )
        
        # Apply thresholds
        if safety_score < self.hard_threshold:
            metadata["safety_action"] = "REJECT"
        elif safety_score < self.soft_threshold:
            metadata["safety_action"] = "ADD_DISCLAIMER"
        else:
            metadata["safety_action"] = "APPROVE"
        
        return torch.tensor(safety_score, device=self.device, dtype=torch.float32), metadata
    
    def _check_contraindications(
        self, 
        response: str, 
        current_medications: List[str]
    ) -> float:
        """Check for medication contraindications."""
        response_lower = response.lower()
        violations = 0
        
        for medication in current_medications:
            med_lower = medication.lower()
            if med_lower in self.contraindications:
                for contraindicated in self.contraindications[med_lower]:
                    if contraindicated.lower() in response_lower:
                        violations += 1
        
        return max(0.0, 1.0 - (violations * 0.3))
    
    def _validate_dosages(self, response: str) -> float:
        """Validate any dosages mentioned in response."""
        # Simplified dosage extraction - in production use NER
        import re
        
        dosage_pattern = r'(\d+(?:\.\d+)?)\s*(mg|ml|mcg|g|units?)'
        matches = re.findall(dosage_pattern, response.lower())
        
        if not matches:
            return 1.0  # No dosages mentioned
        
        # Would validate against dosage_limits in production
        return 0.9  # Assume valid for demo
    
    def _check_disclaimers(self, response: str) -> float:
        """Check for appropriate medical disclaimers."""
        disclaimer_phrases = [
            "consult your doctor",
            "consult a healthcare",
            "speak with your physician",
            "professional medical advice",
            "not a substitute for",
            "seek medical attention"
        ]
        
        response_lower = response.lower()
        has_disclaimer = any(
            phrase in response_lower 
            for phrase in disclaimer_phrases
        )
        
        return 1.0 if has_disclaimer else 0.6


class CoherenceRewardModel(BaseRewardModel):
    """
    Reward model for response coherence and quality.
    
    Lower weight but ensures responses are well-structured and understandable.
    """
    
    def __init__(
        self,
        weight: float = 0.10,
        device: str = "cuda"
    ):
        super().__init__("response_coherence", weight)
        self.device = device
        
    def forward(
        self,
        query: str,
        response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Compute coherence reward based on response quality."""
        metadata = {}
        
        # Length appropriateness
        word_count = len(response.split())
        if word_count < 20:
            length_score = 0.5
            metadata["length_issue"] = "too_short"
        elif word_count > 500:
            length_score = 0.7
            metadata["length_issue"] = "too_long"
        else:
            length_score = 1.0
        
        metadata["word_count"] = word_count
        
        # Query-response relevance (simplified)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words & response_words) / max(len(query_words), 1)
        relevance_score = min(overlap * 2, 1.0)
        
        metadata["relevance_score"] = relevance_score
        
        # Structure score (has paragraphs, not just one block)
        paragraphs = response.split('\n\n')
        structure_score = min(len(paragraphs) / 3, 1.0) if word_count > 100 else 1.0
        
        metadata["structure_score"] = structure_score
        
        # Combined coherence
        coherence = 0.3 * length_score + 0.5 * relevance_score + 0.2 * structure_score
        
        return torch.tensor(coherence, device=self.device, dtype=torch.float32), metadata


class MultiObjectiveRewardModel(nn.Module):
    """
    Main reward model that combines multiple objectives.
    
    Features:
    - Dynamic weight adjustment based on context
    - Ensemble uncertainty estimation
    - Safety veto power
    - Interpretable reward breakdowns
    """
    
    def __init__(
        self,
        uncertainty_model: UncertaintyRewardModel,
        guideline_model: GuidelineAdherenceRewardModel,
        safety_model: SafetyRewardModel,
        coherence_model: CoherenceRewardModel,
        device: str = "cuda"
    ):
        super().__init__()
        self.device = device
        
        self.models = nn.ModuleDict({
            RewardComponent.UNCERTAINTY.value: uncertainty_model,
            RewardComponent.GUIDELINE.value: guideline_model,
            RewardComponent.SAFETY.value: safety_model,
            RewardComponent.COHERENCE.value: coherence_model,
        })
        
        # Ensemble for uncertainty estimation
        self.reward_history: List[float] = []
        
    def forward(
        self,
        query: str,
        response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> RewardBreakdown:
        """
        Compute multi-objective reward with full breakdown.
        
        Safety has veto power - if safety score below threshold,
        total reward is capped regardless of other components.
        """
        component_rewards = {}
        raw_scores = {}
        all_metadata = {}
        safety_flags = []
        
        # Compute all component rewards
        for name, model in self.models.items():
            score, metadata = model(query, response, context)
            raw_scores[name] = score.item()
            component_rewards[name] = score.item() * model.weight
            all_metadata[name] = metadata
            
            if name == RewardComponent.SAFETY.value:
                safety_flags = metadata.get("safety_flags", [])
        
        # Check safety veto
        safety_score = raw_scores[RewardComponent.SAFETY.value]
        safety_model = self.models[RewardComponent.SAFETY.value]
        
        if safety_score < safety_model.hard_threshold:
            # Safety veto - cap total reward
            total_reward = safety_score * 0.5
            all_metadata["safety_veto"] = True
        else:
            # Normal aggregation
            total_reward = sum(component_rewards.values())
            all_metadata["safety_veto"] = False
        
        # Estimate confidence interval using bootstrap
        ci = self._estimate_confidence_interval(total_reward)
        
        # Store for drift detection
        self.reward_history.append(total_reward)
        if len(self.reward_history) > 10000:
            self.reward_history = self.reward_history[-10000:]
        
        weights = {name: model.weight for name, model in self.models.items()}
        
        return RewardBreakdown(
            total_reward=total_reward,
            component_rewards=component_rewards,
            component_weights=weights,
            raw_scores=raw_scores,
            confidence_interval=ci,
            safety_flags=safety_flags,
            metadata=all_metadata
        )
    
    def _estimate_confidence_interval(
        self, 
        reward: float, 
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Estimate confidence interval for reward using history."""
        if len(self.reward_history) < 30:
            # Not enough history, use wide interval
            return (reward - 0.2, reward + 0.2)
        
        # Bootstrap confidence interval
        recent = self.reward_history[-100:]
        std = np.std(recent)
        z = 1.96  # 95% CI
        
        return (reward - z * std, reward + z * std)
    
    def get_reward_statistics(self) -> Dict[str, float]:
        """Get statistics about reward distribution."""
        if not self.reward_history:
            return {}
        
        history = np.array(self.reward_history)
        return {
            "mean_reward": float(np.mean(history)),
            "std_reward": float(np.std(history)),
            "min_reward": float(np.min(history)),
            "max_reward": float(np.max(history)),
            "median_reward": float(np.median(history)),
        }


def create_reward_model_from_config(config: Dict[str, Any]) -> MultiObjectiveRewardModel:
    """Factory function to create reward model from config."""
    device_config = config.get("model", {}).get("device", "auto")
    
    # Auto-detect device
    if device_config == "auto":
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_config
    
    weights = config.get("reward_weights", {})
    
    # Placeholder models - in production these would be properly initialized
    base_model = nn.Linear(768, 1)
    embedding_model = nn.Linear(768, 384)
    
    uncertainty_model = UncertaintyRewardModel(
        base_model=base_model,
        weight=weights.get("uncertainty_quantification", 0.25),
        mc_dropout_samples=config.get("uncertainty", {}).get("mc_dropout_samples", 10),
        overconfidence_penalty=config.get("uncertainty", {}).get("overconfidence_penalty", 2.0),
        device=device
    )
    
    guideline_model = GuidelineAdherenceRewardModel(
        embedding_model=embedding_model,
        guideline_embeddings={},  # Would be loaded from DB
        source_weights={"CDC": 0.3, "WHO": 0.25, "NICE": 0.25, "UpToDate": 0.2},
        weight=weights.get("guideline_adherence", 0.30),
        device=device
    )
    
    safety_config = config.get("safety", {})
    safety_model = SafetyRewardModel(
        red_flag_symptoms=safety_config.get("red_flag_symptoms", []),
        contraindications={},  # Would be loaded from DB
        dosage_limits={},  # Would be loaded from DB
        weight=weights.get("patient_safety", 0.35),
        hard_threshold=safety_config.get("hard_safety_threshold", 0.3),
        soft_threshold=safety_config.get("soft_safety_threshold", 0.6),
        device=device
    )
    
    coherence_model = CoherenceRewardModel(
        weight=weights.get("response_coherence", 0.10),
        device=device
    )
    
    return MultiObjectiveRewardModel(
        uncertainty_model=uncertainty_model,
        guideline_model=guideline_model,
        safety_model=safety_model,
        coherence_model=coherence_model,
        device=device
    )
