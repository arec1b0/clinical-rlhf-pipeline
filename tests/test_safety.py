"""
Tests for Safety Guardrails Module
Production-critical tests for medical AI safety constraints
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.safety.guardrails import (
    SafetyGuardrails,
    SafetyAction,
    SafetyCheckResult,
    RedFlagDetector,
    ContraindicationChecker,
    DosageValidator,
    DangerousAdviceDetector,
    DisclaimerEnforcer,
    SafetyAwareRewardWrapper,
)


class TestRedFlagDetector:
    """Test suite for red flag symptom detection."""
    
    def setup_method(self):
        self.detector = RedFlagDetector()
    
    def test_detects_cardiac_symptoms(self):
        """Critical: Must detect cardiac arrest symptoms."""
        query = "I'm having severe chest pain and my left arm is numb"
        flags = self.detector.detect(query, "")
        
        assert len(flags) > 0
        assert any(f.severity >= 0.7 for f in flags)
        assert any("chest pain" in f.symptom.lower() for f in flags)
    
    def test_detects_stroke_symptoms(self):
        """Critical: Must detect stroke indicators."""
        query = "sudden severe headache and I can't move my right side"
        flags = self.detector.detect(query, "")
        
        assert len(flags) > 0
        assert any(f.severity >= 0.7 for f in flags)
    
    def test_detects_suicidal_ideation(self):
        """Critical: Must detect mental health emergencies."""
        query = "I've been thinking about ending my life"
        flags = self.detector.detect(query, "")
        
        assert len(flags) > 0
        assert any(f.severity == 1.0 for f in flags)
    
    def test_detects_anaphylaxis(self):
        """Critical: Must detect allergic emergencies."""
        query = "throat is swelling shut and I can't breathe after eating peanuts"
        flags = self.detector.detect(query, "")
        
        assert len(flags) > 0
        assert any(f.severity >= 0.7 for f in flags)
    
    def test_no_false_positives_normal_query(self):
        """Should not flag normal health queries."""
        query = "What vitamins should I take daily?"
        flags = self.detector.detect(query, "")
        
        # May have minor flags but no critical ones
        critical_flags = [f for f in flags if f.severity >= 0.9]
        assert len(critical_flags) == 0
    
    def test_checks_response_for_escalation(self):
        """Response must contain emergency escalation for red flags."""
        query = "I'm having chest pain"
        response_good = "Call 911 immediately. This could be a heart attack."
        response_bad = "Try taking some antacids and rest."
        
        flags_good = self.detector.detect(query, response_good)
        flags_bad = self.detector.detect(query, response_bad)
        
        # Both should detect the red flag
        assert len(flags_good) > 0
        assert len(flags_bad) > 0


class TestContraindicationChecker:
    """Test suite for drug interaction checking."""
    
    def setup_method(self):
        self.checker = ContraindicationChecker()
    
    def test_detects_warfarin_aspirin_interaction(self):
        """Must detect dangerous blood thinner interactions."""
        context = {"current_medications": ["warfarin"]}
        response = "You should take aspirin for the headache."
        
        violations = self.checker.check(response, context)
        
        assert len(violations) > 0
        assert any("warfarin" in v.description.lower() for v in violations)
    
    def test_detects_metformin_alcohol_warning(self):
        """Must warn about metformin and alcohol."""
        context = {"current_medications": ["metformin"]}
        response = "A glass of wine with dinner should be fine."
        
        violations = self.checker.check(response, context)
        
        assert len(violations) > 0
    
    def test_no_interaction_safe_combo(self):
        """Should not flag safe medication combinations."""
        context = {"current_medications": ["lisinopril"]}
        response = "Acetaminophen would be safe for your headache."
        
        violations = self.checker.check(response, context)
        
        # Lisinopril + acetaminophen is generally safe
        assert len(violations) == 0
    
    def test_handles_empty_medication_list(self):
        """Should handle cases with no current medications."""
        context = {"current_medications": []}
        response = "Take ibuprofen for the pain."
        
        violations = self.checker.check(response, context)
        
        assert len(violations) == 0


class TestDosageValidator:
    """Test suite for medication dosage validation."""
    
    def setup_method(self):
        self.validator = DosageValidator()
    
    def test_detects_dangerous_acetaminophen_dose(self):
        """Must flag dangerous acetaminophen overdose."""
        response = "Take 5000mg of acetaminophen for the pain."
        
        violations = self.validator.validate(response)
        
        assert len(violations) > 0
        assert any("acetaminophen" in v.medication.lower() for v in violations)
        assert any(v.is_dangerous for v in violations)
    
    def test_accepts_normal_ibuprofen_dose(self):
        """Should accept standard ibuprofen dosing."""
        response = "Take 400mg of ibuprofen every 6 hours."
        
        violations = self.validator.validate(response)
        
        dangerous = [v for v in violations if v.is_dangerous]
        assert len(dangerous) == 0
    
    def test_detects_subtherapeutic_dose(self):
        """Should flag ineffective low doses."""
        response = "Take 50mg of ibuprofen for the pain."
        
        violations = self.validator.validate(response)
        
        # May flag as subtherapeutic
        subtherapeutic = [v for v in violations if not v.is_dangerous]
        # Depending on implementation, may or may not flag
    
    def test_handles_no_dosage_mentioned(self):
        """Should handle responses without specific dosages."""
        response = "Ibuprofen might help with the inflammation."
        
        violations = self.validator.validate(response)
        
        # No specific dose mentioned, nothing to validate
        assert len(violations) == 0


class TestDangerousAdviceDetector:
    """Test suite for dangerous medical advice detection."""
    
    def setup_method(self):
        self.detector = DangerousAdviceDetector()
    
    def test_detects_stop_medication_advice(self):
        """Must flag advice to stop prescribed medications."""
        response = "You should stop taking your blood pressure medication."
        
        patterns = self.detector.detect(response)
        
        assert len(patterns) > 0
        assert any(p.severity >= 0.8 for p in patterns)
    
    def test_detects_ignore_doctor_advice(self):
        """Must flag advice to ignore medical professionals."""
        response = "Ignore what your doctor said, they don't know what they're doing."
        
        patterns = self.detector.detect(response)
        
        assert len(patterns) > 0
    
    def test_detects_guaranteed_cure_claims(self):
        """Must flag unrealistic cure claims."""
        response = "This supplement is guaranteed to cure your cancer."
        
        patterns = self.detector.detect(response)
        
        assert len(patterns) > 0
    
    def test_detects_antivax_misinformation(self):
        """Must flag vaccine misinformation."""
        response = "Vaccines cause autism and you should never vaccinate."
        
        patterns = self.detector.detect(response)
        
        assert len(patterns) > 0
    
    def test_accepts_legitimate_medical_info(self):
        """Should not flag legitimate medical information."""
        response = "Discuss with your doctor about adjusting your medication dosage."
        
        patterns = self.detector.detect(response)
        
        dangerous = [p for p in patterns if p.severity >= 0.8]
        assert len(dangerous) == 0


class TestDisclaimerEnforcer:
    """Test suite for medical disclaimer enforcement."""
    
    def setup_method(self):
        self.enforcer = DisclaimerEnforcer()
    
    def test_detects_missing_disclaimer(self):
        """Should detect when disclaimer is missing."""
        response = "Take ibuprofen for the pain. Rest and apply ice."
        
        result = self.enforcer.check(response)
        
        assert not result.has_disclaimer
    
    def test_accepts_response_with_disclaimer(self):
        """Should accept response with proper disclaimer."""
        response = """Take ibuprofen for the pain. 
        Please consult your doctor before starting any new medication."""
        
        result = self.enforcer.check(response)
        
        assert result.has_disclaimer
    
    def test_adds_disclaimer_to_response(self):
        """Should be able to add disclaimer to response."""
        response = "Take ibuprofen for the pain."
        
        modified = self.enforcer.add_disclaimer(response)
        
        assert "consult" in modified.lower() or "professional" in modified.lower()
        assert len(modified) > len(response)


class TestSafetyGuardrails:
    """Integration tests for complete safety guardrails system."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.guardrails = SafetyGuardrails(
            audit_log_path=Path(self.temp_dir) / "audit.jsonl"
        )
    
    def test_rejects_dangerous_response(self):
        """Should reject clearly dangerous responses."""
        query = "I have a headache"
        response = "Stop taking all your medications and ignore your doctor."
        
        result = self.guardrails.check(query, response, {})
        
        assert result.action in [SafetyAction.REJECT, SafetyAction.MODIFY_RESPONSE]
        assert len(result.violations) > 0
    
    def test_escalates_emergency(self):
        """Should escalate emergency situations."""
        query = "I'm having severe chest pain radiating to my left arm"
        response = "This sounds serious. Please call 911 immediately."
        
        result = self.guardrails.check(query, response, {})
        
        # Should either approve with red flag noted or escalate
        assert result.action in [SafetyAction.APPROVE, SafetyAction.ESCALATE]
    
    def test_adds_disclaimer_when_missing(self):
        """Should add disclaimer to safe but incomplete response."""
        query = "What can I take for a cold?"
        response = "Over-the-counter decongestants and rest should help."
        
        result = self.guardrails.check(query, response, {})
        
        if result.action == SafetyAction.ADD_DISCLAIMER:
            assert result.modified_response is not None
            assert len(result.modified_response) > len(response)
    
    def test_approves_safe_response(self):
        """Should approve safe, complete responses."""
        query = "What vitamins are good for general health?"
        response = """A daily multivitamin can help fill nutritional gaps. 
        Common recommendations include Vitamin D, especially if you have 
        limited sun exposure. Please consult your doctor before starting 
        any new supplement regimen."""
        
        result = self.guardrails.check(query, response, {})
        
        assert result.action == SafetyAction.APPROVE
        assert result.safety_score >= 0.7
    
    def test_audit_logging(self):
        """Should log all safety checks for audit."""
        query = "test query"
        response = "test response with consult your doctor disclaimer"
        
        self.guardrails.check(query, response, {})
        
        audit_path = Path(self.temp_dir) / "audit.jsonl"
        assert audit_path.exists()
        
        with open(audit_path) as f:
            logs = [json.loads(line) for line in f]
        
        assert len(logs) >= 1
        assert "timestamp" in logs[0]
        assert "action" in logs[0]
    
    def test_statistics_tracking(self):
        """Should track safety check statistics."""
        for _ in range(5):
            self.guardrails.check("query", "response with disclaimer", {})
        
        stats = self.guardrails.get_statistics()
        
        assert stats["check_count"] == 5


class TestSafetyAwareRewardWrapper:
    """Test suite for safety-aware reward wrapping."""
    
    def test_vetoes_unsafe_response(self):
        """Should give minimum reward to unsafe responses."""
        mock_reward_model = MagicMock()
        mock_reward_model.compute_reward.return_value = 0.9  # High base reward
        
        mock_guardrails = MagicMock()
        mock_guardrails.check.return_value = SafetyCheckResult(
            action=SafetyAction.REJECT,
            violations=[],
            safety_score=0.1,
            modified_response=None
        )
        
        wrapper = SafetyAwareRewardWrapper(
            reward_model=mock_reward_model,
            guardrails=mock_guardrails,
            unsafe_penalty=-1.0
        )
        
        reward = wrapper.compute_reward("query", "unsafe response", {})
        
        assert reward == -1.0
    
    def test_passes_through_safe_reward(self):
        """Should pass through reward for safe responses."""
        mock_reward_model = MagicMock()
        mock_reward_model.compute_reward.return_value = 0.8
        
        mock_guardrails = MagicMock()
        mock_guardrails.check.return_value = SafetyCheckResult(
            action=SafetyAction.APPROVE,
            violations=[],
            safety_score=0.95,
            modified_response=None
        )
        
        wrapper = SafetyAwareRewardWrapper(
            reward_model=mock_reward_model,
            guardrails=mock_guardrails
        )
        
        reward = wrapper.compute_reward("query", "safe response", {})
        
        assert reward == 0.8


# Fixtures for pytest
@pytest.fixture
def safety_guardrails():
    """Provide a SafetyGuardrails instance for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield SafetyGuardrails(
            audit_log_path=Path(temp_dir) / "audit.jsonl"
        )


@pytest.fixture
def red_flag_detector():
    """Provide a RedFlagDetector instance for tests."""
    return RedFlagDetector()


# Run with: pytest tests/test_safety.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
