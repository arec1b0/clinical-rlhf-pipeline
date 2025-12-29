"""
Safety Guardrails for Clinical RLHF

Production-grade safety layer with hard constraints, anomaly detection,
and audit logging for medical AI systems.

Author: Dani (MLOps Lead)
"""

from __future__ import annotations

import json
import hashlib
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from pathlib import Path
import re

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class SafetyAction(Enum):
    """Actions taken by safety guardrails."""
    APPROVE = "approve"
    ADD_DISCLAIMER = "add_disclaimer"
    MODIFY_RESPONSE = "modify_response"
    REJECT = "reject"
    ESCALATE = "escalate"


class ViolationType(Enum):
    """Types of safety violations."""
    RED_FLAG_UNADDRESSED = "red_flag_symptom_unaddressed"
    CONTRAINDICATION = "medication_contraindication"
    DOSAGE_ERROR = "dosage_error"
    DANGEROUS_ADVICE = "dangerous_advice"
    MISSING_DISCLAIMER = "missing_disclaimer"
    OVERCONFIDENT = "overconfident_diagnosis"
    SCOPE_VIOLATION = "out_of_scope"


@dataclass
class SafetyViolation:
    """Detailed record of a safety violation."""
    violation_type: ViolationType
    severity: float  # 0.0 to 1.0
    description: str
    evidence: str
    recommended_action: SafetyAction
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.violation_type.value,
            "severity": self.severity,
            "description": self.description,
            "evidence": self.evidence[:200],  # Truncate for logging
            "action": self.recommended_action.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SafetyCheckResult:
    """Result of running all safety checks."""
    is_safe: bool
    action: SafetyAction
    violations: List[SafetyViolation]
    modified_response: Optional[str]
    safety_score: float
    check_duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_audit_log(self) -> Dict[str, Any]:
        """Generate audit log entry."""
        return {
            "is_safe": self.is_safe,
            "action": self.action.value,
            "num_violations": len(self.violations),
            "violations": [v.to_dict() for v in self.violations],
            "safety_score": self.safety_score,
            "check_duration_ms": self.check_duration_ms,
            "timestamp": datetime.utcnow().isoformat(),
        }


class RedFlagDetector:
    """
    Detects medical red flag symptoms requiring immediate attention.
    
    Uses both keyword matching and semantic understanding.
    """
    
    # Red flags categorized by urgency
    CRITICAL_RED_FLAGS = {
        "cardiac": [
            "chest pain", "crushing chest", "radiating arm pain",
            "heart attack", "cardiac arrest"
        ],
        "respiratory": [
            "difficulty breathing", "can't breathe", "severe shortness of breath",
            "choking", "airway obstruction"
        ],
        "neurological": [
            "stroke symptoms", "sudden weakness one side", "facial droop",
            "slurred speech sudden", "worst headache of life"
        ],
        "psychiatric": [
            "suicidal ideation", "want to kill myself", "plan to end life",
            "harm myself", "suicide plan"
        ],
        "anaphylaxis": [
            "severe allergic reaction", "throat closing", "can't swallow",
            "anaphylaxis", "epinephrine needed"
        ]
    }
    
    URGENT_RED_FLAGS = {
        "infection": [
            "high fever", "fever over 103", "sepsis symptoms",
            "severe infection"
        ],
        "bleeding": [
            "uncontrolled bleeding", "vomiting blood", "blood in stool",
            "severe hemorrhage"
        ],
        "trauma": [
            "head injury", "loss of consciousness", "severe trauma",
            "broken bone visible"
        ],
        "obstetric": [
            "severe abdominal pain pregnant", "vaginal bleeding pregnant",
            "preeclampsia symptoms"
        ]
    }
    
    def __init__(self, semantic_model: Optional[Any] = None):
        self.semantic_model = semantic_model
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        self.critical_patterns = {}
        self.urgent_patterns = {}
        
        for category, phrases in self.CRITICAL_RED_FLAGS.items():
            patterns = [re.compile(rf'\b{re.escape(p)}\b', re.IGNORECASE) 
                       for p in phrases]
            self.critical_patterns[category] = patterns
            
        for category, phrases in self.URGENT_RED_FLAGS.items():
            patterns = [re.compile(rf'\b{re.escape(p)}\b', re.IGNORECASE) 
                       for p in phrases]
            self.urgent_patterns[category] = patterns
    
    def detect(self, text: str) -> List[Tuple[str, str, float]]:
        """
        Detect red flags in text.
        
        Returns:
            List of (category, matched_phrase, severity)
        """
        detected = []
        
        # Check critical red flags (severity 1.0)
        for category, patterns in self.critical_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    detected.append((category, pattern.pattern, 1.0))
        
        # Check urgent red flags (severity 0.7)
        for category, patterns in self.urgent_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    detected.append((category, pattern.pattern, 0.7))
        
        return detected


class ContraindicationChecker:
    """
    Checks for medication contraindications.
    
    Uses a knowledge base of drug interactions and contraindications.
    """
    
    def __init__(self, contraindication_db: Optional[Dict[str, List[str]]] = None):
        # Default contraindication database
        self.db = contraindication_db or self._get_default_db()
        
    def _get_default_db(self) -> Dict[str, List[str]]:
        """Default contraindication database for common medications."""
        return {
            "warfarin": ["aspirin", "ibuprofen", "nsaids", "vitamin k"],
            "metformin": ["alcohol", "contrast dye", "iodinated contrast"],
            "lisinopril": ["potassium supplements", "spironolactone", "nsaids"],
            "simvastatin": ["grapefruit", "erythromycin", "clarithromycin"],
            "methotrexate": ["nsaids", "trimethoprim", "live vaccines"],
            "ssri": ["maoi", "tramadol", "triptans"],
            "maoi": ["ssri", "tyramine", "meperidine", "dextromethorphan"],
            "digoxin": ["amiodarone", "verapamil", "quinidine"],
            "lithium": ["nsaids", "ace inhibitors", "diuretics"],
        }
    
    def check(
        self, 
        response: str, 
        current_medications: List[str]
    ) -> List[Tuple[str, str, str]]:
        """
        Check for contraindications between response recommendations and current meds.
        
        Returns:
            List of (current_med, contraindicated_item, reason)
        """
        violations = []
        response_lower = response.lower()
        
        for medication in current_medications:
            med_key = medication.lower()
            
            # Direct lookup
            if med_key in self.db:
                for contraindicated in self.db[med_key]:
                    if contraindicated in response_lower:
                        violations.append((
                            medication,
                            contraindicated,
                            f"{medication} has interaction with {contraindicated}"
                        ))
            
            # Check drug class
            for db_med, contraindications in self.db.items():
                if db_med in med_key or med_key in db_med:
                    for contraindicated in contraindications:
                        if contraindicated in response_lower:
                            violations.append((
                                medication,
                                contraindicated,
                                f"{medication} (similar to {db_med}) may interact with {contraindicated}"
                            ))
        
        return violations


class DosageValidator:
    """
    Validates medication dosages mentioned in responses.
    
    Uses standard dosage ranges from pharmacological references.
    """
    
    # Standard dosage ranges: {medication: {unit: (min_safe, max_safe, typical)}}
    DOSAGE_RANGES = {
        "acetaminophen": {"mg": (325, 4000, 500), "per_day_max": 4000},
        "ibuprofen": {"mg": (200, 3200, 400), "per_day_max": 3200},
        "metformin": {"mg": (500, 2550, 1000), "per_day_max": 2550},
        "lisinopril": {"mg": (2.5, 80, 10), "per_day_max": 80},
        "atorvastatin": {"mg": (10, 80, 20), "per_day_max": 80},
        "omeprazole": {"mg": (10, 40, 20), "per_day_max": 40},
        "metoprolol": {"mg": (25, 400, 50), "per_day_max": 400},
        "amlodipine": {"mg": (2.5, 10, 5), "per_day_max": 10},
        "gabapentin": {"mg": (100, 3600, 300), "per_day_max": 3600},
        "prednisone": {"mg": (1, 80, 20), "per_day_max": 80},
    }
    
    def __init__(self):
        self.dosage_pattern = re.compile(
            r'(\w+)\s+(\d+(?:\.\d+)?)\s*(mg|ml|mcg|g|units?|iu)',
            re.IGNORECASE
        )
    
    def validate(self, response: str) -> List[Tuple[str, float, str, str]]:
        """
        Validate dosages in response.
        
        Returns:
            List of (medication, dose, unit, issue) for problematic dosages
        """
        issues = []
        matches = self.dosage_pattern.findall(response.lower())
        
        for medication, dose_str, unit in matches:
            dose = float(dose_str)
            
            # Check if medication is in our database
            for known_med, ranges in self.DOSAGE_RANGES.items():
                if known_med in medication or medication in known_med:
                    if unit.lower() in ['mg', 'ml']:
                        min_safe, max_safe, typical = ranges.get('mg', (0, float('inf'), 0))
                        
                        if dose < min_safe:
                            issues.append((
                                medication, dose, unit,
                                f"Dose {dose}{unit} below typical minimum ({min_safe}{unit})"
                            ))
                        elif dose > max_safe:
                            issues.append((
                                medication, dose, unit,
                                f"DANGER: Dose {dose}{unit} exceeds maximum safe dose ({max_safe}{unit})"
                            ))
        
        return issues


class DangerousAdviceDetector:
    """
    Detects potentially dangerous medical advice patterns.
    """
    
    DANGEROUS_PATTERNS = [
        (r"stop\s+taking\s+(?:your\s+)?(?:prescribed\s+)?medication", 0.9),
        (r"ignore\s+(?:your\s+)?doctor", 1.0),
        (r"no\s+need\s+(?:to\s+)?see\s+(?:a\s+)?doctor", 0.8),
        (r"(?:this|it)\s+will\s+(?:definitely|certainly)\s+cure", 0.7),
        (r"guaranteed\s+to\s+(?:work|cure|heal)", 0.8),
        (r"instead\s+of\s+(?:prescribed\s+)?(?:your\s+)?medication", 0.7),
        (r"don't\s+(?:go\s+to|visit)\s+(?:the\s+)?(?:hospital|doctor|er)", 0.9),
        (r"natural\s+(?:cure|remedy)\s+(?:is\s+)?better\s+than\s+medicine", 0.6),
        (r"vaccines?\s+(?:cause|caused|causes)\s+(?:autism|harm)", 1.0),
        (r"(?:bleach|chlorine\s+dioxide)\s+(?:can|will)\s+(?:cure|treat)", 1.0),
    ]
    
    def __init__(self):
        self.patterns = [
            (re.compile(pattern, re.IGNORECASE), severity)
            for pattern, severity in self.DANGEROUS_PATTERNS
        ]
    
    def detect(self, response: str) -> List[Tuple[str, float]]:
        """
        Detect dangerous advice patterns.
        
        Returns:
            List of (matched_pattern, severity)
        """
        detected = []
        
        for pattern, severity in self.patterns:
            match = pattern.search(response)
            if match:
                detected.append((match.group(), severity))
        
        return detected


class DisclaimerEnforcer:
    """
    Ensures appropriate medical disclaimers are present.
    """
    
    REQUIRED_DISCLAIMERS = [
        "consult your doctor",
        "consult a healthcare",
        "speak with your physician",
        "professional medical advice",
        "not a substitute for",
        "seek medical attention",
        "talk to your doctor",
        "medical professional",
        "healthcare provider"
    ]
    
    STANDARD_DISCLAIMER = (
        "\n\n*This information is for educational purposes only and is not "
        "a substitute for professional medical advice. Please consult with "
        "a qualified healthcare provider for personalized medical guidance.*"
    )
    
    def check(self, response: str) -> bool:
        """Check if response has appropriate disclaimer."""
        response_lower = response.lower()
        return any(
            disclaimer in response_lower 
            for disclaimer in self.REQUIRED_DISCLAIMERS
        )
    
    def add_disclaimer(self, response: str) -> str:
        """Add standard disclaimer to response."""
        if self.check(response):
            return response
        return response + self.STANDARD_DISCLAIMER


class SafetyGuardrails:
    """
    Main safety guardrails orchestrator.
    
    Combines all safety checks with configurable thresholds and actions.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        audit_log_path: Optional[Path] = None
    ):
        self.config = config or {}
        self.audit_log_path = audit_log_path
        
        # Initialize components
        self.red_flag_detector = RedFlagDetector()
        self.contraindication_checker = ContraindicationChecker()
        self.dosage_validator = DosageValidator()
        self.dangerous_advice_detector = DangerousAdviceDetector()
        self.disclaimer_enforcer = DisclaimerEnforcer()
        
        # Thresholds
        self.hard_reject_threshold = self.config.get("hard_safety_threshold", 0.3)
        self.soft_threshold = self.config.get("soft_safety_threshold", 0.6)
        
        # Counters for monitoring
        self.check_count = 0
        self.violation_count = 0
        self.reject_count = 0
        
    def check(
        self,
        query: str,
        response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SafetyCheckResult:
        """
        Run all safety checks on a query-response pair.
        
        Args:
            query: User's medical query
            response: Model's response
            context: Additional context (patient info, medications, etc.)
            
        Returns:
            SafetyCheckResult with action and violations
        """
        import time
        start_time = time.time()
        
        violations = []
        context = context or {}
        
        # 1. Red flag detection
        red_flags_in_query = self.red_flag_detector.detect(query)
        if red_flags_in_query:
            # Check if response appropriately addresses red flags
            escalation_phrases = [
                "emergency", "911", "urgent care", "hospital",
                "immediate medical attention", "seek help immediately"
            ]
            has_escalation = any(
                phrase in response.lower() 
                for phrase in escalation_phrases
            )
            
            if not has_escalation:
                for category, pattern, severity in red_flags_in_query:
                    violations.append(SafetyViolation(
                        violation_type=ViolationType.RED_FLAG_UNADDRESSED,
                        severity=severity,
                        description=f"Red flag '{category}' not properly addressed",
                        evidence=pattern,
                        recommended_action=SafetyAction.REJECT
                    ))
        
        # 2. Contraindication check
        if "patient_medications" in context:
            contraindications = self.contraindication_checker.check(
                response, 
                context["patient_medications"]
            )
            for current_med, contra, reason in contraindications:
                violations.append(SafetyViolation(
                    violation_type=ViolationType.CONTRAINDICATION,
                    severity=0.8,
                    description=reason,
                    evidence=f"{current_med} + {contra}",
                    recommended_action=SafetyAction.REJECT
                ))
        
        # 3. Dosage validation
        dosage_issues = self.dosage_validator.validate(response)
        for med, dose, unit, issue in dosage_issues:
            severity = 1.0 if "DANGER" in issue else 0.6
            violations.append(SafetyViolation(
                violation_type=ViolationType.DOSAGE_ERROR,
                severity=severity,
                description=issue,
                evidence=f"{med} {dose}{unit}",
                recommended_action=SafetyAction.REJECT if severity > 0.8 else SafetyAction.MODIFY_RESPONSE
            ))
        
        # 4. Dangerous advice detection
        dangerous_advice = self.dangerous_advice_detector.detect(response)
        for pattern, severity in dangerous_advice:
            violations.append(SafetyViolation(
                violation_type=ViolationType.DANGEROUS_ADVICE,
                severity=severity,
                description="Potentially dangerous advice detected",
                evidence=pattern,
                recommended_action=SafetyAction.REJECT
            ))
        
        # 5. Disclaimer check
        has_disclaimer = self.disclaimer_enforcer.check(response)
        if not has_disclaimer and len(response.split()) > 50:
            violations.append(SafetyViolation(
                violation_type=ViolationType.MISSING_DISCLAIMER,
                severity=0.3,
                description="Medical response lacks appropriate disclaimer",
                evidence="No disclaimer found",
                recommended_action=SafetyAction.ADD_DISCLAIMER
            ))
        
        # Calculate safety score
        if violations:
            max_severity = max(v.severity for v in violations)
            avg_severity = sum(v.severity for v in violations) / len(violations)
            safety_score = 1.0 - (0.7 * max_severity + 0.3 * avg_severity)
        else:
            safety_score = 1.0
        
        # Determine action
        if safety_score < self.hard_reject_threshold:
            action = SafetyAction.REJECT
            modified_response = None
            is_safe = False
        elif safety_score < self.soft_threshold:
            action = SafetyAction.ADD_DISCLAIMER
            modified_response = self.disclaimer_enforcer.add_disclaimer(response)
            is_safe = True
        else:
            action = SafetyAction.APPROVE
            modified_response = response
            is_safe = True
        
        # Update counters
        self.check_count += 1
        self.violation_count += len(violations)
        if action == SafetyAction.REJECT:
            self.reject_count += 1
        
        duration_ms = (time.time() - start_time) * 1000
        
        result = SafetyCheckResult(
            is_safe=is_safe,
            action=action,
            violations=violations,
            modified_response=modified_response,
            safety_score=safety_score,
            check_duration_ms=duration_ms,
            metadata={
                "query_hash": hashlib.md5(query.encode()).hexdigest()[:8],
                "response_length": len(response),
                "has_disclaimer": has_disclaimer,
            }
        )
        
        # Audit logging
        if self.audit_log_path and violations:
            self._write_audit_log(result)
        
        return result
    
    def _write_audit_log(self, result: SafetyCheckResult):
        """Write safety check result to audit log."""
        try:
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(result.to_audit_log()) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get safety check statistics for monitoring."""
        return {
            "total_checks": self.check_count,
            "total_violations": self.violation_count,
            "total_rejects": self.reject_count,
            "violation_rate": self.violation_count / max(self.check_count, 1),
            "reject_rate": self.reject_count / max(self.check_count, 1),
        }


class SafetyAwareRewardWrapper:
    """
    Wrapper that applies safety guardrails to any reward model.
    
    Provides safety veto power over reward signals.
    """
    
    def __init__(
        self,
        reward_model: Any,
        guardrails: SafetyGuardrails,
        safety_weight: float = 0.5
    ):
        self.reward_model = reward_model
        self.guardrails = guardrails
        self.safety_weight = safety_weight
        
    def compute_reward(
        self,
        query: str,
        response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute safety-modulated reward.
        
        Safety has veto power: unsafe responses get negative reward regardless
        of other reward components.
        """
        # Run safety checks first
        safety_result = self.guardrails.check(query, response, context)
        
        # If unsafe, return large negative reward
        if not safety_result.is_safe:
            return -1.0, {
                "safety_vetoed": True,
                "safety_score": safety_result.safety_score,
                "violations": [v.to_dict() for v in safety_result.violations],
            }
        
        # Otherwise, compute reward model score
        reward_breakdown = self.reward_model(query, response, context)
        
        # Blend base reward with safety score
        base_reward = reward_breakdown.total_reward
        final_reward = (
            (1 - self.safety_weight) * base_reward +
            self.safety_weight * safety_result.safety_score
        )
        
        return final_reward, {
            "safety_vetoed": False,
            "safety_score": safety_result.safety_score,
            "base_reward": base_reward,
            "final_reward": final_reward,
            **reward_breakdown.to_dict()
        }
