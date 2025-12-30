"""
Medical Hallucination Detection Module

Detects factual errors and hallucinations in medical AI responses:
- Entity verification (drugs, conditions, dosages)
- Contradiction detection
- Confidence calibration
- Source grounding checks

Author: Dani (MLOps Lead)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)


class HallucinationType(Enum):
    """Types of medical hallucinations."""
    FABRICATED_DRUG = "fabricated_drug"
    WRONG_DOSAGE = "wrong_dosage"
    FABRICATED_CONDITION = "fabricated_condition"
    CONTRAINDICATION_MISSED = "contraindication_missed"
    INCORRECT_INTERACTION = "incorrect_interaction"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    INTERNAL_CONTRADICTION = "internal_contradiction"
    TEMPORAL_ERROR = "temporal_error"
    OVERCONFIDENT_STATEMENT = "overconfident_statement"


class Severity(Enum):
    """Severity of detected hallucination."""
    LOW = "low"           # Minor inaccuracy, unlikely harmful
    MEDIUM = "medium"     # Potentially misleading
    HIGH = "high"         # Could cause harm if followed
    CRITICAL = "critical" # Immediate danger potential


@dataclass
class HallucinationDetection:
    """A detected hallucination instance."""
    hallucination_type: HallucinationType
    severity: Severity
    description: str
    evidence: str
    span_start: int = 0
    span_end: int = 0
    confidence: float = 1.0
    suggested_correction: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "type": self.hallucination_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "evidence": self.evidence,
            "span": [self.span_start, self.span_end],
            "confidence": self.confidence,
            "suggested_correction": self.suggested_correction,
        }


@dataclass
class HallucinationCheckResult:
    """Result of hallucination checking."""
    has_hallucinations: bool
    detections: List[HallucinationDetection]
    overall_score: float  # 0-1, higher is better (fewer hallucinations)
    check_duration_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "has_hallucinations": self.has_hallucinations,
            "num_detections": len(self.detections),
            "overall_score": self.overall_score,
            "detections": [d.to_dict() for d in self.detections],
            "check_duration_ms": self.check_duration_ms,
        }


class MedicalKnowledgeBase:
    """
    Knowledge base for medical fact checking.
    
    Contains verified medical information for hallucination detection.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data")
        
        # Known valid medications
        self.valid_medications: Set[str] = self._load_medications()
        
        # Known conditions
        self.valid_conditions: Set[str] = self._load_conditions()
        
        # Dosage ranges (drug -> (min_mg, max_mg, unit, frequency))
        self.dosage_ranges: Dict[str, Dict] = self._load_dosages()
        
        # Known drug interactions
        self.drug_interactions: Dict[str, Set[str]] = self._load_interactions()
        
        # Common fabricated/fake drug patterns
        self.fake_drug_patterns = [
            r"\b[A-Z][a-z]+(?:mab|nib|zole|pril|olol|statin|mycin|cillin|azole)\b",
        ]
        
        logger.info(f"Loaded knowledge base: {len(self.valid_medications)} medications, {len(self.valid_conditions)} conditions")
    
    def _load_medications(self) -> Set[str]:
        """Load list of valid medications."""
        meds_path = self.data_dir / "medications.json"
        
        if meds_path.exists():
            with open(meds_path) as f:
                data = json.load(f)
            return set(m.lower() for m in data.get("medications", []))
        
        # Default common medications if no file
        return {
            "acetaminophen", "tylenol", "ibuprofen", "advil", "motrin",
            "aspirin", "naproxen", "aleve", "omeprazole", "prilosec",
            "metformin", "lisinopril", "amlodipine", "metoprolol",
            "atorvastatin", "lipitor", "simvastatin", "zocor",
            "losartan", "gabapentin", "hydrochlorothiazide",
            "sertraline", "zoloft", "fluoxetine", "prozac",
            "amoxicillin", "azithromycin", "ciprofloxacin",
            "prednisone", "albuterol", "insulin", "warfarin",
            "levothyroxine", "synthroid", "pantoprazole",
        }
    
    def _load_conditions(self) -> Set[str]:
        """Load list of valid medical conditions."""
        cond_path = self.data_dir / "conditions.json"
        
        if cond_path.exists():
            with open(cond_path) as f:
                data = json.load(f)
            return set(c.lower() for c in data.get("conditions", []))
        
        # Default common conditions
        return {
            "diabetes", "hypertension", "high blood pressure",
            "heart disease", "coronary artery disease", "heart failure",
            "asthma", "copd", "pneumonia", "bronchitis",
            "depression", "anxiety", "bipolar disorder",
            "arthritis", "osteoporosis", "fibromyalgia",
            "cancer", "stroke", "alzheimer's", "parkinson's",
            "migraine", "epilepsy", "multiple sclerosis",
            "hypothyroidism", "hyperthyroidism",
            "kidney disease", "liver disease", "hepatitis",
            "influenza", "flu", "covid-19", "coronavirus",
        }
    
    def _load_dosages(self) -> Dict[str, Dict]:
        """Load dosage ranges for medications."""
        dosage_path = self.data_dir / "medication_dosages.json"
        
        if dosage_path.exists():
            with open(dosage_path) as f:
                return json.load(f)
        
        # Default dosage ranges
        return {
            "acetaminophen": {"min": 325, "max": 1000, "max_daily": 4000, "unit": "mg"},
            "ibuprofen": {"min": 200, "max": 800, "max_daily": 3200, "unit": "mg"},
            "aspirin": {"min": 81, "max": 650, "max_daily": 4000, "unit": "mg"},
            "metformin": {"min": 500, "max": 2550, "max_daily": 2550, "unit": "mg"},
            "lisinopril": {"min": 2.5, "max": 40, "max_daily": 80, "unit": "mg"},
            "atorvastatin": {"min": 10, "max": 80, "max_daily": 80, "unit": "mg"},
            "metoprolol": {"min": 25, "max": 200, "max_daily": 400, "unit": "mg"},
            "omeprazole": {"min": 10, "max": 40, "max_daily": 40, "unit": "mg"},
            "sertraline": {"min": 25, "max": 200, "max_daily": 200, "unit": "mg"},
            "gabapentin": {"min": 100, "max": 1200, "max_daily": 3600, "unit": "mg"},
        }
    
    def _load_interactions(self) -> Dict[str, Set[str]]:
        """Load known drug interactions."""
        inter_path = self.data_dir / "drug_interactions.json"
        
        if inter_path.exists():
            with open(inter_path) as f:
                data = json.load(f)
            return {k: set(v) for k, v in data.items()}
        
        # Default critical interactions
        return {
            "warfarin": {"aspirin", "ibuprofen", "naproxen", "vitamin k"},
            "metformin": {"alcohol", "contrast dye"},
            "lisinopril": {"potassium", "spironolactone"},
            "ssri": {"maoi", "tramadol", "triptans"},
            "statin": {"grapefruit", "gemfibrozil"},
        }
    
    def is_valid_medication(self, name: str) -> bool:
        """Check if medication name is valid."""
        return name.lower() in self.valid_medications
    
    def is_valid_condition(self, name: str) -> bool:
        """Check if condition name is valid."""
        return name.lower() in self.valid_conditions
    
    def check_dosage(self, drug: str, amount: float, unit: str) -> Tuple[bool, Optional[str]]:
        """
        Check if dosage is within valid range.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        drug_lower = drug.lower()
        
        if drug_lower not in self.dosage_ranges:
            return True, None  # Unknown drug, can't validate
        
        info = self.dosage_ranges[drug_lower]
        
        if unit.lower() != info["unit"].lower():
            return False, f"Unit mismatch: expected {info['unit']}, got {unit}"
        
        if amount < info["min"]:
            return False, f"Dosage too low: {amount}{unit} < minimum {info['min']}{info['unit']}"
        
        if amount > info["max"]:
            return False, f"Dosage too high: {amount}{unit} > maximum {info['max']}{info['unit']}"
        
        return True, None


class HallucinationDetector:
    """
    Detects hallucinations in medical AI responses.
    
    Checks for:
    - Fabricated drug names
    - Invalid dosages
    - Unsupported medical claims
    - Internal contradictions
    - Overconfident statements
    
    Usage:
        detector = HallucinationDetector()
        result = detector.check(query, response)
        
        if result.has_hallucinations:
            for detection in result.detections:
                print(f"Found: {detection.hallucination_type}")
    """
    
    def __init__(
        self,
        knowledge_base: Optional[MedicalKnowledgeBase] = None,
        audit_log_path: Optional[Path] = None,
    ):
        self.kb = knowledge_base or MedicalKnowledgeBase()
        self.audit_log_path = audit_log_path
        
        # Patterns for extracting medical entities
        self.dosage_pattern = re.compile(
            r"(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg|iu|units?)\b",
            re.IGNORECASE
        )
        
        self.drug_mention_pattern = re.compile(
            r"\b([A-Z][a-z]+(?:mab|nib|zole|pril|olol|statin|mycin|cillin|azole|ine|ide|ate|ol)?)\b"
        )
        
        # Overconfidence indicators
        self.overconfidence_phrases = [
            r"\balways\b.*\b(?:take|use|works|effective)\b",
            r"\bnever\b.*\b(?:take|use|cause|happens)\b",
            r"\b100%\s*(?:safe|effective|guaranteed)\b",
            r"\bguaranteed\s+to\b",
            r"\bwill\s+definitely\b",
            r"\babsolutely\s+(?:safe|works)\b",
            r"\bno\s+(?:risk|side\s+effects|danger)\b",
            r"\bcures?\b",
        ]
        
        # Contradiction patterns
        self.contradiction_pairs = [
            (r"is\s+safe", r"is\s+(?:not\s+safe|dangerous|harmful)"),
            (r"should\s+take", r"should\s+(?:not\s+take|avoid)"),
            (r"recommended", r"not\s+recommended"),
            (r"effective", r"(?:in)?effective"),
        ]
        
        # Fabricated drug patterns (suspicious endings)
        self.suspicious_drug_pattern = re.compile(
            r"\b([A-Z][a-z]{4,}(?:umab|tinib|zumab|prazole|vudine|navir))\b"
        )
        
        logger.info("HallucinationDetector initialized")
    
    def check(
        self,
        query: str,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> HallucinationCheckResult:
        """
        Check response for hallucinations.
        
        Args:
            query: User's medical query
            response: AI-generated response
            context: Optional context (patient history, etc.)
            
        Returns:
            HallucinationCheckResult with all detections
        """
        import time
        start_time = time.time()
        
        detections: List[HallucinationDetection] = []
        
        # Check for fabricated drugs
        detections.extend(self._check_fabricated_drugs(response))
        
        # Check dosages
        detections.extend(self._check_dosages(response))
        
        # Check for overconfident statements
        detections.extend(self._check_overconfidence(response))
        
        # Check for internal contradictions
        detections.extend(self._check_contradictions(response))
        
        # Check for unsupported claims
        detections.extend(self._check_unsupported_claims(response))
        
        # Calculate overall score
        # Weight by severity
        severity_weights = {
            Severity.LOW: 0.1,
            Severity.MEDIUM: 0.3,
            Severity.HIGH: 0.6,
            Severity.CRITICAL: 1.0,
        }
        
        total_weight = sum(severity_weights[d.severity] for d in detections)
        overall_score = max(0.0, 1.0 - total_weight * 0.2)  # Each detection reduces score
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        result = HallucinationCheckResult(
            has_hallucinations=len(detections) > 0,
            detections=detections,
            overall_score=overall_score,
            check_duration_ms=elapsed_ms,
        )
        
        # Log if hallucinations found
        if result.has_hallucinations and self.audit_log_path:
            self._log_detection(query, response, result)
        
        return result
    
    def _check_fabricated_drugs(self, response: str) -> List[HallucinationDetection]:
        """Check for fabricated drug names."""
        detections = []
        
        # Find potential drug names
        potential_drugs = self.suspicious_drug_pattern.findall(response)
        
        for drug in potential_drugs:
            if not self.kb.is_valid_medication(drug):
                # Check if it looks like a real drug name but isn't in our database
                if len(drug) > 6:  # Long names are more suspicious
                    detections.append(HallucinationDetection(
                        hallucination_type=HallucinationType.FABRICATED_DRUG,
                        severity=Severity.HIGH,
                        description=f"Potentially fabricated medication name: '{drug}'",
                        evidence=drug,
                        confidence=0.7,
                        suggested_correction="Verify medication name with authoritative source",
                    ))
        
        return detections
    
    def _check_dosages(self, response: str) -> List[HallucinationDetection]:
        """Check for invalid dosages."""
        detections = []
        
        # Find dosage mentions
        dosage_matches = self.dosage_pattern.finditer(response)
        
        for match in dosage_matches:
            amount = float(match.group(1))
            unit = match.group(2)
            
            # Find nearby drug name
            start = max(0, match.start() - 100)
            context = response[start:match.end()]
            
            # Look for drug names in context
            for drug in self.kb.dosage_ranges.keys():
                if drug.lower() in context.lower():
                    is_valid, error = self.kb.check_dosage(drug, amount, unit)
                    
                    if not is_valid:
                        detections.append(HallucinationDetection(
                            hallucination_type=HallucinationType.WRONG_DOSAGE,
                            severity=Severity.CRITICAL,
                            description=f"Invalid dosage for {drug}: {error}",
                            evidence=f"{amount}{unit}",
                            span_start=match.start(),
                            span_end=match.end(),
                            confidence=0.9,
                            suggested_correction=f"Check approved dosage range for {drug}",
                        ))
                    break
        
        return detections
    
    def _check_overconfidence(self, response: str) -> List[HallucinationDetection]:
        """Check for overconfident medical statements."""
        detections = []
        
        for pattern in self.overconfidence_phrases:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            
            for match in matches:
                detections.append(HallucinationDetection(
                    hallucination_type=HallucinationType.OVERCONFIDENT_STATEMENT,
                    severity=Severity.MEDIUM,
                    description="Overconfident medical statement without appropriate uncertainty",
                    evidence=match.group(),
                    span_start=match.start(),
                    span_end=match.end(),
                    confidence=0.8,
                    suggested_correction="Add appropriate hedging language (e.g., 'may', 'typically', 'in many cases')",
                ))
        
        return detections
    
    def _check_contradictions(self, response: str) -> List[HallucinationDetection]:
        """Check for internal contradictions."""
        detections = []
        response_lower = response.lower()
        
        for pattern_a, pattern_b in self.contradiction_pairs:
            matches_a = list(re.finditer(pattern_a, response_lower))
            matches_b = list(re.finditer(pattern_b, response_lower))
            
            if matches_a and matches_b:
                detections.append(HallucinationDetection(
                    hallucination_type=HallucinationType.INTERNAL_CONTRADICTION,
                    severity=Severity.HIGH,
                    description="Response contains contradictory statements",
                    evidence=f"'{matches_a[0].group()}' vs '{matches_b[0].group()}'",
                    confidence=0.85,
                    suggested_correction="Ensure consistency throughout the response",
                ))
        
        return detections
    
    def _check_unsupported_claims(self, response: str) -> List[HallucinationDetection]:
        """Check for unsupported medical claims."""
        detections = []
        
        # Patterns that suggest specific claims that should be verified
        claim_patterns = [
            (r"studies\s+(?:show|prove|demonstrate)", "studies"),
            (r"research\s+(?:shows|proves|indicates)", "research"),
            (r"according\s+to\s+(?:experts|doctors|research)", "expert claim"),
            (r"(\d+)%\s+(?:of\s+)?(?:patients|people|cases)", "statistical claim"),
            (r"FDA\s+(?:approved|recommends)", "FDA claim"),
            (r"clinical\s+trials?\s+(?:show|prove)", "clinical trial claim"),
        ]
        
        for pattern, claim_type in claim_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            
            for match in matches:
                # These claims need sources - flag as potentially unsupported
                detections.append(HallucinationDetection(
                    hallucination_type=HallucinationType.UNSUPPORTED_CLAIM,
                    severity=Severity.LOW,  # Low severity until verified false
                    description=f"Unverified {claim_type} - may need citation",
                    evidence=match.group(),
                    span_start=match.start(),
                    span_end=match.end(),
                    confidence=0.5,  # Lower confidence - may be valid
                    suggested_correction="Consider adding source reference or hedging language",
                ))
        
        return detections
    
    def _log_detection(
        self,
        query: str,
        response: str,
        result: HallucinationCheckResult,
    ):
        """Log detection to audit file."""
        if self.audit_log_path is None:
            return
        
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:200],
            "response": response[:500],
            "result": result.to_dict(),
        }
        
        with open(self.audit_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def get_severity_stats(self, results: List[HallucinationCheckResult]) -> Dict[str, int]:
        """Get statistics by severity level."""
        stats = {s.value: 0 for s in Severity}
        
        for result in results:
            for detection in result.detections:
                stats[detection.severity.value] += 1
        
        return stats


def create_hallucination_detector(config: dict) -> HallucinationDetector:
    """Create detector from config dictionary."""
    halluc_config = config.get("hallucination_detection", {})
    
    data_dir = Path(halluc_config.get("data_dir", "data"))
    kb = MedicalKnowledgeBase(data_dir=data_dir)
    
    return HallucinationDetector(
        knowledge_base=kb,
        audit_log_path=Path(halluc_config.get("audit_log", "logs/hallucination_audit.jsonl")),
    )
