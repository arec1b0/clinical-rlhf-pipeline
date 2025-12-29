# Clinical RLHF Pipeline

Production-grade Reinforcement Learning from Human Feedback (RLHF) pipeline for medical AI systems with multi-objective reward design, safety guardrails, and comprehensive monitoring.

## Overview

This pipeline implements a safety-first RLHF training system specifically designed for clinical/medical AI applications where patient safety is paramount.

### Key Features

- **Multi-Objective Reward Model**: Four components (uncertainty, guideline adherence, safety, coherence) with configurable weights
- **Safety Guardrails**: Hard constraints with red flag detection, contraindication checking, and dosage validation
- **Conservative PPO Training**: Low KL divergence, gradient clipping, automatic safety-based early stopping
- **Expert Feedback Collection**: Inter-annotator agreement metrics, quality-weighted consensus
- **Production Monitoring**: Drift detection (PSI, KS, CUSUM), Prometheus metrics, audit logging

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Clinical RLHF Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Expert    │───▶│   Reward    │───▶│    PPO      │         │
│  │  Feedback   │    │   Model     │    │   Trainer   │         │
│  └─────────────┘    └──────┬──────┘    └──────┬──────┘         │
│                            │                   │                │
│                     ┌──────▼──────┐     ┌──────▼──────┐        │
│                     │   Safety    │     │   Monitor   │         │
│                     │ Guardrails  │     │   & Drift   │         │
│                     └─────────────┘     └─────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone repository
git clone https://github.com/arec1b0/clinical-rlhf-pipeline.git
cd clinical-rlhf-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### Training

```bash
python main.py train --config config/clinical_rlhf_config.yaml
```

### Evaluation

```bash
python main.py evaluate \
    --config config/clinical_rlhf_config.yaml \
    --checkpoint checkpoints/best
```

### Safety Demo

```bash
python main.py safety-demo --config config/clinical_rlhf_config.yaml
```

## Configuration

Key configuration parameters in `config/clinical_rlhf_config.yaml`:

```yaml
reward:
  weights:
    uncertainty_quantification: 0.25
    guideline_adherence: 0.30
    patient_safety: 0.35      # Highest weight
    response_coherence: 0.10

safety:
  hard_safety_threshold: 0.3   # Below = reject
  soft_safety_threshold: 0.6   # Below = add disclaimer
  max_unsafe_ratio: 0.05       # Training stops if exceeded

ppo:
  learning_rate: 1.41e-5       # Conservative
  clip_range: 0.2
  target_kl: 0.01              # Very conservative for medical
  entropy_coef: 0.01           # Low for deterministic advice
```

## Multi-Objective Reward Model

Four reward components with weighted aggregation:

| Component | Weight | Purpose |
|-----------|--------|---------|
| **Safety** | 0.35 | Red flag detection, contraindications, dosage validation |
| **Guideline Adherence** | 0.30 | Semantic similarity to CDC/WHO/NICE guidelines |
| **Uncertainty** | 0.25 | Appropriate hedging, calibration-aware scoring |
| **Coherence** | 0.10 | Length, relevance, structure quality |

**Safety Veto**: If safety score < threshold, total reward is capped regardless of other components.

## Safety Guardrails

### Red Flag Detection

Critical symptoms that require emergency escalation:
- Cardiac: chest pain, arm numbness, shortness of breath
- Neurological: stroke symptoms, severe headache, confusion
- Mental health: suicidal ideation, self-harm
- Allergic: anaphylaxis, throat swelling

### Contraindication Checking

Database of drug interactions:
- Warfarin + Aspirin/NSAIDs
- Metformin + Alcohol
- MAOIs + Tyramine-containing foods
- And more...

### Dosage Validation

Standard ranges for common medications:
- Acetaminophen: 325-4000 mg/day
- Ibuprofen: 200-3200 mg/day
- Metformin: 500-2550 mg/day

### Dangerous Advice Detection

Patterns flagged:
- "Stop taking your medication"
- "Ignore what your doctor said"
- "Guaranteed to cure"
- Anti-vaccine misinformation

## Monitoring & Drift Detection

### Statistical Methods

| Method | Use Case | Threshold |
|--------|----------|-----------|
| **PSI** | Distribution shift | > 0.25 = significant |
| **KS Test** | Distribution comparison | p < 0.05 |
| **Mann-Whitney U** | Non-parametric comparison | p < 0.05 |
| **CUSUM** | Mean shift detection | Control chart limits |

### Prometheus Metrics

```
clinical_rlhf_reward_mean
clinical_rlhf_safety_mean
clinical_rlhf_unsafe_rate
clinical_rlhf_requests_total
clinical_rlhf_latency_seconds
clinical_rlhf_drift_detected_total
```

### Alerts

- **INFO**: Minor distribution changes
- **WARNING**: Moderate drift detected
- **CRITICAL**: Safety degradation or significant drift

## Expert Feedback Collection

### Annotation Types

- **Pairwise**: A vs B comparison
- **Rating**: 1-5 scale
- **Ranking**: Order multiple responses
- **Binary**: Accept/Reject

### Inter-Annotator Agreement

- Cohen's Kappa (two annotators)
- Fleiss' Kappa (multiple annotators)
- Krippendorff's Alpha (handles missing values)

### Quality Control

- Minimum 3 annotations per sample
- Agreement threshold: 0.7
- Expert quality tracking based on consensus agreement

## Project Structure

```
clinical-rlhf-pipeline/
├── config/
│   └── clinical_rlhf_config.yaml
├── src/
│   ├── __init__.py
│   ├── reward_models/
│   │   ├── __init__.py
│   │   └── multi_objective_reward.py
│   ├── safety/
│   │   ├── __init__.py
│   │   └── guardrails.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── ppo_trainer.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── expert_feedback.py
│   └── evaluation/
│       ├── __init__.py
│       └── monitoring.py
├── tests/
│   ├── __init__.py
│   ├── test_safety.py
│   ├── test_reward_models.py
│   └── test_monitoring.py
├── main.py
├── requirements.txt
└── README.md
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_safety.py -v
```

## MLflow Integration

Training automatically logs to MLflow:

```bash
# Start MLflow UI
mlflow ui --port 5000

# View experiments at http://localhost:5000
```

Tracked metrics:
- Reward components over time
- Safety scores
- KL divergence
- Policy/value losses
- Learning rate schedule

## Checkpointing

Automatic checkpoints saved:
- Every N steps (configurable)
- On early stopping
- On safety violation (emergency checkpoint)
- On training completion

Checkpoint contents:
- Policy model weights
- Value model weights
- Optimizer states
- Training state (step, epoch, metrics)

## Regulatory Considerations

This pipeline includes features for regulatory compliance:

1. **Audit Logging**: All safety checks logged to JSONL
2. **Reproducibility**: Full config + checkpoint + data versioning
3. **Explainability**: Reward breakdown with component scores
4. **Safety Documentation**: Clear thresholds and constraints

## License

Apache License 2.0
## Citation

```bibtex
@software{clinical_rlhf_pipeline,
  title = {Clinical RLHF Pipeline},
  author = {Daniil Krizhanovskiy},
  year = {2025},
  url = {https://github.com/arec1b0/clinical-rlhf-pipeline}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Acknowledgments

- Anthropic's Constitutional AI paper
- InstructGPT and RLHF research
- Medical AI safety literature
