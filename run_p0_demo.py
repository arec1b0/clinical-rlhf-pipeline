#!/usr/bin/env python
"""
P0 Critical Features Demo Runner

Runs all P0 demonstrations to verify production readiness:
- P0.1: Memory Monitoring
- P0.2: Automatic Rollback
- P0.3: Hallucination Detection
- P0.4: Integration Tests

Usage:
    python run_p0_demo.py
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run command and print result."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print(f"\n✓ {description} - PASSED")
    else:
        print(f"\n✗ {description} - FAILED (exit code: {result.returncode})")
    
    return result.returncode == 0


def main():
    print("="*70)
    print("  CLINICAL RLHF PIPELINE - P0 CRITICAL FEATURES DEMO")
    print("="*70)
    
    results = []
    
    # P0.1: Memory Monitoring
    results.append(run_command(
        "python main.py memory-demo --config config/clinical_rlhf_config.yaml",
        "P0.1: Memory Monitoring Demo"
    ))
    
    # P0.2: Automatic Rollback
    results.append(run_command(
        "python main.py rollback-demo --config config/clinical_rlhf_config.yaml",
        "P0.2: Automatic Rollback Demo"
    ))
    
    # P0.3: Hallucination Detection
    results.append(run_command(
        "python main.py hallucination-demo --config config/clinical_rlhf_config.yaml",
        "P0.3: Hallucination Detection Demo"
    ))
    
    # Safety Demo (existing)
    results.append(run_command(
        "python main.py safety-demo --config config/clinical_rlhf_config.yaml",
        "Safety Guardrails Demo"
    ))
    
    # P0.4: Integration Tests (subset)
    print(f"\n{'='*70}")
    print("  P0.4: Integration Tests")
    print(f"{'='*70}\n")
    
    test_result = subprocess.run(
        "python -m pytest tests/test_integration.py -v --tb=short -x",
        shell=True
    )
    results.append(test_result.returncode == 0)
    
    # Summary
    print("\n" + "="*70)
    print("  P0 DEMO SUMMARY")
    print("="*70)
    
    labels = [
        "P0.1: Memory Monitoring",
        "P0.2: Automatic Rollback", 
        "P0.3: Hallucination Detection",
        "Safety Guardrails",
        "P0.4: Integration Tests",
    ]
    
    for label, passed in zip(labels, results):
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {label}: {status}")
    
    passed_count = sum(results)
    total_count = len(results)
    
    print(f"\n  Total: {passed_count}/{total_count} passed")
    print("="*70)
    
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
