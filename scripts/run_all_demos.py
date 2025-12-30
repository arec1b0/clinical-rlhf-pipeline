#!/usr/bin/env python
"""
Run all P0 demos for Clinical RLHF Pipeline.

Usage:
    python scripts/run_all_demos.py
"""

import subprocess
import sys
from pathlib import Path

# Ensure we're in the right directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode == 0:
        print(f"\n✓ {description} - SUCCESS")
    else:
        print(f"\n✗ {description} - FAILED (code: {result.returncode})")
    
    return result.returncode == 0


def main():
    """Run all demos."""
    print("="*60)
    print("CLINICAL RLHF PIPELINE - P0 DEMOS")
    print("="*60)
    
    demos = [
        (["python", "main.py", "safety-demo"], "Safety Guardrails Demo"),
        (["python", "main.py", "memory-demo"], "P0.1: Memory Monitoring Demo"),
        (["python", "main.py", "rollback-demo"], "P0.2: Automatic Rollback Demo"),
        (["python", "main.py", "hallucination-demo"], "P0.3: Hallucination Detection Demo"),
    ]
    
    results = []
    
    for cmd, description in demos:
        success = run_command(cmd, description)
        results.append((description, success))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for desc, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {desc}")
    
    print(f"\nTotal: {passed}/{total} demos passed")
    print("="*60)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
