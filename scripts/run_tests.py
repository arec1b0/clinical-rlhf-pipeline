#!/usr/bin/env python
"""
Run integration tests for Clinical RLHF Pipeline.

Usage:
    python scripts/run_tests.py           # Run all tests
    python scripts/run_tests.py --fast    # Skip slow tests
    python scripts/run_tests.py --cov     # With coverage
"""

import argparse
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description="Run tests")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--cov", action="store_true", help="Run with coverage")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        cmd.append("-v")
    
    if args.fast:
        cmd.extend(["-m", "not slow"])
    
    if args.cov:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    cmd.append("tests/")
    
    print(f"Running: {' '.join(cmd)}")
    print("="*60)
    
    result = subprocess.run(cmd, cwd=project_root)
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
