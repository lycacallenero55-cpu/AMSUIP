#!/usr/bin/env python3
"""
Test runner for AI training system
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_tests(test_type="all", verbose=False, coverage=False):
    """Run tests with specified options"""
    
    # Change to the ai_training directory
    os.chdir(Path(__file__).parent)
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test directory
    cmd.append("tests/")
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Add coverage
    if coverage:
        cmd.append("--cov=ai_training")
        cmd.append("--cov-report=html")
        cmd.append("--cov-report=term")
    
    # Filter by test type
    if test_type == "unit":
        cmd.append("tests/test_s3_supabase_sync.py")
    elif test_type == "integration":
        cmd.append("tests/test_training_integration.py")
    elif test_type == "e2e":
        cmd.append("tests/test_e2e_workflow.py")
    elif test_type == "all":
        pass  # Run all tests
    else:
        print(f"Unknown test type: {test_type}")
        return False
    
    # Run tests
    print(f"Running {test_type} tests...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("❌ pytest not found. Please install it with: pip install pytest")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run AI training system tests")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "e2e", "all"], 
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage", "-c", 
        action="store_true",
        help="Generate coverage report"
    )
    
    args = parser.parse_args()
    
    success = run_tests(
        test_type=args.type,
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()