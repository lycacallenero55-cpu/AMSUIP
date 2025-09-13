#!/usr/bin/env python3
"""
Test runner for the AI signature verification system
"""
import subprocess
import sys
import os
import argparse
from pathlib import Path


def run_command(command, cwd=None):
    """Run a command and return the result"""
    print(f"Running: {command}")
    if cwd:
        print(f"Working directory: {cwd}")
    
    result = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    
    print(f"Exit code: {result.returncode}")
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    return result.returncode == 0


def run_frontend_tests():
    """Run frontend tests"""
    print("\n" + "="*60)
    print("RUNNING FRONTEND TESTS")
    print("="*60)
    
    frontend_dir = Path(__file__).parent / "src"
    
    # Install dependencies if needed
    if not (frontend_dir / "node_modules").exists():
        print("Installing frontend dependencies...")
        if not run_command("npm install", cwd=frontend_dir):
            print("Failed to install frontend dependencies")
            return False
    
    # Run Jest tests
    print("Running Jest tests...")
    return run_command("npm test -- --coverage --watchAll=false", cwd=frontend_dir)


def run_backend_tests():
    """Run backend tests"""
    print("\n" + "="*60)
    print("RUNNING BACKEND TESTS")
    print("="*60)
    
    backend_dir = Path(__file__).parent / "ai-training"
    
    # Install dependencies if needed
    if not (backend_dir / "venv").exists():
        print("Creating virtual environment...")
        if not run_command("python -m venv venv", cwd=backend_dir):
            print("Failed to create virtual environment")
            return False
    
    # Activate virtual environment and install dependencies
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/MacOS
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    print("Installing backend dependencies...")
    install_cmd = f"{activate_cmd} && {pip_cmd} install -r requirements.txt"
    if not run_command(install_cmd, cwd=backend_dir):
        print("Failed to install backend dependencies")
        return False
    
    # Run pytest
    print("Running pytest...")
    test_cmd = f"{activate_cmd} && python -m pytest tests/ -v --tb=short"
    return run_command(test_cmd, cwd=backend_dir)


def run_integration_tests():
    """Run integration tests"""
    print("\n" + "="*60)
    print("RUNNING INTEGRATION TESTS")
    print("="*60)
    
    backend_dir = Path(__file__).parent / "ai-training"
    
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
    else:  # Unix/Linux/MacOS
        activate_cmd = "source venv/bin/activate"
    
    # Run integration tests
    print("Running integration tests...")
    test_cmd = f"{activate_cmd} && python -m pytest tests/test_integration.py -v --tb=short"
    return run_command(test_cmd, cwd=backend_dir)


def run_e2e_tests():
    """Run end-to-end tests"""
    print("\n" + "="*60)
    print("RUNNING END-TO-END TESTS")
    print("="*60)
    
    backend_dir = Path(__file__).parent / "ai-training"
    
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
    else:  # Unix/Linux/MacOS
        activate_cmd = "source venv/bin/activate"
    
    # Run E2E tests
    print("Running E2E tests...")
    test_cmd = f"{activate_cmd} && python -m pytest tests/test_e2e.py -v --tb=short"
    return run_command(test_cmd, cwd=backend_dir)


def run_smoke_tests():
    """Run smoke tests to verify basic functionality"""
    print("\n" + "="*60)
    print("RUNNING SMOKE TESTS")
    print("="*60)
    
    backend_dir = Path(__file__).parent / "ai-training"
    
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
    else:  # Unix/Linux/MacOS
        activate_cmd = "source venv/bin/activate"
    
    # Run smoke tests
    print("Running smoke tests...")
    smoke_test_cmd = f"{activate_cmd} && python simple_test.py"
    return run_command(smoke_test_cmd, cwd=backend_dir)


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Run AI signature verification system tests")
    parser.add_argument("--frontend", action="store_true", help="Run frontend tests only")
    parser.add_argument("--backend", action="store_true", help="Run backend tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end tests only")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests only")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if not any([args.frontend, args.backend, args.integration, args.e2e, args.smoke, args.all]):
        args.all = True  # Default to running all tests
    
    print("AI Signature Verification System - Test Runner")
    print("=" * 60)
    
    results = []
    
    if args.frontend or args.all:
        results.append(("Frontend Tests", run_frontend_tests()))
    
    if args.backend or args.all:
        results.append(("Backend Tests", run_backend_tests()))
    
    if args.integration or args.all:
        results.append(("Integration Tests", run_integration_tests()))
    
    if args.e2e or args.all:
        results.append(("End-to-End Tests", run_e2e_tests()))
    
    if args.smoke or args.all:
        results.append(("Smoke Tests", run_smoke_tests()))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("ALL TESTS PASSED! ✅")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED! ❌")
        sys.exit(1)


if __name__ == "__main__":
    main()