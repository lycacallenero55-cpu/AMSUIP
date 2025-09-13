#!/usr/bin/env python3
"""
Test Runner for Signature AI System
Runs all tests in sequence
"""

import asyncio
import subprocess
import sys
from pathlib import Path

def run_test(test_name: str, test_script: str) -> bool:
    """Run a single test script"""
    print(f"\n{'='*60}")
    print(f"🧪 Running {test_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([
            sys.executable, test_script
        ], capture_output=True, text=True, timeout=300)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {test_name} PASSED")
            return True
        else:
            print(f"❌ {test_name} FAILED (exit code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_name} TIMED OUT")
        return False
    except Exception as e:
        print(f"💥 {test_name} ERROR: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Signature AI Test Suite")
    print("=" * 60)
    
    tests = [
        ("Smoke Test", "test_smoke.py"),
        ("S3-Supabase Sync Test", "test_sync.py"),
        ("End-to-End Test", "test_end_to_end.py"),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_script in tests:
        if run_test(test_name, test_script):
            passed += 1
    
    print(f"\n{'='*60}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    print(f"{'='*60}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()