#!/usr/bin/env python3
"""
Comprehensive test runner for the AI signature verification system
Runs all tests and provides detailed reporting
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def run_test_suite(test_file, test_name):
    """Run a specific test suite and return results"""
    print(f"\n{'='*60}")
    print(f"Running {test_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            test_file, 
            "-v", 
            "--tb=short",
            "--color=yes"
        ], capture_output=True, text=True, cwd="/workspace/ai-training")
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Duration: {duration:.2f} seconds")
        print(f"Exit code: {result.returncode}")
        
        if result.stdout:
            print("\nSTDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        return {
            'name': test_name,
            'file': test_file,
            'success': result.returncode == 0,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Error running {test_name}: {e}")
        return {
            'name': test_name,
            'file': test_file,
            'success': False,
            'duration': duration,
            'error': str(e)
        }

def main():
    """Run all test suites"""
    print("AI Signature Verification System - Test Suite")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define test suites
    test_suites = [
        ("test_s3_supabase_sync.py", "S3-Supabase Sync Unit Tests"),
        ("test_training_integration.py", "Training Pipeline Integration Tests"),
        ("test_e2e_flow.py", "End-to-End Flow Tests"),
        ("test_verification_fix.py", "Verification Fix Tests"),
        ("validate_ai_system.py", "AI System Validation"),
        ("run_ai_tests.py", "AI System Tests")
    ]
    
    results = []
    total_start_time = time.time()
    
    # Run each test suite
    for test_file, test_name in test_suites:
        if os.path.exists(test_file):
            result = run_test_suite(test_file, test_name)
            results.append(result)
        else:
            print(f"\nWarning: Test file {test_file} not found, skipping...")
            results.append({
                'name': test_name,
                'file': test_file,
                'success': False,
                'duration': 0,
                'error': 'File not found'
            })
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    successful_tests = sum(1 for r in results if r['success'])
    total_tests = len(results)
    
    print(f"Total test suites: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
    print(f"Total duration: {total_duration:.2f} seconds")
    
    print(f"\nDetailed Results:")
    print("-" * 40)
    
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{status} {result['name']} ({result['duration']:.2f}s)")
        
        if not result['success'] and 'error' in result:
            print(f"    Error: {result['error']}")
    
    # Check if all tests passed
    if successful_tests == total_tests:
        print(f"\n🎉 All tests passed! The AI signature verification system is working correctly.")
        return 0
    else:
        print(f"\n⚠️  Some tests failed. Please review the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())