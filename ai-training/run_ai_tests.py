#!/usr/bin/env python3
"""
Comprehensive test runner for AI training pipeline
"""
import sys
import os
import subprocess
import logging

# Add the ai-training directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_tests():
    """Run all AI tests"""
    logger.info("Starting AI training pipeline tests...")
    
    # Test files to run
    test_files = [
        "tests/test_image_processing.py",
        "tests/test_training.py", 
        "tests/test_integration.py"
    ]
    
    results = []
    
    for test_file in test_files:
        if os.path.exists(test_file):
            logger.info(f"Running {test_file}...")
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
                ], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
                
                if result.returncode == 0:
                    logger.info(f"✅ {test_file} passed")
                    results.append((test_file, True, result.stdout))
                else:
                    logger.error(f"❌ {test_file} failed")
                    logger.error(result.stderr)
                    results.append((test_file, False, result.stderr))
            except Exception as e:
                logger.error(f"❌ Error running {test_file}: {e}")
                results.append((test_file, False, str(e)))
        else:
            logger.warning(f"⚠️ Test file {test_file} not found")
    
    # Summary
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Summary: {passed}/{total} test suites passed")
    logger.info(f"{'='*50}")
    
    for test_file, success, output in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{status}: {test_file}")
    
    return passed == total

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)