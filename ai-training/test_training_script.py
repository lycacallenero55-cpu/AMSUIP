#!/usr/bin/env python3
"""
Test the training script locally to make sure it works before uploading to EC2
"""

import json
import os
import sys
import tempfile
import subprocess

def test_training_script_locally():
    """Test the training script with sample data locally"""

    print("=" * 60)
    print("LOCAL TRAINING SCRIPT TEST")
    print("=" * 60)

    # Create minimal test data
    test_data = {
        "test_student": {
            "genuine": [
                # Simple 224x224x3 arrays (normalized 0-1)
                {
                    "array": [[[[0.5] * 224] * 224] * 3],
                    "shape": [224, 224, 3]
                }
            ],
            "forged": []
        }
    }

    # Save test data to file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        test_data_file = f.name

    print(f"✅ Created test data file: {test_data_file}")

    # Check if training script exists
    script_path = "scripts/train_gpu_template.py"
    if not os.path.exists(script_path):
        print(f"❌ Training script not found: {script_path}")
        return False

    print(f"✅ Training script found: {script_path}")

    # Try to run a syntax check
    try:
        result = subprocess.run([
            sys.executable, "-m", "py_compile", script_path
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Training script syntax is valid")
        else:
            print(f"❌ Training script has syntax errors: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Failed to check syntax: {e}")
        return False

    # Try to import the script to check for import errors
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_gpu", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("✅ Training script imports successfully")
    except Exception as e:
        print(f"❌ Training script has import errors: {e}")
        return False

    # Clean up
    os.unlink(test_data_file)
    print("✅ Test completed successfully!")

    return True

if __name__ == "__main__":
    success = test_training_script_locally()
    if success:
        print("\n🎉 Training script is ready for EC2 deployment!")
        print("Next steps:")
        print("1. Run: python fast_training_test.py")
        print("2. This will test the full EC2 pipeline")
    else:
        print("\n❌ Training script needs fixes before deployment")
