#!/usr/bin/env python3
"""
Docker Training Setup Verification Script
Tests all components of the Docker-based training system
"""

import os
import sys
import json
import subprocess
import time
import asyncio
from pathlib import Path
from typing import Dict, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(text: str):
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}{text.center(60)}{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}\n")

def print_success(text: str):
    print(f"{GREEN}✅ {text}{RESET}")

def print_error(text: str):
    print(f"{RED}❌ {text}{RESET}")

def print_warning(text: str):
    print(f"{YELLOW}⚠️  {text}{RESET}")

def print_info(text: str):
    print(f"{BLUE}ℹ️  {text}{RESET}")

def run_command(cmd: list, timeout: int = 30) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)

class DockerSetupVerifier:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.container_name = os.getenv("DOCKER_CONTAINER_NAME", "ai-training-container")
        self.host_models_dir = os.getenv("HOST_MODELS_DIR", "/home/ubuntu/ai-training/models")
        self.host_data_dir = os.getenv("HOST_DATA_DIR", "/home/ubuntu/ai-training/training_data")
        
    def verify_environment(self) -> bool:
        """Verify environment variables and configuration"""
        print_header("Environment Configuration")
        
        # Check .env file
        env_file = Path(".env")
        if not env_file.exists():
            self.errors.append(".env file not found")
            print_error(".env file not found")
            return False
        
        print_success(".env file found")
        
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check critical environment variables
        required_vars = {
            "USE_DOCKER_TRAINING": os.getenv("USE_DOCKER_TRAINING"),
            "DOCKER_CONTAINER_NAME": os.getenv("DOCKER_CONTAINER_NAME"),
            "HOST_MODELS_DIR": os.getenv("HOST_MODELS_DIR"),
            "HOST_DATA_DIR": os.getenv("HOST_DATA_DIR"),
            "S3_BUCKET": os.getenv("S3_BUCKET"),
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY")
        }
        
        all_set = True
        for var, value in required_vars.items():
            if value:
                print_success(f"{var}: {value[:20]}..." if len(str(value)) > 20 else f"{var}: {value}")
            else:
                print_error(f"{var}: Not set")
                self.errors.append(f"Environment variable {var} not set")
                all_set = False
        
        # Check if Docker training is enabled
        if os.getenv("USE_DOCKER_TRAINING", "false").lower() != "true":
            print_warning("USE_DOCKER_TRAINING is not set to 'true'")
            self.warnings.append("Docker training is disabled in configuration")
        
        return all_set
    
    def verify_docker_installation(self) -> bool:
        """Verify Docker is installed and running"""
        print_header("Docker Installation")
        
        # Check Docker command
        returncode, stdout, stderr = run_command(["docker", "--version"])
        if returncode != 0:
            print_error("Docker is not installed or not in PATH")
            self.errors.append("Docker not available")
            return False
        
        docker_version = stdout.strip()
        print_success(f"Docker installed: {docker_version}")
        
        # Check Docker daemon
        returncode, stdout, stderr = run_command(["docker", "info"], timeout=10)
        if returncode != 0:
            print_error("Docker daemon is not running")
            self.errors.append("Docker daemon not running")
            return False
        
        print_success("Docker daemon is running")
        
        # Check GPU support (optional)
        returncode, stdout, stderr = run_command(["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:11.8.0-base-ubuntu22.04", "nvidia-smi"], timeout=30)
        if returncode == 0:
            print_success("GPU support is available")
        else:
            print_warning("GPU support not available (training will use CPU)")
            self.warnings.append("No GPU support detected")
        
        return True
    
    def verify_container(self) -> bool:
        """Verify Docker container exists and is properly configured"""
        print_header("Docker Container Status")
        
        # Check if container exists
        returncode, stdout, stderr = run_command(["docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"])
        
        if self.container_name not in stdout:
            print_error(f"Container '{self.container_name}' does not exist")
            print_info("Run the setup script to create the container")
            self.errors.append("Container not found")
            return False
        
        print_success(f"Container '{self.container_name}' exists")
        
        # Check if container is running
        returncode, stdout, stderr = run_command(["docker", "inspect", "-f", "{{.State.Running}}", self.container_name])
        
        if stdout.strip().lower() != "true":
            print_warning("Container is not running, starting it...")
            returncode, stdout, stderr = run_command(["docker", "start", self.container_name])
            if returncode != 0:
                print_error(f"Failed to start container: {stderr}")
                self.errors.append("Container failed to start")
                return False
            time.sleep(3)
        
        print_success("Container is running")
        
        # Verify Python and TensorFlow
        print_info("Verifying container environment...")
        
        # Check Python
        returncode, stdout, stderr = run_command(["docker", "exec", self.container_name, "python3", "--version"])
        if returncode == 0:
            print_success(f"Python installed: {stdout.strip()}")
        else:
            print_error("Python not available in container")
            self.errors.append("Python not found in container")
            return False
        
        # Check TensorFlow
        returncode, stdout, stderr = run_command([
            "docker", "exec", self.container_name, "python3", "-c",
            "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
        ])
        if returncode == 0:
            print_success(f"TensorFlow installed: {stdout.strip()}")
        else:
            print_error("TensorFlow not available in container")
            self.errors.append("TensorFlow not found in container")
            return False
        
        # Check training script
        returncode, stdout, stderr = run_command([
            "docker", "exec", self.container_name, "test", "-f", "/workspace/train_gpu_template.py"
        ])
        if returncode == 0:
            print_success("Training script found at /workspace/train_gpu_template.py")
        else:
            print_warning("Training script not found, will be copied during training")
        
        return True
    
    def verify_directories(self) -> bool:
        """Verify host directories exist and are accessible"""
        print_header("Directory Structure")
        
        all_good = True
        
        # Check models directory
        models_dir = Path(self.host_models_dir)
        if models_dir.exists():
            print_success(f"Models directory exists: {self.host_models_dir}")
        else:
            print_warning(f"Models directory does not exist, creating: {self.host_models_dir}")
            try:
                models_dir.mkdir(parents=True, exist_ok=True)
                print_success(f"Created models directory")
            except Exception as e:
                print_error(f"Failed to create models directory: {e}")
                self.errors.append("Cannot create models directory")
                all_good = False
        
        # Check data directory
        data_dir = Path(self.host_data_dir)
        if data_dir.exists():
            print_success(f"Data directory exists: {self.host_data_dir}")
        else:
            print_warning(f"Data directory does not exist, creating: {self.host_data_dir}")
            try:
                data_dir.mkdir(parents=True, exist_ok=True)
                print_success(f"Created data directory")
            except Exception as e:
                print_error(f"Failed to create data directory: {e}")
                self.errors.append("Cannot create data directory")
                all_good = False
        
        # Check scripts directory
        scripts_dir = Path("scripts")
        if scripts_dir.exists():
            print_success(f"Scripts directory exists: {scripts_dir.absolute()}")
            
            # Check for training script
            train_script = scripts_dir / "train_gpu_template.py"
            if train_script.exists():
                print_success(f"Training script found: train_gpu_template.py")
            else:
                print_error("Training script not found: train_gpu_template.py")
                self.errors.append("Training script missing")
                all_good = False
        else:
            print_error("Scripts directory not found")
            self.errors.append("Scripts directory missing")
            all_good = False
        
        return all_good
    
    async def test_training_manager(self) -> bool:
        """Test the Docker training manager"""
        print_header("Training Manager Test")
        
        try:
            # Import the training manager
            from utils.docker_training_production import DockerTrainingManager
            
            print_success("Docker training manager imported successfully")
            
            # Create instance
            manager = DockerTrainingManager()
            print_success("Training manager initialized")
            
            # Check container health
            if manager.is_container_healthy():
                print_success("Container health check passed")
            else:
                print_warning("Container health check failed, attempting to fix...")
                if await manager.ensure_container_ready():
                    print_success("Container is now ready")
                else:
                    print_error("Failed to prepare container")
                    self.errors.append("Container not healthy")
                    return False
            
            # Test data preparation
            test_data = {
                "test_student": {
                    "genuine": [[[[0.5] * 224] * 224] * 3],  # Dummy image data
                    "forged": []
                }
            }
            
            try:
                data_file = await manager._prepare_training_data(test_data, "test_job")
                if Path(data_file).exists():
                    print_success(f"Test data prepared: {data_file}")
                    # Clean up test file
                    Path(data_file).unlink()
                else:
                    print_error("Failed to prepare test data")
                    self.errors.append("Data preparation failed")
                    return False
            except Exception as e:
                print_error(f"Data preparation error: {e}")
                self.errors.append(f"Data preparation error: {e}")
                return False
            
            return True
            
        except ImportError as e:
            print_error(f"Failed to import training manager: {e}")
            self.errors.append("Training manager import failed")
            return False
        except Exception as e:
            print_error(f"Training manager test failed: {e}")
            self.errors.append(f"Training manager error: {e}")
            return False
    
    async def run_verification(self) -> bool:
        """Run all verification steps"""
        print_header("DOCKER TRAINING SETUP VERIFICATION")
        print(f"Container Name: {self.container_name}")
        print(f"Models Directory: {self.host_models_dir}")
        print(f"Data Directory: {self.host_data_dir}")
        
        # Run all checks
        checks = [
            ("Environment", self.verify_environment()),
            ("Docker Installation", self.verify_docker_installation()),
            ("Container", self.verify_container()),
            ("Directories", self.verify_directories()),
            ("Training Manager", await self.test_training_manager())
        ]
        
        # Summary
        print_header("Verification Summary")
        
        all_passed = True
        for name, passed in checks:
            if passed:
                print_success(f"{name}: PASSED")
            else:
                print_error(f"{name}: FAILED")
                all_passed = False
        
        # Print errors and warnings
        if self.errors:
            print(f"\n{RED}Errors found:{RESET}")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print(f"\n{YELLOW}Warnings:{RESET}")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        # Final result
        print("")
        if all_passed:
            print_success(f"{BOLD}✅ All checks passed! The system is ready for Docker-based training.{RESET}")
            return True
        else:
            print_error(f"{BOLD}❌ Some checks failed. Please fix the issues above.{RESET}")
            return False

async def main():
    """Main entry point"""
    verifier = DockerSetupVerifier()
    success = await verifier.run_verification()
    
    if success:
        print(f"\n{GREEN}Next steps:{RESET}")
        print("1. Start the FastAPI server: python main.py")
        print("2. Click the 'Train' button in the UI")
        print("3. Training will automatically run in the Docker container")
        sys.exit(0)
    else:
        print(f"\n{RED}Please fix the issues and run this script again.{RESET}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
