"""
Production-Ready Docker Training Manager
Handles all training execution inside a Docker container with robust error handling
"""

import subprocess
import json
import logging
import os
import tempfile
import shutil
from typing import Dict, Optional, Any, Tuple
from pathlib import Path
import asyncio
import time
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DockerTrainingManager:
    """
    Production-ready Docker training manager with automatic container management
    """
    
    def __init__(self):
        # Load configuration from environment
        self.container_name = os.getenv("DOCKER_CONTAINER_NAME", "ai-training-container")
        self.container_image = os.getenv("DOCKER_CONTAINER_IMAGE", "tensorflow/tensorflow:2.15.0-gpu")
        
        # Container paths
        self.workspace_path = "/workspace"
        self.script_path = f"{self.workspace_path}/train_gpu_template.py"
        self.temp_dir = "/tmp/ai-models"
        
        # Host paths - using absolute paths from environment
        self.host_models_dir = os.getenv("HOST_MODELS_DIR", "/home/ubuntu/ai-training/models")
        self.host_data_dir = os.getenv("HOST_DATA_DIR", "/home/ubuntu/ai-training/training_data")
        
        # Ensure host directories exist
        self._ensure_directories()
        
        # Container state tracking
        self._container_healthy = False
        self._last_health_check = 0
        self._health_check_interval = 30  # seconds
        
        logger.info(f"Docker Training Manager initialized")
        logger.info(f"Container: {self.container_name}")
        logger.info(f"Models dir: {self.host_models_dir}")
        logger.info(f"Data dir: {self.host_data_dir}")
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        try:
            os.makedirs(self.host_models_dir, exist_ok=True)
            os.makedirs(self.host_data_dir, exist_ok=True)
            logger.info("Host directories verified")
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            raise
    
    def _run_command(self, cmd: list, timeout: int = 30) -> Tuple[int, str, str]:
        """Run a command with timeout and capture output"""
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
            logger.error(f"Command timed out: {' '.join(cmd)}")
            return -1, "", "Command timed out"
        except Exception as e:
            logger.error(f"Command failed: {e}")
            return -1, "", str(e)
    
    def is_container_running(self) -> bool:
        """Check if the Docker container is running"""
        try:
            # Use docker inspect for more reliable status check
            cmd = ["docker", "inspect", "-f", "{{.State.Running}}", self.container_name]
            returncode, stdout, stderr = self._run_command(cmd, timeout=5)
            
            if returncode == 0 and stdout.strip().lower() == "true":
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to check container status: {e}")
            return False
    
    def is_container_healthy(self) -> bool:
        """Check if container is healthy and can execute commands"""
        try:
            # Cache health check results
            current_time = time.time()
            if current_time - self._last_health_check < self._health_check_interval:
                return self._container_healthy
            
            # Perform health check
            if not self.is_container_running():
                self._container_healthy = False
                return False
            
            # Test command execution
            cmd = ["docker", "exec", self.container_name, "python3", "-c", "import tensorflow as tf; print('OK')"]
            returncode, stdout, stderr = self._run_command(cmd, timeout=10)
            
            self._container_healthy = (returncode == 0 and "OK" in stdout)
            self._last_health_check = current_time
            
            if self._container_healthy:
                logger.debug("Container health check passed")
            else:
                logger.warning(f"Container health check failed: {stderr}")
            
            return self._container_healthy
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self._container_healthy = False
            return False
    
    async def ensure_container_ready(self) -> bool:
        """Ensure container is running and ready for training"""
        try:
            # Check if container is already running and healthy
            if self.is_container_healthy():
                logger.info("Container is already running and healthy")
                return True
            
            # Check if container exists but is stopped
            cmd = ["docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"]
            returncode, stdout, stderr = self._run_command(cmd)
            
            if self.container_name in stdout:
                # Container exists, start it
                logger.info(f"Starting existing container: {self.container_name}")
                start_cmd = ["docker", "start", self.container_name]
                returncode, stdout, stderr = self._run_command(start_cmd)
                
                if returncode != 0:
                    logger.error(f"Failed to start container: {stderr}")
                    return False
            else:
                # Container doesn't exist, create it
                logger.info(f"Creating new container: {self.container_name}")
                if not await self._create_container():
                    return False
            
            # Wait for container to be ready
            max_retries = 30
            for i in range(max_retries):
                if self.is_container_healthy():
                    logger.info("Container is ready for training")
                    return True
                
                logger.info(f"Waiting for container to be ready... ({i+1}/{max_retries})")
                await asyncio.sleep(2)
            
            logger.error("Container failed to become ready")
            return False
            
        except Exception as e:
            logger.error(f"Failed to ensure container ready: {e}")
            return False
    
    async def _create_container(self) -> bool:
        """Create a new Docker container with proper configuration"""
        try:
            # Get the directory containing the training script
            script_dir = Path(__file__).parent.parent / "scripts"
            
            create_cmd = [
                "docker", "run", "-d",
                "--name", self.container_name,
                "--gpus", "all",
                "--restart", "unless-stopped",
                "-v", f"{script_dir}:{self.workspace_path}",
                "-v", f"{self.host_models_dir}:/models",
                "-v", f"{self.host_data_dir}:/training_data",
                "-e", "NVIDIA_VISIBLE_DEVICES=all",
                "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                "-e", "TF_FORCE_GPU_ALLOW_GROWTH=true",
                "-e", "TF_CPP_MIN_LOG_LEVEL=2",
                self.container_image,
                "tail", "-f", "/dev/null"
            ]
            
            logger.info(f"Creating container with command: {' '.join(create_cmd)}")
            returncode, stdout, stderr = self._run_command(create_cmd, timeout=60)
            
            if returncode != 0:
                logger.error(f"Failed to create container: {stderr}")
                return False
            
            logger.info(f"Container created successfully: {stdout.strip()}")
            
            # Install dependencies if needed
            await self._install_dependencies()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create container: {e}")
            return False
    
    async def _install_dependencies(self):
        """Install required Python packages in container"""
        try:
            logger.info("Installing Python dependencies in container...")
            
            packages = [
                "pillow==10.1.0",
                "numpy==1.24.3",
                "scipy==1.11.4",
                "scikit-learn==1.3.2",
                "boto3==1.28.85"
            ]
            
            install_cmd = ["docker", "exec", self.container_name, "pip", "install", "--no-cache-dir"] + packages
            returncode, stdout, stderr = self._run_command(install_cmd, timeout=300)
            
            if returncode == 0:
                logger.info("Dependencies installed successfully")
            else:
                logger.warning(f"Some dependencies may have failed to install: {stderr}")
                
        except Exception as e:
            logger.warning(f"Failed to install dependencies: {e}")
    
    async def train_in_container(self, 
                                training_data: Dict,
                                job_id: str,
                                student_id: int,
                                epochs: int = 50) -> Dict:
        """
        Execute training inside the Docker container with full error handling
        """
        try:
            logger.info(f"Starting Docker training for job {job_id}")
            start_time = time.time()
            
            # Import job queue for progress updates
            try:
                from utils.job_queue import job_queue
                has_job_queue = True
            except ImportError:
                logger.warning("Job queue not available, progress updates disabled")
                has_job_queue = False
                job_queue = None
            
            # Step 1: Ensure container is ready
            if has_job_queue:
                job_queue.update_job_progress(job_id, 5.0, "Preparing Docker container...")
            
            if not await self.ensure_container_ready():
                raise Exception("Failed to prepare Docker container")
            
            # Step 2: Prepare training data
            if has_job_queue:
                job_queue.update_job_progress(job_id, 10.0, "Preparing training data...")
            
            data_file = await self._prepare_training_data(training_data, job_id)
            
            # Step 3: Copy data to container
            if has_job_queue:
                job_queue.update_job_progress(job_id, 15.0, "Transferring data to container...")
            
            container_data_path = f"/training_data/training_data_{job_id}.json"
            
            # Step 4: Execute training script in container
            if has_job_queue:
                job_queue.update_job_progress(job_id, 20.0, "Starting training process...")
            
            # Build the training command
            training_command = [
                "docker", "exec", "-i", self.container_name,
                "python3", self.script_path,
                "--data", container_data_path,
                "--job_id", job_id,
                "--student_id", str(student_id),
                "--epochs", str(epochs),
                "--output_dir", "/models",
                "--s3_bucket", os.getenv("S3_BUCKET", "signatureai-uploads")
            ]
            
            # Execute training with real-time progress monitoring
            success, training_output = await self._execute_training_with_progress(
                training_command, job_id, job_queue if has_job_queue else None
            )
            
            if not success:
                raise Exception(f"Training failed: {training_output}")
            
            # Step 5: Retrieve results
            if has_job_queue:
                job_queue.update_job_progress(job_id, 85.0, "Retrieving training results...")
            
            results = await self._retrieve_results(job_id)
            
            # Step 6: Cleanup
            if has_job_queue:
                job_queue.update_job_progress(job_id, 95.0, "Cleaning up...")
            
            await self._cleanup_container_files(job_id)
            
            # Calculate training time
            training_time = int(time.time() - start_time)
            
            if has_job_queue:
                job_queue.update_job_progress(job_id, 100.0, f"Training completed in {training_time}s!")
            
            logger.info(f"Training completed successfully for job {job_id} in {training_time}s")
            
            return {
                'success': True,
                'job_id': job_id,
                'model_paths': results.get('model_paths', {}),
                'accuracy': results.get('accuracy'),
                'training_metrics': results.get('training_metrics', {}),
                'training_time': training_time,
                'container_used': self.container_name
            }
            
        except Exception as e:
            logger.error(f"Docker training failed for job {job_id}: {e}")
            logger.error(traceback.format_exc())
            
            # Update job status to failed
            try:
                if has_job_queue and job_queue:
                    job_queue.update_job_progress(job_id, 0.0, f"Training failed: {str(e)}")
            except:
                pass
            
            return {
                'success': False,
                'job_id': job_id,
                'error': str(e),
                'error_details': traceback.format_exc()
            }
    
    async def _prepare_training_data(self, training_data: Dict, job_id: str) -> str:
        """Prepare and save training data with proper serialization"""
        try:
            # Convert images to serializable format
            serializable_data = {}
            
            for student_key, signatures in training_data.items():
                serializable_data[student_key] = {}
                
                # Handle different data structures
                if isinstance(signatures, dict):
                    for key in ['genuine', 'forged']:
                        if key in signatures:
                            serializable_data[student_key][key] = []
                            
                            for img in signatures.get(key, []):
                                # Serialize image data
                                img_data = self._serialize_image(img)
                                if img_data:
                                    serializable_data[student_key][key].append(img_data)
                else:
                    # Legacy format - list of images
                    serializable_data[student_key] = {
                        'genuine': [self._serialize_image(img) for img in signatures if self._serialize_image(img)],
                        'forged': []
                    }
            
            # Save to file in container-accessible location
            data_file = os.path.join(self.host_data_dir, f"training_data_{job_id}.json")
            
            with open(data_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            logger.info(f"Saved training data to {data_file}")
            logger.info(f"Data contains {len(serializable_data)} students")
            
            return data_file
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            raise
    
    def _serialize_image(self, img: Any) -> Optional[Dict]:
        """Serialize image data to JSON-compatible format"""
        try:
            if hasattr(img, 'tolist'):
                # NumPy array
                return {
                    "type": "array",
                    "data": img.tolist(),
                    "shape": list(img.shape) if hasattr(img, 'shape') else None,
                    "dtype": str(img.dtype) if hasattr(img, 'dtype') else "float32"
                }
            elif isinstance(img, str):
                # Base64 string or file path
                if img.startswith('data:') or len(img) > 1000:
                    # Likely base64
                    return {"type": "base64", "data": img}
                else:
                    # Likely file path
                    return {"type": "path", "data": img}
            elif isinstance(img, (list, tuple)):
                # Already a list
                return {"type": "list", "data": list(img)}
            else:
                # Unknown type
                logger.warning(f"Unknown image type: {type(img)}")
                return None
        except Exception as e:
            logger.error(f"Failed to serialize image: {e}")
            return None
    
    async def _execute_training_with_progress(self, 
                                             command: list, 
                                             job_id: str, 
                                             job_queue: Optional[Any]) -> Tuple[bool, str]:
        """Execute training command with real-time progress monitoring"""
        try:
            logger.info(f"Executing training command: {' '.join(command[:5])}...")
            
            # Start the process
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Collect output
            output_lines = []
            error_lines = []
            last_progress = 20.0
            
            async def read_stream(stream, is_error=False):
                nonlocal last_progress
                
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    
                    line_str = line.decode('utf-8', errors='ignore').strip()
                    if line_str:
                        if is_error:
                            error_lines.append(line_str)
                            if "ERROR" in line_str or "Error" in line_str:
                                logger.error(f"Training error: {line_str}")
                        else:
                            output_lines.append(line_str)
                            logger.debug(f"Training output: {line_str}")
                        
                        # Parse progress indicators
                        if job_queue and not is_error:
                            progress_info = self._parse_progress(line_str, last_progress)
                            if progress_info:
                                last_progress = progress_info['progress']
                                job_queue.update_job_progress(
                                    job_id, 
                                    progress_info['progress'], 
                                    progress_info['message']
                                )
            
            # Read both stdout and stderr concurrently
            await asyncio.gather(
                read_stream(process.stdout, False),
                read_stream(process.stderr, True)
            )
            
            # Wait for process to complete
            return_code = await process.wait()
            
            # Combine output
            full_output = '\n'.join(output_lines + error_lines)
            
            if return_code == 0:
                logger.info(f"Training completed successfully with return code 0")
                return True, full_output
            else:
                logger.error(f"Training failed with return code {return_code}")
                return False, full_output
            
        except Exception as e:
            logger.error(f"Failed to execute training: {e}")
            return False, str(e)
    
    def _parse_progress(self, line: str, current_progress: float) -> Optional[Dict]:
        """Parse training output for progress indicators"""
        try:
            # Parse epoch progress
            if "Epoch" in line:
                import re
                epoch_match = re.search(r'Epoch\s+(\d+)/(\d+)', line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    total_epochs = int(epoch_match.group(2))
                    progress = 20.0 + (current_epoch / total_epochs * 60.0)
                    return {
                        'progress': min(progress, 80.0),
                        'message': f"Training epoch {current_epoch}/{total_epochs}"
                    }
            
            # Parse accuracy
            if "accuracy:" in line.lower():
                import re
                acc_match = re.search(r'accuracy[:\s]+([0-9.]+)', line.lower())
                if acc_match:
                    accuracy = float(acc_match.group(1))
                    if accuracy <= 1.0:
                        accuracy *= 100
                    return {
                        'progress': current_progress,
                        'message': f"Current accuracy: {accuracy:.1f}%"
                    }
            
            # Parse specific milestones
            if "TRAINING COMPLETED" in line:
                return {'progress': 80.0, 'message': "Training completed, saving models..."}
            
            if "Saved" in line and "model" in line.lower():
                return {'progress': 82.0, 'message': "Models saved successfully"}
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to parse progress: {e}")
            return None
    
    async def _retrieve_results(self, job_id: str) -> Dict:
        """Retrieve training results from container"""
        try:
            results = {'model_paths': {}}
            
            # Define expected model files
            model_files = [
                ('embedding', f'signature_model_embedding.keras'),
                ('classification', f'signature_model_classification.keras'),
                ('mappings', f'signature_model_mappings.json'),
                ('training_results', f'training_results.json'),
                ('centroids', f'centroids.json')
            ]
            
            # Check for models in the host models directory
            job_models_dir = os.path.join(self.host_models_dir, job_id)
            
            for model_type, filename in model_files:
                model_path = os.path.join(job_models_dir, filename)
                
                if os.path.exists(model_path):
                    results['model_paths'][model_type] = model_path
                    logger.info(f"Found {model_type} model: {model_path}")
                else:
                    # Try alternative location
                    alt_path = os.path.join(self.host_models_dir, f"{job_id}_{filename}")
                    if os.path.exists(alt_path):
                        results['model_paths'][model_type] = alt_path
                        logger.info(f"Found {model_type} model at alternative location: {alt_path}")
                    else:
                        logger.warning(f"{model_type} model not found: {model_path}")
            
            # Try to load training results for metrics
            if 'training_results' in results['model_paths']:
                try:
                    with open(results['model_paths']['training_results'], 'r') as f:
                        training_results = json.load(f)
                        results['accuracy'] = training_results.get('final_accuracy')
                        results['training_metrics'] = training_results.get('training_metadata', {})
                        logger.info(f"Loaded training metrics: accuracy={results['accuracy']}")
                except Exception as e:
                    logger.error(f"Failed to load training results: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve results: {e}")
            return {'model_paths': {}}
    
    async def _cleanup_container_files(self, job_id: str):
        """Clean up temporary files in container"""
        try:
            # Remove training data file
            data_file = f"/training_data/training_data_{job_id}.json"
            cleanup_cmd = ["docker", "exec", self.container_name, "rm", "-f", data_file]
            self._run_command(cleanup_cmd, timeout=10)
            
            # Remove temporary model files if they exist in /tmp
            temp_model_dir = f"/tmp/ai-models/{job_id}_models"
            cleanup_cmd = ["docker", "exec", self.container_name, "rm", "-rf", temp_model_dir]
            self._run_command(cleanup_cmd, timeout=10)
            
            logger.info(f"Cleaned up container files for job {job_id}")
            
        except Exception as e:
            logger.warning(f"Failed to cleanup container files: {e}")
    
    def stop_container(self):
        """Stop the Docker container gracefully"""
        try:
            if self.is_container_running():
                logger.info(f"Stopping container: {self.container_name}")
                stop_cmd = ["docker", "stop", "-t", "10", self.container_name]
                returncode, stdout, stderr = self._run_command(stop_cmd, timeout=15)
                
                if returncode == 0:
                    logger.info(f"Container stopped successfully")
                else:
                    logger.warning(f"Failed to stop container gracefully: {stderr}")
        except Exception as e:
            logger.error(f"Failed to stop container: {e}")
    
    def get_container_logs(self, lines: int = 100) -> str:
        """Get recent container logs for debugging"""
        try:
            cmd = ["docker", "logs", "--tail", str(lines), self.container_name]
            returncode, stdout, stderr = self._run_command(cmd, timeout=10)
            
            if returncode == 0:
                return stdout
            else:
                return f"Failed to get logs: {stderr}"
        except Exception as e:
            return f"Error getting logs: {e}"


# Global instance
docker_training_manager = DockerTrainingManager()
