"""
Docker-based Training Manager
Executes all training inside a pre-configured Docker container
"""

import subprocess
import json
import logging
import os
import tempfile
import shutil
from typing import Dict, Optional
from pathlib import Path
import asyncio
import time

logger = logging.getLogger(__name__)

class DockerTrainingManager:
    """
    Manages training execution inside a Docker container
    """
    
    def __init__(self):
        # Docker configuration
        self.container_name = os.getenv("DOCKER_CONTAINER_NAME", "ai-training-gpu")
        self.container_image = os.getenv("DOCKER_CONTAINER_IMAGE", "tensorflow/tensorflow:2.15.0-gpu")
        self.workspace_path = "/workspace"
        self.script_path = f"{self.workspace_path}/train_gpu_template.py"
        self.temp_dir = "/tmp/ai-models"
        
        # Host paths
        self.host_models_dir = os.getenv("HOST_MODELS_DIR", "./models")
        self.host_data_dir = os.getenv("HOST_DATA_DIR", "./training_data")
        
        # Ensure host directories exist
        os.makedirs(self.host_models_dir, exist_ok=True)
        os.makedirs(self.host_data_dir, exist_ok=True)
        
    def is_container_running(self) -> bool:
        """Check if the Docker container is running"""
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                check=False
            )
            return self.container_name in result.stdout
        except Exception as e:
            logger.error(f"Failed to check container status: {e}")
            return False
    
    def start_container_if_needed(self) -> bool:
        """Start the container if it's not running"""
        try:
            if not self.is_container_running():
                logger.info(f"Starting Docker container: {self.container_name}")
                
                # Check if container exists but is stopped
                result = subprocess.run(
                    ["docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if self.container_name in result.stdout:
                    # Container exists, just start it
                    subprocess.run(
                        ["docker", "start", self.container_name],
                        check=True
                    )
                    logger.info(f"Started existing container: {self.container_name}")
                else:
                    # Container doesn't exist, create and start it
                    logger.warning(f"Container {self.container_name} not found. Please ensure it's created with all dependencies.")
                    return False
                
                # Wait for container to be ready
                time.sleep(2)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to start container: {e}")
            return False
    
    async def train_in_container(self, 
                                training_data: Dict,
                                job_id: str,
                                student_id: int,
                                epochs: int = 50) -> Dict:
        """
        Execute training inside the Docker container
        """
        try:
            logger.info(f"Starting Docker-based training for job {job_id}")
            
            # Import job queue for progress updates
            from utils.job_queue import job_queue
            
            # Step 1: Ensure container is running
            job_queue.update_job_progress(job_id, 5.0, "Starting Docker container...")
            if not self.start_container_if_needed():
                raise Exception("Failed to start Docker container")
            
            # Step 2: Prepare training data
            job_queue.update_job_progress(job_id, 10.0, "Preparing training data...")
            data_file = self._prepare_training_data(training_data, job_id)
            
            # Step 3: Copy data to container
            job_queue.update_job_progress(job_id, 15.0, "Copying data to container...")
            container_data_path = f"{self.workspace_path}/training_data_{job_id}.json"
            self._copy_to_container(data_file, container_data_path)
            
            # Step 4: Execute training script in container
            job_queue.update_job_progress(job_id, 20.0, "Starting training in container...")
            
            # Build the training command
            training_command = [
                "python3",
                self.script_path,
                container_data_path,
                job_id,
                str(student_id),
                str(epochs)
            ]
            
            # Execute training with real-time progress monitoring
            success = await self._execute_training_with_progress(
                training_command, job_id, job_queue
            )
            
            if not success:
                raise Exception("Training failed in container")
            
            # Step 5: Copy results from container
            job_queue.update_job_progress(job_id, 85.0, "Retrieving training results...")
            results = self._retrieve_results(job_id)
            
            # Step 6: Cleanup
            job_queue.update_job_progress(job_id, 95.0, "Cleaning up...")
            self._cleanup_container_files(job_id)
            
            job_queue.update_job_progress(job_id, 100.0, "Training completed successfully!")
            
            return {
                'success': True,
                'job_id': job_id,
                'model_paths': results.get('model_paths', {}),
                'accuracy': results.get('accuracy'),
                'training_metrics': results.get('training_metrics', {})
            }
            
        except Exception as e:
            logger.error(f"Docker training failed: {e}")
            
            # Update job status to failed
            try:
                from utils.job_queue import job_queue
                job_queue.update_job_progress(job_id, 0.0, f"Training failed: {str(e)}")
            except:
                pass
            
            return {'success': False, 'error': str(e)}
    
    def _prepare_training_data(self, training_data: Dict, job_id: str) -> str:
        """Prepare and save training data to a temporary file"""
        try:
            # Convert images to serializable format
            serializable_data = {}
            for student_name, signatures in training_data.items():
                serializable_data[student_name] = {}
                
                # Handle different data structures
                if isinstance(signatures, dict):
                    for key in ['genuine', 'forged']:
                        if key in signatures:
                            serializable_data[student_name][key] = []
                            for img in signatures[key]:
                                if hasattr(img, 'tolist'):
                                    # NumPy array
                                    serializable_data[student_name][key].append({
                                        "array": img.tolist(),
                                        "shape": list(img.shape)
                                    })
                                elif isinstance(img, str):
                                    # Base64 string
                                    serializable_data[student_name][key].append({
                                        "base64": img
                                    })
                                else:
                                    # Other format
                                    serializable_data[student_name][key].append({
                                        "raw": str(type(img))
                                    })
                else:
                    # Legacy format - list of images
                    serializable_data[student_name] = {
                        'genuine': [],
                        'forged': []
                    }
                    for img in signatures:
                        if hasattr(img, 'tolist'):
                            serializable_data[student_name]['genuine'].append({
                                "array": img.tolist(),
                                "shape": list(img.shape)
                            })
                        elif isinstance(img, str):
                            serializable_data[student_name]['genuine'].append({
                                "base64": img
                            })
            
            # Save to temporary file
            data_file = os.path.join(self.host_data_dir, f"training_data_{job_id}.json")
            with open(data_file, 'w') as f:
                json.dump(serializable_data, f)
            
            logger.info(f"Saved training data to {data_file}")
            return data_file
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            raise
    
    def _copy_to_container(self, host_path: str, container_path: str):
        """Copy file from host to container"""
        try:
            cmd = ["docker", "cp", host_path, f"{self.container_name}:{container_path}"]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Copied {host_path} to container:{container_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to copy to container: {e.stderr.decode()}")
            raise
    
    def _copy_from_container(self, container_path: str, host_path: str):
        """Copy file from container to host"""
        try:
            cmd = ["docker", "cp", f"{self.container_name}:{container_path}", host_path]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Copied container:{container_path} to {host_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to copy from container: {e.stderr.decode()}")
            raise
    
    async def _execute_training_with_progress(self, command: list, job_id: str, job_queue) -> bool:
        """Execute training command in container with progress monitoring"""
        try:
            # Build docker exec command
            docker_cmd = ["docker", "exec", "-i", self.container_name] + command
            
            logger.info(f"Executing in container: {' '.join(command)}")
            
            # Start the process
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Monitor output for progress updates
            start_time = time.time()
            last_progress = 20.0
            
            async def read_output():
                nonlocal last_progress
                
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    
                    line_str = line.decode('utf-8', errors='ignore').strip()
                    if line_str:
                        logger.info(f"Container output: {line_str}")
                        
                        # Parse progress indicators
                        if "Epoch" in line_str:
                            # Extract epoch number
                            try:
                                if "/" in line_str:
                                    parts = line_str.split()
                                    for part in parts:
                                        if "/" in part:
                                            current, total = part.split("/")
                                            current = int(current)
                                            total = int(total)
                                            progress = 20.0 + (current / total * 60.0)
                                            if progress > last_progress:
                                                last_progress = progress
                                                job_queue.update_job_progress(
                                                    job_id, progress, 
                                                    f"Training epoch {current}/{total}"
                                                )
                            except:
                                pass
                        
                        elif "accuracy:" in line_str.lower():
                            # Extract accuracy
                            try:
                                parts = line_str.split()
                                for i, part in enumerate(parts):
                                    if "accuracy" in part.lower() and i + 1 < len(parts):
                                        acc_str = parts[i + 1].replace(",", "").replace("%", "")
                                        try:
                                            accuracy = float(acc_str)
                                            if accuracy <= 1.0:
                                                accuracy *= 100
                                            job_queue.update_job_progress(
                                                job_id, last_progress,
                                                f"Training accuracy: {accuracy:.1f}%"
                                            )
                                        except:
                                            pass
                            except:
                                pass
                        
                        elif "TRAINING COMPLETED" in line_str:
                            job_queue.update_job_progress(job_id, 80.0, "Training completed, saving models...")
                        
                        elif "Saved" in line_str and "model" in line_str:
                            job_queue.update_job_progress(job_id, 82.0, "Models saved successfully")
            
            # Read stderr for errors
            async def read_errors():
                while True:
                    line = await process.stderr.readline()
                    if not line:
                        break
                    
                    line_str = line.decode('utf-8', errors='ignore').strip()
                    if line_str and not any(ignore in line_str for ignore in ["WARNING", "INFO"]):
                        logger.error(f"Container error: {line_str}")
            
            # Run both readers concurrently
            await asyncio.gather(read_output(), read_errors())
            
            # Wait for process to complete
            return_code = await process.wait()
            
            elapsed = int(time.time() - start_time)
            logger.info(f"Training completed in {elapsed}s with return code: {return_code}")
            
            return return_code == 0
            
        except Exception as e:
            logger.error(f"Failed to execute training: {e}")
            return False
    
    def _retrieve_results(self, job_id: str) -> Dict:
        """Retrieve training results from container"""
        try:
            results = {}
            
            # Define model files to retrieve
            model_files = {
                'embedding': f'{self.temp_dir}/{job_id}_models/signature_model_embedding.keras',
                'classification': f'{self.temp_dir}/{job_id}_models/signature_model_classification.keras',
                'mappings': f'{self.temp_dir}/{job_id}_models/signature_model_mappings.json',
                'training_results': f'{self.temp_dir}/{job_id}_models/training_results.json'
            }
            
            # Create host directory for this job
            job_models_dir = os.path.join(self.host_models_dir, job_id)
            os.makedirs(job_models_dir, exist_ok=True)
            
            # Copy each model file from container
            model_paths = {}
            for model_type, container_path in model_files.items():
                try:
                    # Check if file exists in container
                    check_cmd = ["docker", "exec", self.container_name, "test", "-f", container_path]
                    check_result = subprocess.run(check_cmd, capture_output=True)
                    
                    if check_result.returncode == 0:
                        # File exists, copy it
                        filename = os.path.basename(container_path)
                        host_path = os.path.join(job_models_dir, filename)
                        self._copy_from_container(container_path, host_path)
                        model_paths[model_type] = host_path
                        logger.info(f"Retrieved {model_type} model: {host_path}")
                    else:
                        logger.warning(f"{model_type} model not found in container: {container_path}")
                        
                except Exception as e:
                    logger.error(f"Failed to retrieve {model_type} model: {e}")
            
            results['model_paths'] = model_paths
            
            # Try to load training results for metrics
            if 'training_results' in model_paths:
                try:
                    with open(model_paths['training_results'], 'r') as f:
                        training_results = json.load(f)
                        results['accuracy'] = training_results.get('final_accuracy')
                        results['training_metrics'] = training_results.get('training_metadata', {})
                except Exception as e:
                    logger.error(f"Failed to load training results: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve results: {e}")
            return {}
    
    def _cleanup_container_files(self, job_id: str):
        """Clean up temporary files in container"""
        try:
            # Remove training data
            data_file = f"{self.workspace_path}/training_data_{job_id}.json"
            subprocess.run(
                ["docker", "exec", self.container_name, "rm", "-f", data_file],
                capture_output=True,
                check=False
            )
            
            # Remove model directory
            model_dir = f"{self.temp_dir}/{job_id}_models"
            subprocess.run(
                ["docker", "exec", self.container_name, "rm", "-rf", model_dir],
                capture_output=True,
                check=False
            )
            
            logger.info(f"Cleaned up container files for job {job_id}")
            
        except Exception as e:
            logger.warning(f"Failed to cleanup container files: {e}")
    
    def stop_container(self):
        """Stop the Docker container (optional - for cleanup)"""
        try:
            if self.is_container_running():
                subprocess.run(["docker", "stop", self.container_name], check=True)
                logger.info(f"Stopped container: {self.container_name}")
        except Exception as e:
            logger.error(f"Failed to stop container: {e}")


# Global instance
docker_training_manager = DockerTrainingManager()
