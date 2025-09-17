"""
AWS GPU Training Manager - Fixed Instance Ready Check
Automatically provisions GPU instances for AI training
"""

import boto3
import time
import json
import logging
from typing import Dict, Optional, List
from botocore.exceptions import ClientError
import asyncio
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class AWSGPUTrainingManager:
    """
    Manages AWS GPU instances for AI training
    """
    
    def __init__(self):
        region_name = os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION') or 'us-east-1'
        self.ec2_client = boto3.client('ec2', region_name=region_name)
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.ssm_client = boto3.client('ssm', region_name=region_name)
        
        # GPU instance configuration
        self.gpu_instance_type = os.getenv('AWS_GPU_INSTANCE_TYPE', 'g4dn.xlarge')
        self.ami_id = os.getenv('AWS_GPU_AMI_ID', 'ami-0c02fb55956c7d316')
        self.key_name = os.getenv('AWS_KEY_NAME', 'your-key-pair')
        self.security_group_id = os.getenv('AWS_SECURITY_GROUP_ID', 'sg-xxxxxxxxx')
        self.subnet_id = os.getenv('AWS_SUBNET_ID', 'subnet-xxxxxxxxx')
        self.existing_instance_id = (os.getenv('AWS_GPU_EXISTING_INSTANCE_ID', '').strip() or None)
        
        # Training configuration
        self.training_script_path = '/home/ubuntu/ai-training'
        self.s3_bucket = os.getenv('S3_BUCKET', 'your-s3-bucket')
        self.github_repo = os.getenv('AWS_GPU_GITHUB_REPO', 'https://github.com/your-repo/ai-training.git')
        
        # IAM instance profile name
        self.iam_instance_profile = os.getenv('AWS_IAM_INSTANCE_PROFILE', 'EC2-S3-Access')

    def is_available(self) -> bool:
        """Lightweight capability check to decide if GPU training can be attempted."""
        try:
            # Check if essential configuration is present
            if not self.s3_bucket or self.s3_bucket == 'your-s3-bucket':
                logger.warning("S3 bucket not configured")
                return False
                
            # Simple call that requires basic EC2 perms
            self.ec2_client.describe_instance_types(MaxResults=5)
            return True
        except Exception as e:
            logger.warning(f"GPU manager not available: {e}")
            return False
        
    async def start_gpu_training(self, 
                                training_data: Dict,
                                job_id: str,
                                student_id: int) -> Dict:
        """
        Start training on AWS GPU instance with real-time progress updates
        """
        instance_id = None
        try:
            logger.info(f"Starting GPU training for job {job_id}")
            
            # Import job queue for progress updates
            from utils.job_queue import job_queue
            
            # Step 1: Launch GPU instance
            job_queue.update_job_progress(job_id, 5.0, "Launching GPU instance...")
            instance_id = await self._launch_gpu_instance(job_id)
            if not instance_id:
                raise Exception("Failed to launch GPU instance")
            
            # Step 2: Wait for instance to be ready
            job_queue.update_job_progress(job_id, 15.0, "Waiting for GPU instance to be ready...")
            await self._wait_for_instance_ready(instance_id)
            
            # Step 3: Get instance IP
            instance_ip = await self._get_instance_ip(instance_id)
            logger.info(f"GPU instance {instance_id} ready at {instance_ip}")
            
            # Step 4: Upload training data to S3
            job_queue.update_job_progress(job_id, 25.0, "Uploading training data to S3...")
            training_data_key = await self._upload_training_data(training_data, job_id)
            
            # Step 5: Setup and start training on GPU instance
            job_queue.update_job_progress(job_id, 30.0, "Setting up training environment...")
            await self._setup_training_environment(instance_id, job_id)
            
            job_queue.update_job_progress(job_id, 40.0, "Starting training on GPU instance...")
            logger.info(f"DEBUG: About to start remote training with data key: {training_data_key}")
            training_result = await self._start_remote_training(
                instance_id, training_data_key, job_id, student_id, job_queue
            )
            logger.info(f"DEBUG: Remote training result: {training_result}")
            
            if training_result.get('status') != 'success':
                raise Exception(f"Training failed: {training_result.get('error', 'Unknown error')}")
            
            # Step 6: Download results
            job_queue.update_job_progress(job_id, 85.0, "Downloading training results...")
            results = await self._download_training_results(job_id)
            job_queue.update_job_progress(job_id, 90.0, "Training results downloaded successfully")
            
            # Step 7: Terminate instance (if not reusing)
            job_queue.update_job_progress(job_id, 95.0, "Cleaning up GPU instance...")
            await self._terminate_instance(instance_id)
            
            job_queue.update_job_progress(job_id, 100.0, "GPU training completed successfully!")
            
            return {
                'success': True,
                'instance_id': instance_id,
                'model_urls': results.get('model_urls', {}),
                'accuracy': results.get('accuracy'),
                'training_metrics': results.get('training_metrics', {}),
                'training_result': training_result
            }
            
        except Exception as e:
            logger.error(f"GPU training failed: {e}")
            # Cleanup on failure
            if instance_id and not self.existing_instance_id:
                try:
                    await self._terminate_instance(instance_id)
                except:
                    pass
            
            # Update job status to failed
            try:
                from utils.job_queue import job_queue
                job_queue.update_job_progress(job_id, 0.0, f"GPU training failed: {str(e)}")
            except:
                pass
            return {'success': False, 'error': str(e)}
    
    async def _launch_gpu_instance(self, job_id: str) -> Optional[str]:
        """Launch GPU instance for training"""
        try:
            # Reuse existing instance if configured
            if self.existing_instance_id:
                logger.info(f"Reusing existing GPU instance: {self.existing_instance_id}")
                # Verify instance exists and check its state
                try:
                    response = self.ec2_client.describe_instances(InstanceIds=[self.existing_instance_id])
                    if response['Reservations']:
                        instance = response['Reservations'][0]['Instances'][0]
                        state = instance['State']['Name']
                        
                        if state == 'running':
                            logger.info(f"Instance {self.existing_instance_id} is already running")
                            return self.existing_instance_id
                        elif state == 'stopped':
                            logger.info(f"Starting stopped instance {self.existing_instance_id}")
                            self.ec2_client.start_instances(InstanceIds=[self.existing_instance_id])
                            return self.existing_instance_id
                        elif state == 'stopping':
                            logger.warning(f"Instance {self.existing_instance_id} is stopping, waiting...")
                            # Wait for it to stop then start it
                            waiter = self.ec2_client.get_waiter('instance_stopped')
                            await asyncio.get_event_loop().run_in_executor(
                                None, 
                                lambda: waiter.wait(InstanceIds=[self.existing_instance_id])
                            )
                            self.ec2_client.start_instances(InstanceIds=[self.existing_instance_id])
                            return self.existing_instance_id
                        else:
                            logger.warning(f"Cannot use instance {self.existing_instance_id} in state: {state}")
                    else:
                        logger.warning(f"Instance {self.existing_instance_id} not found")
                except ClientError as e:
                    if e.response['Error']['Code'] == 'InvalidInstanceID.NotFound':
                        logger.warning(f"Instance {self.existing_instance_id} not found")
                    else:
                        logger.error(f"Error checking existing instance: {e}")
                        
            # Launch new instance
            user_data = self._generate_user_data()
            
            run_args = {
                'ImageId': self.ami_id,
                'MinCount': 1,
                'MaxCount': 1,
                'InstanceType': self.gpu_instance_type,
                'UserData': user_data,
                'TagSpecifications': [
                    {
                        'ResourceType': 'instance',
                        'Tags': [
                            {'Key': 'Name', 'Value': f'gpu-training-{job_id}'},
                            {'Key': 'Purpose', 'Value': 'AI-Training'},
                            {'Key': 'JobId', 'Value': job_id},
                            {'Key': 'LaunchTime', 'Value': datetime.utcnow().isoformat()}
                        ]
                    }
                ]
            }
            
            # Add optional parameters if configured
            if self.key_name and self.key_name != 'your-key-pair':
                run_args['KeyName'] = self.key_name
                
            if self.security_group_id and self.security_group_id != 'sg-xxxxxxxxx':
                run_args['SecurityGroupIds'] = [self.security_group_id]
                
            if self.subnet_id and self.subnet_id != 'subnet-xxxxxxxxx':
                run_args['SubnetId'] = self.subnet_id
                
            if self.iam_instance_profile and self.iam_instance_profile != 'EC2-S3-Access':
                run_args['IamInstanceProfile'] = {'Name': self.iam_instance_profile}

            response = self.ec2_client.run_instances(**run_args)
            
            instance_id = response['Instances'][0]['InstanceId']
            logger.info(f"Launched new GPU instance: {instance_id}")
            
            return instance_id
            
        except ClientError as e:
            logger.error(f"Failed to launch GPU instance: {e}")
            return None
    
    def _generate_user_data(self) -> str:
        """Generate user data script for instance initialization"""
        return f"""#!/bin/bash
set -e

# Log output
exec > >(tee -a /var/log/user-data.log)
exec 2>&1

echo "Starting instance setup at $(date)"

# Update system
apt-get update -y

# Install required packages
apt-get install -y python3-pip git curl

# Install AWS CLI if not present
if ! command -v aws &> /dev/null; then
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    apt-get install -y unzip
    unzip awscliv2.zip
    ./aws/install
    rm -rf aws awscliv2.zip
fi

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    usermod -aG docker ubuntu
    rm get-docker.sh
fi

# Install Python packages
pip3 install --upgrade pip
pip3 install tensorflow boto3 numpy pillow opencv-python scikit-learn requests

# Create training directory
mkdir -p {self.training_script_path}
chown -R ubuntu:ubuntu {self.training_script_path}

# Mark setup as complete
touch /tmp/setup_complete
echo "Instance setup completed at $(date)"
"""
    
    async def _wait_for_instance_ready(self, instance_id: str, timeout: int = 300):
        """Wait for instance to be ready - Fixed version with proper status handling"""
        logger.info(f"Waiting for instance {instance_id} to be ready...")
        
        start_time = time.time()
        last_state = None
        status_check_passed = False
        
        while time.time() - start_time < timeout:
            try:
                # First check instance state
                response = self.ec2_client.describe_instances(InstanceIds=[instance_id])
                if not response['Reservations']:
                    logger.debug(f"Instance {instance_id} not found yet, waiting...")
                    await asyncio.sleep(5)
                    continue
                    
                instance = response['Reservations'][0]['Instances'][0]
                state = instance['State']['Name']
                
                if state != last_state:
                    logger.info(f"Instance {instance_id} state: {state}")
                    last_state = state
                
                # For existing instances that are already running
                if self.existing_instance_id and instance_id == self.existing_instance_id:
                    if state == 'running':
                        # For reused instances, check if SSM is available
                        try:
                            # Try to send a simple command to check if SSM is ready
                            test_response = self.ssm_client.send_command(
                                InstanceIds=[instance_id],
                                DocumentName='AWS-RunShellScript',
                                Parameters={'commands': ['echo "SSM Ready"']}
                            )
                            logger.info(f"Instance {instance_id} is ready (SSM available)!")
                            return True
                        except ClientError as e:
                            if e.response['Error']['Code'] in ['InvalidInstanceId', 'InvalidInstanceInformationFilterValue']:
                                logger.debug(f"SSM not ready yet for instance {instance_id}")
                                # Instance is running but SSM agent might not be ready
                                await asyncio.sleep(10)
                                continue
                            else:
                                # Some other error - log but continue
                                logger.debug(f"SSM check error: {e}")
                                await asyncio.sleep(10)
                                continue
                    elif state == 'pending':
                        # Wait for it to become running
                        await asyncio.sleep(10)
                        continue
                    else:
                        # Unexpected state
                        raise Exception(f"Instance {instance_id} in unexpected state: {state}")
                
                # For new instances
                if state == 'pending':
                    await asyncio.sleep(10)
                    continue
                elif state == 'running':
                    # Wait for status checks only for new instances
                    if not self.existing_instance_id or instance_id != self.existing_instance_id:
                        if not status_check_passed:
                            try:
                                status_response = self.ec2_client.describe_instance_status(
                                    InstanceIds=[instance_id]
                                )
                                if status_response['InstanceStatuses']:
                                    status = status_response['InstanceStatuses'][0]
                                    instance_status = status.get('InstanceStatus', {}).get('Status')
                                    system_status = status.get('SystemStatus', {}).get('Status')
                                    
                                    logger.debug(f"Instance status: {instance_status}, System status: {system_status}")
                                    
                                    if instance_status == 'ok' and system_status == 'ok':
                                        status_check_passed = True
                                        logger.info(f"Status checks passed for instance {instance_id}")
                                    elif instance_status == 'initializing' or system_status == 'initializing':
                                        logger.debug("Status checks initializing...")
                                        await asyncio.sleep(10)
                                        continue
                                else:
                                    logger.debug("No status information available yet")
                                    await asyncio.sleep(10)
                                    continue
                            except Exception as e:
                                logger.debug(f"Error checking instance status: {e}")
                                await asyncio.sleep(10)
                                continue
                    
                    # Check if SSM is available (final check for readiness)
                    try:
                        test_response = self.ssm_client.send_command(
                            InstanceIds=[instance_id],
                            DocumentName='AWS-RunShellScript',
                            Parameters={'commands': ['echo "SSM Ready"']}
                        )
                        logger.info(f"Instance {instance_id} is fully ready!")
                        return True
                    except ClientError as e:
                        if e.response['Error']['Code'] in ['InvalidInstanceId', 'InvalidInstanceInformationFilterValue']:
                            logger.debug(f"SSM agent not ready on instance {instance_id}, waiting...")
                            await asyncio.sleep(10)
                            continue
                        else:
                            raise
                elif state in ['terminated', 'terminating', 'shutting-down']:
                    raise Exception(f"Instance {instance_id} is {state}")
                
                await asyncio.sleep(10)
                
            except Exception as e:
                if "InvalidInstanceID.NotFound" in str(e):
                    logger.debug(f"Instance {instance_id} not found yet, waiting...")
                    await asyncio.sleep(10)
                else:
                    logger.error(f"Error checking instance {instance_id}: {e}")
                    raise
        
        # Timeout reached - provide helpful error message
        try:
            response = self.ec2_client.describe_instances(InstanceIds=[instance_id])
            if response['Reservations']:
                instance = response['Reservations'][0]['Instances'][0]
                state = instance['State']['Name']
                error_msg = f"Instance {instance_id} did not become ready within {timeout} seconds. Current state: {state}"
                
                # Check status checks for more info
                try:
                    status_response = self.ec2_client.describe_instance_status(InstanceIds=[instance_id])
                    if status_response['InstanceStatuses']:
                        status = status_response['InstanceStatuses'][0]
                        error_msg += f", Instance status: {status.get('InstanceStatus', {}).get('Status', 'unknown')}"
                        error_msg += f", System status: {status.get('SystemStatus', {}).get('Status', 'unknown')}"
                except:
                    pass
                    
                raise Exception(error_msg)
            else:
                raise Exception(f"Instance {instance_id} not found after {timeout} seconds")
        except:
            raise Exception(f"Instance {instance_id} did not become ready within {timeout} seconds")
    
    async def _get_instance_ip(self, instance_id: str) -> str:
        """Get instance public IP"""
        try:
            response = self.ec2_client.describe_instances(InstanceIds=[instance_id])
            if response['Reservations']:
                instance = response['Reservations'][0]['Instances'][0]
                return instance.get('PublicIpAddress', instance.get('PrivateIpAddress', 'N/A'))
            return 'N/A'
        except Exception as e:
            logger.error(f"Failed to get instance IP: {e}")
            return 'N/A'
    
    async def _upload_training_data(self, training_data: Dict, job_id: str) -> str:
        """Upload training data to S3 (gzip-compressed to reduce time)."""
        import gzip
        # Convert images to serializable format
        serializable_data = {}
        for student_name, signatures in training_data.items():
            serializable_data[student_name] = {}
            # Handle different data structures; map legacy keys to 'genuine'
            if isinstance(signatures, dict):
                # Prefer 'genuine' if present
                if 'genuine' in signatures and signatures['genuine']:
                    serializable_data[student_name]['genuine'] = [
                        (img.tolist() if hasattr(img, 'tolist') else img) for img in signatures['genuine']
                    ]
                elif 'genuine_images' in signatures and signatures['genuine_images']:
                    serializable_data[student_name]['genuine'] = [
                        (img.tolist() if hasattr(img, 'tolist') else img) for img in signatures['genuine_images']
                    ]
                else:
                    serializable_data[student_name]['genuine'] = []
                # Explicitly ignore forged for owner detection
                serializable_data[student_name]['forged'] = []
            else:
                # If it's a list, treat as genuine signatures
                serializable_data[student_name] = {
                    'genuine': [img.tolist() if hasattr(img, 'tolist') else img for img in signatures],
                    'forged': []
                }
        
        # Upload to S3 (gzip)
        payload = json.dumps(serializable_data).encode('utf-8')
        compressed = gzip.compress(payload)
        key = f'training_data/{job_id}.json.gz'
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=key,
            Body=compressed,
            ContentType='application/json',
            ContentEncoding='gzip'
        )
        
        logger.info(f"Uploaded training data to s3://{self.s3_bucket}/{key}")
        return key
    
    async def _setup_training_environment(self, instance_id: str, job_id: str):
        """Setup training environment on the instance"""
        try:
            # Create training script
            training_script = self._generate_training_script()
            logger.info(f"DEBUG: Generated training script length: {len(training_script)} characters")
            
            # Upload script to S3 first
            script_key = f'scripts/{job_id}/train_gpu.py'
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=script_key,
                Body=training_script,
                ContentType='text/x-python'
            )
            logger.info(f"DEBUG: Uploaded training script to s3://{self.s3_bucket}/{script_key}")
            
            # Download and setup script on instance using SSM
            setup_commands = [
                f'mkdir -p {self.training_script_path}',
                f'cd {self.training_script_path}',
                f'aws s3 cp s3://{self.s3_bucket}/{script_key} train_gpu.py',
                'chmod +x train_gpu.py',
                f'echo "Setup complete for job {job_id}"'
            ]
            
            response = self.ssm_client.send_command(
                InstanceIds=[instance_id],
                DocumentName='AWS-RunShellScript',
                Parameters={'commands': setup_commands}
            )
            
            command_id = response['Command']['CommandId']
            
            # Wait for setup to complete
            await self._wait_for_command(command_id, instance_id)
            logger.info(f"Training environment setup completed for job {job_id}")
            
        except Exception as e:
            logger.error(f"Failed to setup training environment: {e}")
            raise
    
    def _generate_training_script(self) -> str:
        """Generate the training script without outer f-string to avoid brace conflicts."""
        script = '''#!/usr/bin/env python3
import sys
import os
import json
import boto3
import numpy as np
from PIL import Image
import io
import requests
import traceback
import tensorflow as tf
from tensorflow import keras
import tempfile
import shutil

# Set up TensorFlow to use GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s), using GPU acceleration")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("No GPU found, using CPU")

def train_on_gpu(training_data_key, job_id, student_id):
    """Train model on GPU instance"""
    try:
        # Initialize S3 client
        s3 = boto3.client('s3')
        bucket = '__S3_BUCKET__'
        
        print(f"Starting REAL training for job {job_id}")
        
        # Download training data from S3 (supports gzip)
        print(f"Downloading training data from s3://{bucket}/{training_data_key}")
        local_path = '/tmp/training_data.json'
        if training_data_key.endswith('.gz'):
            local_path += '.gz'
        s3.download_file(bucket, training_data_key, local_path)
        
        if local_path.endswith('.gz'):
            import gzip
            with gzip.open(local_path, 'rb') as f:
                training_data = json.loads(f.read().decode('utf-8'))
        else:
            with open(local_path, 'r') as f:
                training_data = json.load(f)
        
        print(f"Loaded training data for {len(training_data)} students")
        
        # Install required packages
        print("Installing required packages...")
        os.system("pip install tensorflow pillow numpy scikit-learn")
        
        # Self-contained preprocessor to avoid import errors
        class _Preproc:
            def __init__(self, target_size=224):
                self.target_size = target_size
            def preprocess_signature(self, img):
                if isinstance(img, Image.Image):
                    img = np.array(img)
                if img.ndim == 2:
                    img = np.stack([img]*3, axis=-1)
                if img.shape[-1] == 4:
                    img = img[:, :, :3]
                img = img.astype(np.float32)
                if img.max() > 1.0:
                    img = img / 255.0
                from PIL import Image as _PIL
                pil = _PIL.fromarray((img*255).astype(np.uint8))
                pil = pil.resize((224, 224))
                out = np.array(pil).astype(np.float32) / 255.0
                return out
        
        # Process training data
        print("Processing training data...")
        processed_data = {}
        preprocessor = _Preproc(target_size=224)
        
        for student_name, data in training_data.items():
            print(f"Processing {student_name}...")
            genuine_images = []
            forged_images = []
            
            # Compatible with {'genuine': [...] } and legacy {'genuine_images': [...]}
            raw_genuine = data.get('genuine')
            if raw_genuine is None:
                raw_genuine = data.get('genuine_images', [])
            raw_forged = data.get('forged')
            if raw_forged is None:
                raw_forged = data.get('forged_images', [])
            
            # Process genuine images
            for img_data in raw_genuine:
                try:
                    if isinstance(img_data, str):
                        import base64
                        img_bytes = base64.b64decode(img_data)
                        img = Image.open(io.BytesIO(img_bytes))
                    else:
                        img = Image.fromarray((np.array(img_data) * 255).astype(np.uint8))
                    processed_img = preprocessor.preprocess_signature(img)
                    genuine_images.append(processed_img)
                except Exception as e:
                    print(f"Error processing image: {e}")
                    continue
            
            # Process forged images
            for img_data in raw_forged:
                try:
                    if isinstance(img_data, str):
                        import base64
                        img_bytes = base64.b64decode(img_data)
                        img = Image.open(io.BytesIO(img_bytes))
                    else:
                        img = Image.fromarray((np.array(img_data) * 255).astype(np.uint8))
                    processed_img = preprocessor.preprocess_signature(img)
                    forged_images.append(processed_img)
                except Exception as e:
                    print(f"Error processing forged image: {e}")
                    continue
            
            if len(genuine_images) > 0:
                processed_data[student_name] = {
                    'genuine': genuine_images,
                    'forged': forged_images
                }
                print(f"{student_name}: {len(genuine_images)} genuine, {len(forged_images)} forged")
            else:
                print(f"Skipping {student_name} (no genuine images)")
        
        # Build dataset
        classes = sorted(list(processed_data.keys()))
        # Build exact mappings: class index -> student id and name
        class_to_student_id = {}
        class_to_student_name = {}
        for idx, cls in enumerate(classes):
            try:
                # Parse "id:name" format from class key
                if isinstance(cls, str) and ':' in cls:
                    parts = cls.split(':', 1)
                    if len(parts) == 2:
                        try:
                            sid = int(parts[0])
                            name = parts[1].strip()
                            class_to_student_id[idx] = sid
                            class_to_student_name[idx] = name
                        except ValueError:
                            # Fallback to using the whole string as name
                            class_to_student_name[idx] = str(cls)
                    else:
                        class_to_student_name[idx] = str(cls)
                else:
                    # Fallback for old format
                    class_to_student_name[idx] = str(cls)
            except Exception:
                class_to_student_name[idx] = str(cls)
        X, y = [], []
        for cname, data in processed_data.items():
            for img in data.get('genuine', []):
                X.append(img)
                y.append(classes.index(cname))
        X = np.array(X, dtype=np.float32) if len(X) > 0 else np.array([])
        y = np.array(y, dtype=np.int64) if len(y) > 0 else np.array([])
        total_samples = int(len(X)) if hasattr(X, '__len__') else 0
        num_classes = int(len(classes))
        
        if total_samples == 0 or num_classes == 0:
            print("No training samples available after preprocessing. Writing zero-accuracy results.")
            results = {
                'job_id': job_id,
                'student_id': student_id,
                'model_urls': {},
                'accuracy': 0.0,
                'training_metrics': {
                    'accuracy': 0.0,
                    'epochs_trained': 0,
                    'final_loss': 0.0,
                    'val_accuracy': 0.0,
                    'val_loss': 0.0
                }
            }
            results_path = '/tmp/training_results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f)
            s3.upload_file(results_path, bucket, f'training_results/{job_id}.json')
            print(f"Training completed with no data for job {job_id}. Final accuracy: 0.0")
            return
        
        # Train a simple real classifier (MobileNetV2 backbone)
        print("Starting model training...")
        print(f"Dataset summary: classes={num_classes}, total_samples={total_samples}")
        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        import numpy as _np
        import random as _random
        tf.random.set_seed(42)
        _np.random.seed(42)
        _random.seed(42)
        if total_samples > 4 and len(np.unique(y)) > 1:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_val, y_train, y_val = X, X, y, y
        
        # Prepare output directory early (for checkpoints)
        temp_dir = f'/tmp/{job_id}_models'
        os.makedirs(temp_dir, exist_ok=True)

        # Create a simple but effective classifier with light augmentation
        aug = keras.Sequential([
            keras.layers.RandomFlip('horizontal'),
            keras.layers.RandomRotation(0.02),
            keras.layers.RandomZoom(0.1)
        ])
        inputs = keras.Input(shape=(224,224,3))
        x = aug(inputs)
        x = keras.layers.Conv2D(32, 3, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(2)(x)
        x = keras.layers.Conv2D(64, 3, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(2)(x)
        x = keras.layers.Conv2D(128, 3, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Compute class weights to reduce collapse to majority class
        class_weight = None
        try:
            if len(y) > 0:
                counts = _np.bincount(y, minlength=num_classes)
                total = counts.sum()
                # Avoid division by zero
                class_weight = {i: (total / (num_classes * counts[i])) if counts[i] > 0 else 0.0 for i in range(num_classes)}
                print(f"Class counts: {counts}, class_weight: {class_weight}")
        except Exception as e:
            print(f"Failed to compute class weights: {e}")

        if len(X_train) == 0:
            # Safety: if still empty, skip training
            hist = type('H', (), {'history': {'accuracy': [0.0], 'val_accuracy': [0.0], 'loss': [0.0], 'val_loss': [0.0]}})()
        else:
            class EpochLogger(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    acc = logs.get('accuracy', 0.0)
                    val_acc = logs.get('val_accuracy', 0.0)
                    loss = logs.get('loss', 0.0)
                    val_loss = logs.get('val_loss', 0.0)
                    print(f"Epoch {epoch+1}: loss={loss:.4f} acc={acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            # Callbacks: early stopping and best checkpoint
            ckpt_path = f"{temp_dir}/best.keras"
            callbacks = [
                EpochLogger(),
                keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=3, restore_best_weights=True),
                keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=False)
            ]
            hist = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val if len(X_val)>0 else y_train),
                epochs=15,
                batch_size=16,
                verbose=0,
                shuffle=True,
                class_weight=class_weight,
                callbacks=callbacks
            )

        print("Training completed! Saving models...")
        
        # Debug: Check if model actually learned anything
        print(f"Final training accuracy: {hist.history.get('accuracy', [0.0])[-1]:.4f}")
        print(f"Final validation accuracy: {hist.history.get('val_accuracy', [0.0])[-1]:.4f}")
        
        # Test prediction on a few samples to see if model is working
        if len(X_val) > 0:
            test_preds = model.predict(X_val[:3], verbose=0)
            print(f"Sample predictions on validation set: {test_preds}")
            print(f"Predicted classes: {np.argmax(test_preds, axis=1)}")
            print(f"True classes: {y_val[:3]}")
        
        # If checkpoint exists, load best model before saving
        try:
            best_path = f"{temp_dir}/best.keras"
            if os.path.exists(best_path):
                best_model = keras.models.load_model(best_path, compile=False)
                model = best_model
                print("Loaded best checkpoint before saving final model")
        except Exception as e:
            print(f"Failed to load best checkpoint: {e}")
        
        # Save only the global classification model
        model_urls = {}
        cls_path = f'{temp_dir}/classification.keras'
        model.save(cls_path)
        s3_key = f'models/{job_id}/classification.keras'
        s3.upload_file(cls_path, bucket, s3_key)
        model_urls['classification'] = f'https://{bucket}.s3.amazonaws.com/{s3_key}'
        print("Uploaded classification model to S3")
        
        # Save mappings
        class_to_idx = {c: i for i, c in enumerate(classes)}
        mappings = {
            'student_id': student_id,
            'students': classes,
            'class_to_idx': class_to_idx,
            'id_to_student_id': class_to_student_id,  # exact IDs when available
            'id_to_student_name': class_to_student_name  # exact names per class index
        }
        mappings_path = f'/tmp/{job_id}_mappings.json'
        with open(mappings_path, 'w') as f:
            json.dump(mappings, f)
        
        s3_key = f'models/{job_id}/mappings.json'
        s3.upload_file(mappings_path, bucket, s3_key)
        model_urls['mappings'] = f'https://{bucket}.s3.amazonaws.com/{s3_key}'
        
        # Calculate final accuracy (always numeric)
        final_accuracy = 0.0
        try:
            if hist.history.get('val_accuracy'):
                final_accuracy = float(hist.history['val_accuracy'][-1])
            elif hist.history.get('accuracy'):
                final_accuracy = float(hist.history['accuracy'][-1])
        except Exception:
            final_accuracy = 0.0
        
        # Save training results
        results = {
            'job_id': job_id,
            'student_id': student_id,
            'model_urls': model_urls,
            'accuracy': final_accuracy,
            'training_metrics': {
                'accuracy': final_accuracy,
                'epochs_trained': len(hist.history.get('loss', [])) if hasattr(hist, 'history') else 0,
                'final_loss': float(hist.history.get('loss', [0])[-1]) if hasattr(hist, 'history') and hist.history.get('loss') else 0.0,
                'val_accuracy': float(hist.history.get('val_accuracy', [0])[-1]) if hasattr(hist, 'history') and hist.history.get('val_accuracy') else 0.0,
                'val_loss': float(hist.history.get('val_loss', [0])[-1]) if hasattr(hist, 'history') and hist.history.get('val_loss') else 0.0
            }
        }
        
        results_path = '/tmp/training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f)
        
        s3.upload_file(results_path, bucket, f'training_results/{job_id}.json')
        print(f"Training completed successfully for job {job_id}! Final accuracy: {final_accuracy}")
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: train_gpu.py <training_data_key> <job_id> <student_id>")
        sys.exit(1)
        
    training_data_key = sys.argv[1]
    job_id = sys.argv[2]
    student_id = int(sys.argv[3])
    
    train_on_gpu(training_data_key, job_id, student_id)
'''
        return script.replace('__S3_BUCKET__', self.s3_bucket)
    
    async def _start_remote_training(self, instance_id: str, training_data_key: str, 
                                   job_id: str, student_id: int, job_queue) -> Dict:
        """Start training on remote GPU instance"""
        try:
            # Run training command
            training_command = [
                f'cd {self.training_script_path}',
                f'ls -la',  # Debug: list files
                f'head -20 train_gpu.py',  # Debug: show first 20 lines of script
                f'python3 train_gpu.py {training_data_key} {job_id} {student_id}'
            ]
            
            logger.info(f"DEBUG: Running training commands: {training_command}")
            response = self.ssm_client.send_command(
                InstanceIds=[instance_id],
                DocumentName='AWS-RunShellScript',
                Parameters={'commands': training_command}
            )
            
            command_id = response['Command']['CommandId']
            logger.info(f"Started training command {command_id} on instance {instance_id}")
            
            # Monitor training progress
            start_time = time.time()
            check_interval = 10  # Check every 10 seconds
            max_duration = 3600  # 1 hour max
            
            while True:
                await asyncio.sleep(check_interval)
                
                try:
                    result = self.ssm_client.get_command_invocation(
                        CommandId=command_id,
                        InstanceId=instance_id
                    )
                    
                    status = result['Status']
                    
                    # Update progress based on output
                    output = result.get('StandardOutputContent', '')
                    if 'Training progress:' in output:
                        # Extract progress from output
                        lines = output.split('\n')
                        for line in reversed(lines):
                            if 'Training progress:' in line:
                                try:
                                    progress = float(line.split(':')[1].strip().replace('%', ''))
                                    job_queue.update_job_progress(job_id, 40 + (progress * 0.4), 
                                                                 f"Training in progress: {progress}%")
                                except:
                                    pass
                                break
                    
                    if status in ['Success', 'Failed', 'Cancelled', 'TimedOut']:
                        break
                    
                    # Check timeout
                    if time.time() - start_time > max_duration:
                        logger.warning(f"Training timeout for job {job_id}")
                        # Cancel the command
                        self.ssm_client.cancel_command(CommandId=command_id)
                        return {'status': 'failed', 'error': 'Training timeout'}
                        
                except ClientError as e:
                    if e.response['Error']['Code'] == 'InvocationDoesNotExist':
                        # Command hasn't started yet
                        continue
                    else:
                        raise
            
            if status == 'Success':
                logger.info(f"Training completed successfully for job {job_id}")
                # Validate that results JSON exists and has model_urls
                try:
                    res = self.s3_client.get_object(Bucket=self.s3_client.meta.region_name and self.s3_client._endpoint.host and self.s3_client._endpoint.host and self.s3_client._endpoint.host or '', Key=f'training_results/{job_id}.json')
                except Exception:
                    pass
                return {
                    'status': 'success', 
                    'output': result.get('StandardOutputContent', ''),
                    'duration': time.time() - start_time
                }
            else:
                error = result.get('StandardErrorContent', 'Unknown error')
                logger.error(f"Training failed for job {job_id}: {error}")
                return {'status': 'failed', 'error': error}
                
        except Exception as e:
            logger.error(f"Failed to start remote training: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _wait_for_command(self, command_id: str, instance_id: str, timeout: int = 60):
        """Wait for SSM command to complete"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                result = self.ssm_client.get_command_invocation(
                    CommandId=command_id,
                    InstanceId=instance_id
                )
                
                if result['Status'] in ['Success', 'Failed', 'Cancelled', 'TimedOut']:
                    if result['Status'] != 'Success':
                        raise Exception(f"Command failed: {result.get('StandardErrorContent', 'Unknown error')}")
                    return result
                    
            except ClientError as e:
                if e.response['Error']['Code'] == 'InvocationDoesNotExist':
                    # Command hasn't started yet
                    pass
                else:
                    raise
                    
            await asyncio.sleep(2)
        
        raise Exception(f"Command {command_id} did not complete within {timeout} seconds")
    
    async def _download_training_results(self, job_id: str) -> Dict:
        """Download training results from S3"""
        try:
            # Download results
            response = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key=f'training_results/{job_id}.json'
            )
            
            results = json.loads(response['Body'].read())
            logger.info(f"Downloaded training results for job {job_id}")
            
            # Return both model URLs and accuracy metrics
            return {
                'model_urls': results.get('model_urls', {}),
                'accuracy': results.get('accuracy'),
                'training_metrics': results.get('training_metrics', {})
            }
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.error(f"Training results not found for job {job_id}")
                return {'model_urls': {}, 'accuracy': None, 'training_metrics': {}}
            else:
                logger.error(f"Failed to download training results: {e}")
                return {'model_urls': {}, 'accuracy': None, 'training_metrics': {}}
    
    async def _terminate_instance(self, instance_id: str):
        """Terminate GPU instance"""
        try:
            if self.existing_instance_id and instance_id == self.existing_instance_id:
                logger.info(f"Reused instance {instance_id}: skipping termination")
                return
                
            self.ec2_client.terminate_instances(InstanceIds=[instance_id])
            logger.info(f"Terminated GPU instance: {instance_id}")
        except Exception as e:
            logger.error(f"Failed to terminate instance {instance_id}: {e}")

    async def check_instance_health(self, instance_id: str) -> Dict:
        """Check the health and status of an instance"""
        try:
            # Get instance details
            response = self.ec2_client.describe_instances(InstanceIds=[instance_id])
            if not response['Reservations']:
                return {'healthy': False, 'error': 'Instance not found'}
            
            instance = response['Reservations'][0]['Instances'][0]
            state = instance['State']['Name']
            
            # Get status checks
            status_response = self.ec2_client.describe_instance_status(InstanceIds=[instance_id])
            
            health_info = {
                'instance_id': instance_id,
                'state': state,
                'instance_type': instance.get('InstanceType'),
                'launch_time': str(instance.get('LaunchTime')),
                'public_ip': instance.get('PublicIpAddress'),
                'private_ip': instance.get('PrivateIpAddress'),
                'healthy': False,
                'ssm_available': False
            }
            
            if status_response['InstanceStatuses']:
                status = status_response['InstanceStatuses'][0]
                health_info['instance_status'] = status.get('InstanceStatus', {}).get('Status')
                health_info['system_status'] = status.get('SystemStatus', {}).get('Status')
                health_info['status_checks_passed'] = (
                    health_info['instance_status'] == 'ok' and 
                    health_info['system_status'] == 'ok'
                )
            
            # Check SSM availability
            if state == 'running':
                try:
                    # Try to get SSM instance information
                    ssm_response = self.ssm_client.describe_instance_information(
                        Filters=[
                            {'Key': 'InstanceIds', 'Values': [instance_id]}
                        ]
                    )
                    if ssm_response['InstanceInformationList']:
                        ssm_info = ssm_response['InstanceInformationList'][0]
                        health_info['ssm_available'] = ssm_info.get('PingStatus') == 'Online'
                        health_info['ssm_ping_status'] = ssm_info.get('PingStatus')
                        health_info['ssm_last_ping'] = str(ssm_info.get('LastPingDateTime', ''))
                        health_info['ssm_agent_version'] = ssm_info.get('AgentVersion')
                except ClientError as e:
                    logger.debug(f"SSM check failed: {e}")
                    health_info['ssm_error'] = str(e)
            
            # Determine overall health
            health_info['healthy'] = (
                state == 'running' and 
                health_info.get('status_checks_passed', False) and
                health_info.get('ssm_available', False)
            )
            
            return health_info
            
        except Exception as e:
            logger.error(f"Failed to check instance health: {e}")
            return {'healthy': False, 'error': str(e)}

    async def restart_instance(self, instance_id: str) -> bool:
        """Restart an instance that's not responding properly"""
        try:
            logger.info(f"Attempting to restart instance {instance_id}")
            
            # Stop the instance
            self.ec2_client.stop_instances(InstanceIds=[instance_id])
            
            # Wait for it to stop
            waiter = self.ec2_client.get_waiter('instance_stopped')
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: waiter.wait(
                    InstanceIds=[instance_id],
                    WaiterConfig={'Delay': 10, 'MaxAttempts': 30}
                )
            )
            
            logger.info(f"Instance {instance_id} stopped, starting it again...")
            
            # Start the instance
            self.ec2_client.start_instances(InstanceIds=[instance_id])
            
            logger.info(f"Instance {instance_id} restart initiated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restart instance {instance_id}: {e}")
            return False

    async def install_ssm_agent(self, instance_id: str) -> bool:
        """Try to install/restart SSM agent on the instance via user data"""
        try:
            logger.info(f"Attempting to install/restart SSM agent on {instance_id}")
            
            # Get instance details
            response = self.ec2_client.describe_instances(InstanceIds=[instance_id])
            if not response['Reservations']:
                return False
                
            instance = response['Reservations'][0]['Instances'][0]
            
            # Create a user data script to install/restart SSM
            ssm_install_script = """#!/bin/bash
# Install or restart SSM agent
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
fi

if [ "$OS" == "ubuntu" ] || [ "$OS" == "debian" ]; then
    # Ubuntu/Debian
    snap install amazon-ssm-agent --classic || true
    snap start amazon-ssm-agent || true
    systemctl restart snap.amazon-ssm-agent.amazon-ssm-agent.service || true
elif [ "$OS" == "amzn" ]; then
    # Amazon Linux
    yum install -y amazon-ssm-agent || true
    systemctl restart amazon-ssm-agent || true
fi

# Alternative installation method
cd /tmp
wget https://s3.amazonaws.com/ec2-downloads-windows/SSMAgent/latest/debian_amd64/amazon-ssm-agent.deb
dpkg -i amazon-ssm-agent.deb || true
systemctl enable amazon-ssm-agent
systemctl restart amazon-ssm-agent

echo "SSM agent installation/restart attempted at $(date)" >> /var/log/ssm-install.log
"""
            
            # Update instance user data
            import base64
            encoded_script = base64.b64encode(ssm_install_script.encode()).decode()
            
            # Note: This requires stopping the instance first
            if instance['State']['Name'] == 'running':
                logger.warning(f"Cannot update user data while instance {instance_id} is running")
                return False
            
            self.ec2_client.modify_instance_attribute(
                InstanceId=instance_id,
                UserData={'Value': encoded_script}
            )
            
            logger.info(f"Updated user data for {instance_id} with SSM installation script")
            return True
            
        except Exception as e:
            logger.error(f"Failed to install SSM agent: {e}")
            return False

# Global instance
gpu_training_manager = AWSGPUTrainingManager()
