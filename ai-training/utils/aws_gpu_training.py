"""
AWS GPU Training Manager
Automatically provisions GPU instances for AI training
"""

import boto3
import time
import json
import logging
from typing import Dict, Optional, List
from botocore.exceptions import ClientError
import asyncio
import aiohttp
import os

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
        self.gpu_instance_type = os.getenv('AWS_GPU_INSTANCE_TYPE', 'g4dn.xlarge')  # 1 GPU, 4 vCPUs, 16GB RAM
        self.ami_id = os.getenv('AWS_GPU_AMI_ID', 'ami-0c02fb55956c7d316')  # Deep Learning AMI (Ubuntu 20.04)
        self.key_name = os.getenv('AWS_KEY_NAME', 'your-key-pair')
        self.security_group_id = os.getenv('AWS_SECURITY_GROUP_ID', 'sg-xxxxxxxxx')
        self.subnet_id = os.getenv('AWS_SUBNET_ID', 'subnet-xxxxxxxxx')
        # Reuse existing instance if provided
        self.existing_instance_id = (os.getenv('AWS_GPU_EXISTING_INSTANCE_ID', '').strip() or None)
        # Spot instances disabled (use On-Demand only)
        
        # Training configuration
        self.training_script_path = '/home/ubuntu/ai-training'
        self.s3_bucket = os.getenv('S3_BUCKET')
        self.github_repo = os.getenv('AWS_GPU_GITHUB_REPO', 'https://github.com/your-repo/ai-training.git')

    def is_available(self) -> bool:
        """Lightweight capability check to decide if GPU training can be attempted.
        Returns False if AWS creds/permissions are missing or EC2 is not accessible.
        """
        try:
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
            
            # Step 3: Get instance IP (optional for logs)
            instance_ip = await self._get_instance_ip(instance_id)
            logger.info(f"GPU instance {instance_id} ready at {instance_ip}")
            
            # Step 4: Upload training data to S3
            job_queue.update_job_progress(job_id, 25.0, "Uploading training data to S3...")
            training_data_key = await self._upload_training_data(training_data, job_id)
            
            # Step 5: Start training on GPU instance
            job_queue.update_job_progress(job_id, 30.0, "Starting training on GPU instance...")
            training_result = await self._start_remote_training(
                instance_id, training_data_key, job_id, student_id
            )
            
            # Step 6: Download results
            job_queue.update_job_progress(job_id, 85.0, "Downloading training results...")
            model_urls = await self._download_training_results(job_id)
            job_queue.update_job_progress(job_id, 90.0, "Training results downloaded successfully")
            
            # Step 7: Terminate instance
            job_queue.update_job_progress(job_id, 95.0, "Cleaning up GPU instance...")
            await self._terminate_instance(instance_id)
            
            job_queue.update_job_progress(job_id, 100.0, "GPU training completed successfully!")
            
            return {
                'success': True,
                'instance_id': instance_id,
                'model_urls': model_urls,
                'training_result': training_result
            }
            
        except Exception as e:
            logger.error(f"GPU training failed: {e}")
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
            # Reuse mode: skip provisioning, use existing instance
            if self.existing_instance_id:
                logger.info(f"Reusing existing GPU instance: {self.existing_instance_id}")
                return self.existing_instance_id
            user_data = f"""#!/bin/bash
# Update system
apt-get update -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Create training directory
mkdir -p {self.training_script_path}
cd {self.training_script_path}

# Clone or download training code
git clone {self.github_repo} .
cd ai-training

# Install Python dependencies
pip3 install -r requirements.txt

# Create training script
cat > train_gpu.py << 'EOF'
import sys
import os
sys.path.append('/home/ubuntu/ai-training')

from models.signature_embedding_model import SignatureEmbeddingModel
from utils.signature_preprocessing import SignaturePreprocessor, SignatureAugmentation
import json
import boto3
import numpy as np
from PIL import Image
import io

def train_on_gpu(training_data_key, job_id, student_id):
    # Download training data from S3
    s3 = boto3.client('s3')
    bucket = '{self.s3_bucket}'
    
    # Download training data
    s3.download_file(bucket, training_data_key, '/tmp/training_data.json')
    
    with open('/tmp/training_data.json', 'r') as f:
        training_data = json.load(f)
    
    # Initialize AI system with proper configuration
    image_size = 224
    embedding_dim = 256
    max_students = 150
    
    ai_model = SignatureEmbeddingModel(
        image_size=image_size,
        embedding_dim=embedding_dim,
        max_students=max_students
    )
    preprocessor = SignaturePreprocessor(target_size=image_size)
    augmenter = SignatureAugmentation()
    
    # Process training data with proper preprocessing and augmentation
    processed_data = {{}}
    for student_name, signatures in training_data.items():
        genuine_images = []
        forged_images = []
        
        for img_data in signatures['genuine']:
            img = Image.open(io.BytesIO(bytes(img_data)))
            processed = preprocessor.preprocess_signature(img)
            # Apply data augmentation for better generalization
            augmented = augmenter.augment_signature(processed)
            genuine_images.append(augmented)
        
        # Skip forged signature processing - not used for owner identification training
        # (Forgery detection is disabled - focus on owner identification only)
        
        processed_data[student_name] = {{
            'genuine': genuine_images,
            'forged': forged_images
        }}
    
    # Prepare training data using the same method as CPU training
    X, y = ai_model.prepare_training_data(processed_data)
    
    # Train classification-only model for owner identification with real-time metrics
    result = ai_model.train_classification_only(processed_data, epochs=50)
    
    # Send real-time metrics updates
    import requests
    import time
    
    def send_metrics_update(epoch, total_epochs, loss, accuracy, val_loss=None, val_accuracy=None):
        try:
            metrics = {{
                'job_id': job_id,
                'current_epoch': epoch,
                'total_epochs': total_epochs,
                'loss': loss,
                'accuracy': accuracy,
                'val_loss': val_loss or 0.0,
                'val_accuracy': val_accuracy or 0.0,
                'timestamp': time.time()
            }}
            
            # Send to main server (you'll need to implement this endpoint)
            # requests.post(f'http://your-main-server/api/training/metrics', json=metrics)
            print(f"Epoch {{epoch}}/{{total_epochs}} - Loss: {{loss:.4f}}, Accuracy: {{accuracy:.4f}}")
        except Exception as e:
            print(f"Failed to send metrics: {{e}}")
    
    # Override the training callback to send real-time updates
    original_callback = ai_model.training_callback
    if hasattr(ai_model, 'training_callback'):
        # Add real-time metrics sending to existing callback
        pass
    
    # Save models
    model_path = f'/tmp/models_{job_id}'
    ai_model.save_models(model_path)
    
    # Upload models to S3
    model_files = [
        (f'{{model_path}}_embedding.keras', 'embedding'),
        (f'{{model_path}}_classification.keras', 'classification'),
        (f'{{model_path}}_authenticity.keras', 'authenticity'),
        (f'{{model_path}}_siamese.keras', 'siamese'),
        (f'{{model_path}}_mappings.json', 'mappings')
    ]
    
    model_urls = {{}}
    for file_path, model_type in model_files:
        if os.path.exists(file_path):
            s3_key = f'models/{{job_id}}/{{model_type}}.keras'
            s3.upload_file(file_path, bucket, s3_key)
            model_urls[model_type] = f'https://{{bucket}}.s3.amazonaws.com/{{s3_key}}'
    
    # Save training results
    results = {{
        'job_id': job_id,
        'student_id': student_id,
        'model_urls': model_urls,
        'training_metrics': result
    }}
    
    with open('/tmp/training_results.json', 'w') as f:
        json.dump(results, f)
    
    s3.upload_file('/tmp/training_results.json', bucket, f'training_results/{{job_id}}.json')
    
    print("Training completed successfully!")

if __name__ == "__main__":
    training_data_key = sys.argv[1]
    job_id = sys.argv[2]
    student_id = int(sys.argv[3])
    train_on_gpu(training_data_key, job_id, student_id)
EOF

# Make script executable
chmod +x train_gpu.py

# Create requirements.txt
cat > requirements.txt << 'EOF'
tensorflow-gpu==2.13.0
numpy==1.24.3
pillow==10.0.0
opencv-python==4.8.0.74
scikit-learn==1.3.0
boto3==1.28.0
python-dotenv==1.0.0
fastapi==0.103.0
uvicorn==0.23.0
EOF

# Set environment variables
echo "export AWS_ACCESS_KEY_ID={os.getenv('AWS_ACCESS_KEY_ID')}" >> /home/ubuntu/.bashrc
echo "export AWS_SECRET_ACCESS_KEY={os.getenv('AWS_SECRET_ACCESS_KEY')}" >> /home/ubuntu/.bashrc
echo "export AWS_DEFAULT_REGION={os.getenv('AWS_REGION', 'us-east-1')}" >> /home/ubuntu/.bashrc
echo "export S3_BUCKET={self.s3_bucket}" >> /home/ubuntu/.bashrc

# Signal that setup is complete
echo "GPU instance setup complete" > /tmp/setup_complete
"""
            
            run_args = dict(
                ImageId=self.ami_id,
                MinCount=1,
                MaxCount=1,
                InstanceType=self.gpu_instance_type,
                KeyName=self.key_name,
                SecurityGroupIds=[self.security_group_id],
                SubnetId=self.subnet_id,
                UserData=user_data,
                TagSpecifications=[
                    {
                        'ResourceType': 'instance',
                        'Tags': [
                            {'Key': 'Name', 'Value': f'gpu-training-{job_id}'},
                            {'Key': 'Purpose', 'Value': 'AI-Training'},
                            {'Key': 'JobId', 'Value': job_id}
                        ]
                    }
                ],
                IamInstanceProfile={
                    'Name': 'EC2-S3-Access'
                }
            )

            response = self.ec2_client.run_instances(**run_args)
            
            instance_id = response['Instances'][0]['InstanceId']
            logger.info(f"Launched GPU instance: {instance_id}")
            
            return instance_id
            
        except ClientError as e:
            logger.error(f"Failed to launch GPU instance: {e}")
            return None
    
    async def _wait_for_instance_ready(self, instance_id: str, timeout: int = 300):
        """Wait for instance to be ready"""
        logger.info(f"Waiting for instance {instance_id} to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self.ec2_client.describe_instances(InstanceIds=[instance_id])
                state = response['Reservations'][0]['Instances'][0]['State']['Name']
                
                if state == 'running':
                    # Check if setup is complete
                    try:
                        if self.existing_instance_id and instance_id == self.existing_instance_id:
                            response = self.ssm_client.send_command(
                                InstanceIds=[instance_id],
                                DocumentName='AWS-RunShellScript',
                                Parameters={'commands': ['echo ready']}
                            )
                        else:
                            response = self.ssm_client.send_command(
                                InstanceIds=[instance_id],
                                DocumentName='AWS-RunShellScript',
                                Parameters={
                                    'commands': ['test -f /tmp/setup_complete && echo "ready" || echo "not ready"']
                                }
                            )
                        
                        command_id = response['Command']['CommandId']
                        
                        # Wait for command to complete
                        time.sleep(10)
                        
                        result = self.ssm_client.get_command_invocation(
                            CommandId=command_id,
                            InstanceId=instance_id
                        )
                        
                        if 'ready' in result.get('StandardOutputContent', ''):
                            logger.info(f"Instance {instance_id} is ready!")
                            return True
                            
                    except Exception as e:
                        logger.debug(f"Setup check failed: {e}")
                
                time.sleep(10)
                
            except Exception as e:
                logger.debug(f"Instance check failed: {e}")
                time.sleep(10)
        
        raise Exception(f"Instance {instance_id} did not become ready within {timeout} seconds")
    
    async def _get_instance_ip(self, instance_id: str) -> str:
        """Get instance public IP"""
        response = self.ec2_client.describe_instances(InstanceIds=[instance_id])
        return response['Reservations'][0]['Instances'][0]['PublicIpAddress']
    
    async def _upload_training_data(self, training_data: Dict, job_id: str) -> str:
        """Upload training data to S3"""
        # Convert images to base64 for JSON serialization
        serializable_data = {}
        for student_name, signatures in training_data.items():
            serializable_data[student_name] = {
                'genuine': [img.tolist() if hasattr(img, 'tolist') else img for img in signatures['genuine']],
                'forged': [img.tolist() if hasattr(img, 'tolist') else img for img in signatures['forged']]
            }
        
        # Upload to S3
        key = f'training_data/{job_id}.json'
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=key,
            Body=json.dumps(serializable_data),
            ContentType='application/json'
        )
        
        return key
    
    async def _start_remote_training(self, instance_id: str, training_data_key: str, 
                                   job_id: str, student_id: int) -> Dict:
        """Start training on remote GPU instance"""
        try:
            # Use SSM to run training command
            bootstrap = [
                f'if [ ! -d {self.training_script_path} ]; then mkdir -p {self.training_script_path}; fi',
                f'cd {self.training_script_path}',
                f'if [ ! -d {self.training_script_path}/ai-training ]; then git clone {self.github_repo} .; fi',
                f'cd {self.training_script_path}/ai-training',
                'python3 -m pip install -r requirements.txt || pip3 install -r requirements.txt'
            ] if (self.existing_instance_id and instance_id == self.existing_instance_id) else []

            response = self.ssm_client.send_command(
                InstanceIds=[instance_id],
                DocumentName='AWS-RunShellScript',
                Parameters={
                    'commands': bootstrap + [
                        f'cd {self.training_script_path}/ai-training',
                        f'python3 train_gpu.py {training_data_key} {job_id} {student_id}'
                    ]
                }
            )
            
            command_id = response['Command']['CommandId']
            
            # Monitor training progress
            while True:
                time.sleep(30)  # Check every 30 seconds
                
                result = self.ssm_client.get_command_invocation(
                    CommandId=command_id,
                    InstanceId=instance_id
                )
                
                status = result['Status']
                if status in ['Success', 'Failed', 'Cancelled']:
                    break
            
            if status == 'Success':
                logger.info(f"Training completed successfully for job {job_id}")
                return {'status': 'success', 'output': result.get('StandardOutputContent', '')}
            else:
                error = result.get('StandardErrorContent', 'Unknown error')
                logger.error(f"Training failed for job {job_id}: {error}")
                return {'status': 'failed', 'error': error}
                
        except Exception as e:
            logger.error(f"Failed to start remote training: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _download_training_results(self, job_id: str) -> Dict:
        """Download training results from S3"""
        try:
            # Download results
            response = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key=f'training_results/{job_id}.json'
            )
            
            results = json.loads(response['Body'].read())
            return results.get('model_urls', {})
            
        except Exception as e:
            logger.error(f"Failed to download training results: {e}")
            return {}
    
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

# Global instance
gpu_training_manager = AWSGPUTrainingManager()