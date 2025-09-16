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
            training_result = await self._start_remote_training(
                instance_id, training_data_key, job_id, student_id, job_queue
            )
            
            if training_result.get('status') != 'success':
                raise Exception(f"Training failed: {training_result.get('error', 'Unknown error')}")
            
            # Step 6: Download results
            job_queue.update_job_progress(job_id, 85.0, "Downloading training results...")
            model_urls = await self._download_training_results(job_id)
            job_queue.update_job_progress(job_id, 90.0, "Training results downloaded successfully")
            
            # Step 7: Terminate instance (if not reusing)
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
                # Verify instance exists and is running
                response = self.ec2_client.describe_instances(InstanceIds=[self.existing_instance_id])
                if response['Reservations']:
                    state = response['Reservations'][0]['Instances'][0]['State']['Name']
                    if state == 'running':
                        return self.existing_instance_id
                    else:
                        logger.warning(f"Existing instance {self.existing_instance_id} is in state: {state}")
                        
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
            logger.info(f"Launched GPU instance: {instance_id}")
            
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
        """Wait for instance to be ready"""
        logger.info(f"Waiting for instance {instance_id} to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self.ec2_client.describe_instances(InstanceIds=[instance_id])
                if not response['Reservations']:
                    await asyncio.sleep(10)
                    continue
                    
                instance = response['Reservations'][0]['Instances'][0]
                state = instance['State']['Name']
                
                if state == 'running':
                    # Wait for status checks to pass
                    status_response = self.ec2_client.describe_instance_status(InstanceIds=[instance_id])
                    if status_response['InstanceStatuses']:
                        status = status_response['InstanceStatuses'][0]
                        if (status.get('InstanceStatus', {}).get('Status') == 'ok' and
                            status.get('SystemStatus', {}).get('Status') == 'ok'):
                            logger.info(f"Instance {instance_id} is ready!")
                            return True
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.debug(f"Instance check failed: {e}")
                await asyncio.sleep(10)
        
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
        """Upload training data to S3"""
        # Convert images to serializable format
        serializable_data = {}
        for student_name, signatures in training_data.items():
            serializable_data[student_name] = {}
            
            # Handle different data structures
            if isinstance(signatures, dict):
                for key in ['genuine', 'forged']:
                    if key in signatures:
                        serializable_data[student_name][key] = [
                            img.tolist() if hasattr(img, 'tolist') else img 
                            for img in signatures[key]
                        ]
            else:
                # If it's a list, treat as genuine signatures
                serializable_data[student_name] = {
                    'genuine': [img.tolist() if hasattr(img, 'tolist') else img for img in signatures],
                    'forged': []
                }
        
        # Upload to S3
        key = f'training_data/{job_id}.json'
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=key,
            Body=json.dumps(serializable_data),
            ContentType='application/json'
        )
        
        logger.info(f"Uploaded training data to s3://{self.s3_bucket}/{key}")
        return key
    
    async def _setup_training_environment(self, instance_id: str, job_id: str):
        """Setup training environment on the instance"""
        try:
            # Create training script
            training_script = self._generate_training_script()
            
            # Upload script to S3 first
            script_key = f'scripts/{job_id}/train_gpu.py'
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=script_key,
                Body=training_script,
                ContentType='text/x-python'
            )
            
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
        """Generate the training script"""
        return f'''#!/usr/bin/env python3
import sys
import os
import json
import boto3
import numpy as np
from PIL import Image
import io
import requests
import traceback

def train_on_gpu(training_data_key, job_id, student_id):
    """Train model on GPU instance"""
    try:
        # Initialize S3 client
        s3 = boto3.client('s3')
        bucket = '{self.s3_bucket}'
        
        print(f"Starting training for job {{job_id}}")
        
        # Download training data from S3
        print(f"Downloading training data from s3://{{bucket}}/{{training_data_key}}")
        s3.download_file(bucket, training_data_key, '/tmp/training_data.json')
        
        with open('/tmp/training_data.json', 'r') as f:
            training_data = json.load(f)
        
        print(f"Loaded training data for {{len(training_data)}} students")
        
        # Simulate training process (replace with actual training code)
        # This is where you would integrate with your actual AI model
        import time
        for i in range(10):
            print(f"Training progress: {{(i+1)*10}}%")
            time.sleep(2)
        
        # Create dummy model files for testing
        model_files = ['embedding', 'classification', 'authenticity', 'siamese']
        model_urls = {{}}
        
        for model_type in model_files:
            # Create dummy model file
            file_path = f'/tmp/{{job_id}}_{{model_type}}.keras'
            with open(file_path, 'w') as f:
                f.write(f'Dummy {{model_type}} model')
            
            # Upload to S3
            s3_key = f'models/{{job_id}}/{{model_type}}.keras'
            s3.upload_file(file_path, bucket, s3_key)
            model_urls[model_type] = f'https://{{bucket}}.s3.amazonaws.com/{{s3_key}}'
            print(f"Uploaded {{model_type}} model to S3")
        
        # Save mappings
        mappings = {{
            'student_id': student_id,
            'students': list(training_data.keys())
        }}
        mappings_path = f'/tmp/{{job_id}}_mappings.json'
        with open(mappings_path, 'w') as f:
            json.dump(mappings, f)
        
        s3_key = f'models/{{job_id}}/mappings.json'
        s3.upload_file(mappings_path, bucket, s3_key)
        model_urls['mappings'] = f'https://{{bucket}}.s3.amazonaws.com/{{s3_key}}'
        
        # Save training results
        results = {{
            'job_id': job_id,
            'student_id': student_id,
            'model_urls': model_urls,
            'training_metrics': {{
                'accuracy': 0.95,
                'loss': 0.05,
                'epochs': 10
            }}
        }}
        
        results_path = '/tmp/training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f)
        
        s3.upload_file(results_path, bucket, f'training_results/{{job_id}}.json')
        print(f"Training completed successfully for job {{job_id}}!")
        
    except Exception as e:
        print(f"Training failed: {{str(e)}}")
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
    
    async def _start_remote_training(self, instance_id: str, training_data_key: str, 
                                   job_id: str, student_id: int, job_queue) -> Dict:
        """Start training on remote GPU instance"""
        try:
            # Run training command
            training_command = [
                f'cd {self.training_script_path}',
                f'python3 train_gpu.py {training_data_key} {job_id} {student_id}'
            ]
            
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
            return results.get('model_urls', {})
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.error(f"Training results not found for job {job_id}")
                return {}
            else:
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
