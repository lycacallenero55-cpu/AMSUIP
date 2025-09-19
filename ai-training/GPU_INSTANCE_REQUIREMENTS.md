# GPU Instance Pre-Configuration Requirements

To skip the setup phase and use a pre-configured GPU instance, ensure the following dependencies are installed:

## System Requirements

### 1. Operating System
- Ubuntu 20.04 or 22.04 LTS
- NVIDIA GPU with CUDA support

### 2. NVIDIA Drivers & CUDA
```bash
# NVIDIA Driver (version 525 or higher)
sudo apt-get update
sudo apt-get install -y nvidia-driver-525

# CUDA Toolkit (11.8 or compatible with TensorFlow 2.15)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# cuDNN (8.6 or compatible)
# Download from NVIDIA website and install
```

### 3. Python & Core Tools
```bash
# Python 3.9 or 3.10
sudo apt-get install -y python3.10 python3.10-pip python3.10-venv

# Essential tools
sudo apt-get install -y git curl wget unzip build-essential
```

### 4. AWS Tools
```bash
# AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# SSM Agent (for remote command execution)
sudo snap install amazon-ssm-agent --classic
sudo systemctl enable snap.amazon-ssm-agent.amazon-ssm-agent.service
sudo systemctl start snap.amazon-ssm-agent.amazon-ssm-agent.service
```

## Python Dependencies

Install these Python packages globally or in a virtual environment:

```bash
pip3 install --upgrade pip

# Core ML/AI packages
pip3 install tensorflow==2.15.*  # Must be 2.15.x for compatibility
pip3 install tensorflow-gpu==2.15.*  # GPU support

# Image processing
pip3 install pillow==10.1.*
pip3 install opencv-python==4.8.*

# Scientific computing
pip3 install numpy==1.24.*
pip3 install scipy==1.11.*
pip3 install scikit-learn==1.3.*

# AWS integration
pip3 install boto3==1.28.*
pip3 install botocore==1.31.*

# Utilities
pip3 install pandas==2.1.*
pip3 install matplotlib==3.7.*
```

## Directory Structure

Create the following directories:

```bash
# Training workspace
sudo mkdir -p /home/ubuntu/ai-training
sudo chown ubuntu:ubuntu /home/ubuntu/ai-training

# Temp directory for models
sudo mkdir -p /tmp/ai-models
sudo chmod 777 /tmp/ai-models
```

## IAM Role Configuration

Ensure the EC2 instance has an IAM role with the following permissions:
- S3 read/write access to your training bucket
- SSM permissions for remote command execution
- CloudWatch logs (optional but recommended)

Example IAM policy:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket-name/*",
        "arn:aws:s3:::your-bucket-name"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "ssm:UpdateInstanceInformation",
        "ssmmessages:*",
        "ec2messages:*"
      ],
      "Resource": "*"
    }
  ]
}
```

## Configuration Settings

In your `config.py` or environment variables, set:

```python
# Skip the setup phase
SKIP_GPU_SETUP = True

# Use existing pre-configured instance
EXISTING_GPU_INSTANCE_ID = "i-xxxxxxxxxx"  # Your instance ID

# AWS Configuration
AWS_REGION = "us-east-1"
S3_BUCKET_NAME = "your-bucket-name"
```

## Verification

To verify the instance is properly configured:

```bash
# Check GPU availability
nvidia-smi

# Check CUDA
nvcc --version

# Check Python and TensorFlow
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Check AWS CLI
aws --version
aws s3 ls  # Should list your buckets if IAM is configured

# Check SSM Agent
sudo systemctl status snap.amazon-ssm-agent.amazon-ssm-agent.service
```

## Notes

1. **TensorFlow Version**: Must be 2.15.x for compatibility with the training scripts
2. **CUDA Compatibility**: Ensure CUDA version matches TensorFlow requirements
3. **Memory**: GPU should have at least 8GB VRAM for efficient training
4. **Storage**: At least 50GB free space for models and temporary files
5. **Network**: Ensure security groups allow SSM connections

With these pre-configurations, the training script will:
- Skip all setup steps
- Only upload the training script
- Start training immediately
- Save significant time on repeated training runs
