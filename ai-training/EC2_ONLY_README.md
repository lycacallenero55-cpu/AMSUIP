# EC2-Only AI Training System

This system is configured to use **ONLY** your EC2 GPU instance for training. No local Docker or WSL is used.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   Frontend UI   │────▶│  FastAPI Server  │────▶│ EC2 GPU Instance    │
│   (Local)       │     │   (Local)        │     │ (Remote Docker)     │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
                                │                           │
                                │  SSH/SCP                  │
                                └──────────────────────────▶│
                                                   ┌─────────────────────┐
                                                   │ Docker Container    │
                                                   │ - TensorFlow 2.15   │
                                                   │ - GPU Training      │
                                                   └─────────────────────┘
```

## Setup Instructions

### 1. Configure EC2 Instance Details

Your `.env` file is already configured with:

```env
# GPU Training Configuration (Already in your .env)
AWS_KEY_NAME=gpu-training-key
AWS_SECURITY_GROUP_ID=sg-0b99dfa9c4107a3b8 
AWS_SUBNET_ID=subnet-021ae1656852b0225 
AWS_GPU_EXISTING_INSTANCE_ID=i-0756716b845e4c314
AWS_GPU_AMI_ID=ami-0bbdd8c17ed981ef9
AWS_IAM_INSTANCE_PROFILE=EC2-S3-Access
AWS_GPU_INSTANCE_TYPE=g4dn.xlarge

# AWS S3 Configuration
AWS_ACCESS_KEY_ID=AKIAVPBTCU5RY67EFWCL
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1
S3_BUCKET=signatureai-uploads
```

### 2. Install Dependencies

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install requirements
pip install -r requirements.txt
```

### 3. Test Configuration

```bash
# Run the test script
python test_ec2_setup.py
```

Expected output:
```
✅ Config imported successfully
✅ AWS GPU training manager imported successfully
✅ Training API imported successfully
✅ EC2 Instance ID: i-0abc123def456789
✅ GPU training is available
```

### 4. Start the Server

```bash
python main.py
```

The server will start on `http://localhost:8000`

## How It Works

1. **User clicks "Train"** in the UI
2. **FastAPI server** receives the training request
3. **AWS GPU Manager**:
   - Connects to your EC2 instance via SSH
   - Uploads training data via SCP
   - Executes training in the Docker container on EC2
4. **Training runs** entirely on EC2 GPU
5. **Results** are saved to S3 and returned to UI

## EC2 Instance Requirements

Your EC2 instance must have:

- **Instance Type**: g4dn.xlarge (or similar GPU instance)
- **Docker installed** with the training container running
- **Container Name**: `ai-training-container`
- **SSH access** enabled (port 22)
- **Security Group** allows SSH from your IP
- **IAM Role** with S3 access (optional)

### Docker Container on EC2

The container on your EC2 instance should have:
- TensorFlow 2.15 with GPU support
- Python 3.11
- Training script at `/workspace/train_gpu_template.py`
- Required Python packages (numpy, pandas, boto3, etc.)

## API Endpoints

### Start Training
```http
POST /api/training/start-gpu-training
Content-Type: multipart/form-data

student_id: "12345"
genuine_files: [file1.jpg, file2.jpg, ...]
use_gpu: true
```

### Check GPU Availability
```http
GET /api/training/gpu-available
```

### Get Training Progress
```http
GET /api/progress/stream/{job_id}
```

## Files Deleted (No Longer Needed)

The following files have been removed since we're not using local Docker:

- ❌ `docker/` directory
- ❌ `setup_wsl_docker.sh`
- ❌ `WSL_SETUP_INSTRUCTIONS.md`
- ❌ `DOCKER_SETUP_GUIDE.md`
- ❌ `verify_docker_setup.py`
- ❌ `utils/docker_training.py`
- ❌ `utils/docker_training_production.py`

## Monitoring

### Check EC2 Instance Status
```python
from utils.aws_gpu_training import gpu_training_manager
print(gpu_training_manager.is_available())
```

### View Training Logs
Training logs are stored in:
- EC2: `/tmp/training_logs/`
- S3: `s3://your-bucket/training_logs/`

## Troubleshooting

### GPU Manager Not Available
```bash
# Check EC2 instance is running
aws ec2 describe-instances --instance-ids i-your-instance-id

# Check SSH connectivity
ssh -i your-key.pem ubuntu@ec2-ip-address
```

### Training Fails
1. Check EC2 instance is running
2. Verify Docker container is running on EC2
3. Check S3 permissions
4. Review logs in `/tmp/training_logs/`

### Connection Issues
- Verify security group allows SSH (port 22)
- Check EC2 instance public IP
- Ensure SSH key has correct permissions (400)

## Performance

- **Training Speed**: 10-50x faster than CPU
- **GPU Memory**: 16GB (g4dn.xlarge)
- **Batch Size**: Adjust in `config.py` based on model
- **Parallel Training**: Multiple jobs queue automatically

## Cost Optimization

- **Auto-stop**: Instance stops after idle timeout
- **Spot Instances**: Use for 70% cost savings
- **Reserved Instances**: For consistent usage
- **Monitoring**: Track usage in AWS Cost Explorer

## Summary

✅ **No local Docker/WSL required**
✅ **All training on EC2 GPU**
✅ **Simple configuration**
✅ **Production ready**
✅ **Cost effective**

The system now exclusively uses your EC2 GPU instance for all training operations!
