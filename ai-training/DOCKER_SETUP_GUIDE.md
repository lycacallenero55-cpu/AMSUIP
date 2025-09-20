# Production-Ready Docker Training Setup Guide

## Overview
This guide provides complete instructions for setting up and using the Docker-based AI training system. All training runs inside a Docker container with GPU support, ensuring consistent and isolated execution.

## System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Frontend UI   │────▶│  FastAPI Server  │────▶│ Docker Manager  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
                                                  ┌─────────────────┐
                                                  │ Docker Container│
                                                  │   - Python 3.11 │
                                                  │   - TF 2.15 GPU │
                                                  │   - Training    │
                                                  └─────────────────┘
```

## Prerequisites

1. **Docker Engine** (20.10+)
2. **NVIDIA GPU** (optional, for GPU acceleration)
3. **NVIDIA Container Toolkit** (for GPU support)
4. **Python 3.8+** (for the FastAPI server)

## Quick Start

### Step 1: Configure Environment

Edit `.env` file with your settings:

```env
# Docker Training Configuration
USE_DOCKER_TRAINING=true
DOCKER_CONTAINER_NAME=ai-training-container
DOCKER_CONTAINER_IMAGE=tensorflow/tensorflow:2.15.0-gpu
HOST_MODELS_DIR=/home/ubuntu/ai-training/models
HOST_DATA_DIR=/home/ubuntu/ai-training/training_data

# AWS S3 Configuration
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_REGION=us-east-1
S3_BUCKET=your-bucket-name
```

### Step 2: Create Docker Container

Since you already have a pre-configured container, ensure it's running:

```bash
# Check if container exists
docker ps -a | grep ai-training-container

# If not running, start it
docker start ai-training-container

# Verify it's healthy
docker exec ai-training-container python3 -c "import tensorflow as tf; print(tf.__version__)"
```

### Step 3: Verify Setup

Run the verification script:

```bash
cd ai-training
python verify_docker_setup.py
```

Expected output:
```
✅ Environment: PASSED
✅ Docker Installation: PASSED
✅ Container: PASSED
✅ Directories: PASSED
✅ Training Manager: PASSED
```

### Step 4: Start the Server

```bash
# Activate virtual environment
venv/Scripts/activate  # Windows
source venv/bin/activate  # Linux/Mac

# Start the server
python main.py
```

The server will start on `http://localhost:8000`

## How It Works

### Training Flow

1. **User clicks "Train"** in the UI
2. **Server receives request** with training data
3. **Docker manager**:
   - Ensures container is running
   - Prepares training data as JSON
   - Saves to `/home/ubuntu/ai-training/training_data/`
4. **Training execution**:
   - Runs `/workspace/train_gpu_template.py` in container
   - Monitors progress via stdout
   - Updates UI in real-time
5. **Results retrieval**:
   - Models saved to `/home/ubuntu/ai-training/models/`
   - Uploaded to S3 if configured
   - Cleanup temporary files

### Directory Structure

```
/home/ubuntu/ai-training/
├── models/                 # Trained models output
│   ├── job_id_1/
│   │   ├── signature_model_embedding.keras
│   │   ├── signature_model_classification.keras
│   │   └── training_results.json
│   └── job_id_2/
├── training_data/          # Training input data
│   ├── training_data_job1.json
│   └── training_data_job2.json
└── scripts/
    └── train_gpu_template.py  # Training script
```

## Container Management

### Manual Container Creation

If you need to create a new container:

```bash
docker run -d \
  --name ai-training-container \
  --gpus all \
  --restart unless-stopped \
  -v $(pwd)/scripts:/workspace \
  -v /home/ubuntu/ai-training/models:/models \
  -v /home/ubuntu/ai-training/training_data:/training_data \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e TF_FORCE_GPU_ALLOW_GROWTH=true \
  tensorflow/tensorflow:2.15.0-gpu \
  tail -f /dev/null

# Install dependencies
docker exec ai-training-container pip install \
  pillow numpy scipy scikit-learn boto3
```

### Container Commands

```bash
# View logs
docker logs ai-training-container

# Enter container shell
docker exec -it ai-training-container bash

# Stop container
docker stop ai-training-container

# Remove container
docker rm -f ai-training-container

# Check GPU availability
docker exec ai-training-container nvidia-smi
```

## API Endpoints

### Start Training
```http
POST /api/training/start-async
Content-Type: multipart/form-data

student_id: "12345"
genuine_files: [file1.jpg, file2.jpg, ...]
forged_files: []
```

### Check Progress
```http
GET /api/progress/stream/{job_id}
```

### Get Results
```http
GET /api/training/models?student_id=12345
```

## Monitoring & Debugging

### Check Training Logs

```bash
# Container logs
docker logs -f ai-training-container

# Training output
docker exec ai-training-container cat /workspace/logs/training.log

# Server logs
tail -f ai-training/logs/server.log
```

### Common Issues & Solutions

#### Container Not Starting
```bash
# Check Docker daemon
sudo systemctl status docker

# Check container status
docker ps -a

# View error logs
docker logs ai-training-container
```

#### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi

# Check container GPU access
docker exec ai-training-container python3 -c \
  "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

#### Training Fails
```bash
# Check training script syntax
docker exec ai-training-container python3 -m py_compile /workspace/train_gpu_template.py

# Check data file
docker exec ai-training-container ls -la /training_data/

# Run test training
docker exec ai-training-container python3 /workspace/train_gpu_template.py \
  --data /training_data/test.json \
  --job_id test_job \
  --epochs 1
```

#### Permission Issues
```bash
# Fix directory permissions
sudo chown -R $USER:$USER /home/ubuntu/ai-training/
chmod -R 755 /home/ubuntu/ai-training/
```

## Performance Optimization

### GPU Memory
```python
# In train_gpu_template.py
import tensorflow as tf

# Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

### Batch Size
Adjust in `config.py`:
```python
MODEL_BATCH_SIZE = 16  # Reduce if OOM errors
```

### Parallel Processing
The system automatically uses:
- Async training execution
- Parallel data preprocessing
- Concurrent S3 uploads

## Security Best Practices

1. **Never commit `.env` file** to version control
2. **Use IAM roles** for AWS access when possible
3. **Restrict container capabilities**:
   ```bash
   docker run --cap-drop=ALL --cap-add=SYS_PTRACE ...
   ```
4. **Use secrets management** for production
5. **Enable TLS** for API endpoints
6. **Implement rate limiting** on training endpoints

## Production Deployment

### Using Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  ai-training:
    image: tensorflow/tensorflow:2.15.0-gpu
    container_name: ai-training-container
    restart: unless-stopped
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - TF_FORCE_GPU_ALLOW_GROWTH=true
    volumes:
      - ./scripts:/workspace
      - /home/ubuntu/ai-training/models:/models
      - /home/ubuntu/ai-training/training_data:/training_data
    command: tail -f /dev/null

  api-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - USE_DOCKER_TRAINING=true
      - DOCKER_CONTAINER_NAME=ai-training-container
    volumes:
      - .:/app
    depends_on:
      - ai-training
```

### Scaling Considerations

- Use **multiple containers** for parallel training
- Implement **job queue** with Redis/RabbitMQ
- Use **object storage** (S3) for model distribution
- Consider **Kubernetes** for orchestration

## Troubleshooting Checklist

- [ ] Docker daemon running?
- [ ] Container exists and running?
- [ ] Directories exist with correct permissions?
- [ ] Environment variables set correctly?
- [ ] Training script present in container?
- [ ] Python dependencies installed?
- [ ] GPU drivers and CUDA working?
- [ ] Sufficient disk space?
- [ ] Network connectivity for S3?
- [ ] Correct Python version in container?

## Support

For issues:
1. Run `python verify_docker_setup.py`
2. Check logs: `docker logs ai-training-container`
3. Review this guide's troubleshooting section
4. Check container health: `docker exec ai-training-container python3 -c "print('OK')"`

## Summary

The system is designed to be:
- **Isolated**: All training in containers
- **Reproducible**: Same environment every time
- **Scalable**: Easy to add more containers
- **Robust**: Automatic error recovery
- **Production-ready**: Full logging and monitoring

Training will automatically use the Docker container when `USE_DOCKER_TRAINING=true` is set.
