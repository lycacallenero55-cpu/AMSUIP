# Docker-based AI Training Setup

This directory contains scripts to set up a Docker container for AI training with GPU support.

## Prerequisites

1. **Docker Desktop** installed with WSL2 backend (Windows) or Docker Engine (Linux)
2. **NVIDIA GPU** with CUDA support
3. **NVIDIA Container Toolkit** installed
4. **NVIDIA Drivers** (version 525 or higher)

## Quick Setup

### Windows
```batch
cd ai-training\docker
setup_training_container.bat
```

### Linux/Mac
```bash
cd ai-training/docker
chmod +x setup_training_container.sh
./setup_training_container.sh
```

## What the Setup Does

1. **Creates a Docker container** named `ai-training-gpu` with:
   - TensorFlow 2.15 with GPU support
   - Python 3.11
   - All required ML libraries (NumPy, Pandas, Scikit-learn, etc.)
   - Image processing libraries (Pillow, OpenCV)
   - AWS integration (Boto3)

2. **Mounts directories** for:
   - `/workspace` - Training scripts
   - `/models` - Saved models
   - `/training_data` - Training data

3. **Configures GPU access** with:
   - All GPUs visible to container
   - GPU memory growth enabled
   - CUDA compute capabilities

## Container Management

### Start the container
```bash
docker start ai-training-gpu
```

### Stop the container
```bash
docker stop ai-training-gpu
```

### View container logs
```bash
docker logs ai-training-gpu
```

### Enter the container shell
```bash
docker exec -it ai-training-gpu bash
```

### Remove the container
```bash
docker rm -f ai-training-gpu
```

## Configuration

The system is configured via environment variables in `.env`:

```env
# Enable Docker training
USE_DOCKER_TRAINING=true

# Container settings
DOCKER_CONTAINER_NAME=ai-training-gpu
DOCKER_CONTAINER_IMAGE=tensorflow/tensorflow:2.15.0-gpu

# Host directories
HOST_MODELS_DIR=./models
HOST_DATA_DIR=./training_data
```

## How It Works

When you click the "Train" button in the UI:

1. **Training data** is prepared and saved to `./training_data`
2. **Docker container** is started automatically (if not running)
3. **Data is copied** into the container
4. **Training script** (`train_gpu_template.py`) runs inside the container
5. **Progress updates** are streamed back to the UI
6. **Trained models** are saved to `./models`
7. **Cleanup** removes temporary files

## Advantages

- **Isolated Environment**: Training runs in a controlled container
- **No Host Dependencies**: All packages are in the container
- **GPU Acceleration**: Full CUDA/cuDNN support
- **Reproducible**: Same environment every time
- **Easy Updates**: Just rebuild the container

## Troubleshooting

### GPU not detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Container won't start
```bash
# Check if port is in use
docker ps -a

# Remove old container
docker rm -f ai-training-gpu

# Recreate container
./setup_training_container.sh
```

### Training fails
```bash
# Check container logs
docker logs ai-training-gpu

# Enter container to debug
docker exec -it ai-training-gpu bash
python3 /workspace/train_gpu_template.py --test
```

## Custom Container

To use your own pre-built container:

1. Build your container with all dependencies
2. Update `.env`:
   ```env
   DOCKER_CONTAINER_IMAGE=your-image:tag
   ```
3. Ensure your container has:
   - Python 3.10+
   - TensorFlow 2.15
   - Required libraries (see setup script)
   - `/workspace/train_gpu_template.py`
   - `/tmp/ai-models` directory

## Performance Tips

1. **GPU Memory**: Set `TF_FORCE_GPU_ALLOW_GROWTH=true` to prevent OOM
2. **Batch Size**: Adjust in `config.py` based on GPU memory
3. **Mixed Precision**: Enable for faster training on newer GPUs
4. **Data Pipeline**: Use tf.data for efficient data loading

## Security Notes

- Container runs with minimal privileges
- No network access required during training
- Sensitive data stays in mounted volumes
- AWS credentials are not stored in container
