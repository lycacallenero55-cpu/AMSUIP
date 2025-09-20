@echo off
REM Docker Container Setup Script for AI Training (Windows)
REM This script creates a Docker container with all required dependencies for training

set CONTAINER_NAME=ai-training-gpu
set IMAGE_NAME=tensorflow/tensorflow:2.15.0-gpu
set WORKSPACE_DIR=/workspace

echo Setting up Docker container for AI training...

REM Check if container already exists
docker ps -aq -f name=%CONTAINER_NAME% >nul 2>&1
if %errorlevel% == 0 (
    echo Container %CONTAINER_NAME% already exists. Removing old container...
    docker rm -f %CONTAINER_NAME%
)

REM Create the container with GPU support
echo Creating container %CONTAINER_NAME%...
docker run -d ^
    --name %CONTAINER_NAME% ^
    --gpus all ^
    --restart unless-stopped ^
    -v %cd%\scripts:/workspace ^
    -v %cd%\models:/models ^
    -v %cd%\training_data:/training_data ^
    -e NVIDIA_VISIBLE_DEVICES=all ^
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility ^
    -e TF_FORCE_GPU_ALLOW_GROWTH=true ^
    %IMAGE_NAME% ^
    tail -f /dev/null

REM Wait for container to start
timeout /t 3 /nobreak >nul

REM Install additional Python packages
echo Installing Python dependencies...
docker exec %CONTAINER_NAME% pip install --upgrade pip
docker exec %CONTAINER_NAME% pip install ^
    pillow==10.1.0 ^
    numpy==1.24.3 ^
    scipy==1.11.4 ^
    scikit-learn==1.3.2 ^
    pandas==2.1.4 ^
    matplotlib==3.7.4 ^
    opencv-python==4.8.1.78 ^
    boto3==1.28.85 ^
    botocore==1.31.85

REM Copy training script to container
echo Copying training script...
docker cp scripts\train_gpu_template.py %CONTAINER_NAME%:%WORKSPACE_DIR%/train_gpu_template.py

REM Create necessary directories in container
echo Creating directories...
docker exec %CONTAINER_NAME% mkdir -p /tmp/ai-models
docker exec %CONTAINER_NAME% mkdir -p /workspace/logs

REM Verify GPU is accessible
echo Verifying GPU access...
docker exec %CONTAINER_NAME% python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'Found {len(gpus)} GPU(s)') if gpus else print('No GPU found - will use CPU')"

REM Test TensorFlow installation
echo Testing TensorFlow...
docker exec %CONTAINER_NAME% python3 -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print(f'Built with CUDA: {tf.test.is_built_with_cuda()}'); print(f'GPU available: {tf.test.is_gpu_available()}')"

echo.
echo ✅ Docker container setup complete!
echo.
echo Container name: %CONTAINER_NAME%
echo To view logs: docker logs %CONTAINER_NAME%
echo To enter container: docker exec -it %CONTAINER_NAME% bash
echo To stop container: docker stop %CONTAINER_NAME%
echo To start container: docker start %CONTAINER_NAME%
echo.
echo The training system will now automatically use this container when you click 'Train'.
pause
