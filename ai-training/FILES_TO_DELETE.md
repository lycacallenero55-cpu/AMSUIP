# Files and Directories to Delete (Not Needed for EC2-Only Training)

## Docker/WSL Related Files to DELETE:

### Docker Setup Files
- `docker/` (entire directory)
  - `docker/setup_training_container.sh`
  - `docker/setup_training_container.bat`
  - `docker/README.md`
  
### Docker Configuration Files
- `Dockerfile` (if exists)
- `docker-compose.yml` (if exists)
- `.dockerignore` (if exists)

### WSL Setup Files
- `setup_wsl_docker.sh`
- `WSL_SETUP_INSTRUCTIONS.md`
- `.wslconfig` (if exists)

### Docker Training Managers (DELETE these)
- `utils/docker_training.py`
- `utils/docker_training_production.py`

### Docker Documentation
- `DOCKER_SETUP_GUIDE.md`

### Verification Scripts (for Docker)
- `verify_docker_setup.py`

## Files to KEEP:

### EC2 Training Files (KEEP these)
- `utils/aws_gpu_training.py` ✅
- `scripts/train_gpu_template.py` ✅
- `scripts/setup_gpu_instance.sh` ✅

### Core Application Files (KEEP these)
- `main.py` ✅
- `config.py` ✅
- `.env` ✅
- `requirements.txt` ✅
- All files in `api/` directory ✅
- All files in `models/` directory ✅
- All files in `services/` directory ✅

## Summary

You can safely delete all Docker and WSL related files since you're using EC2 exclusively for training. The system will now:

1. **Always use EC2** for training (no local Docker)
2. **Connect via SSH/SCP** to your EC2 instance
3. **Execute training** in the Docker container on EC2
4. **Return results** via S3

Total files/directories to delete: ~15 items
Space saved: ~100KB of unnecessary code
