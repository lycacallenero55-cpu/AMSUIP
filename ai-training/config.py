import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Supabase Configuration
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    # AWS / S3 Configuration
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET = os.getenv("S3_BUCKET")
    # If your bucket is public, this will be the base for URLs; otherwise presign GETs
    S3_PUBLIC_BASE_URL = os.getenv("S3_PUBLIC_BASE_URL", None)
    S3_USE_PRESIGNED_GET = os.getenv("S3_USE_PRESIGNED_GET", "false").lower() == "true"
    
    # Server Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # Model Configuration
    MODEL_IMAGE_SIZE = int(os.getenv("MODEL_IMAGE_SIZE", 224))
    MODEL_BATCH_SIZE = int(os.getenv("MODEL_BATCH_SIZE", 16))  # Smaller batch size for small datasets
    MODEL_EPOCHS = int(os.getenv("MODEL_EPOCHS", 50))  # More epochs for better convergence with small datasets
    MODEL_LEARNING_RATE = float(os.getenv("MODEL_LEARNING_RATE", 0.0001))  # Lower learning rate for transfer learning
    MODEL_FINE_TUNE_EPOCHS = int(os.getenv("MODEL_FINE_TUNE_EPOCHS", 10))  # Additional epochs for fine-tuning

    # CPU Optimization
    USE_CPU_OPTIMIZATION: bool = True
    CPU_THREADS: int = 6  # For Ryzen 5 3400G
    
    # Storage Configuration
    LOCAL_MODELS_DIR = os.getenv("LOCAL_MODELS_DIR", "./models")
    SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "models")
    
    # Training Configuration
    MIN_GENUINE_SAMPLES = int(os.getenv("MIN_GENUINE_SAMPLES", 3))  # Reduced for minimal requirements like Teachable Machine
    MIN_FORGED_SAMPLES = int(os.getenv("MIN_FORGED_SAMPLES", 0))   # Disabled - no forgery detection needed
    MAX_TRAINING_TIME = int(os.getenv("MAX_TRAINING_TIME", 3600))

    # Verification Settings
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.7
    USE_ADAPTIVE_THRESHOLD: bool = False
    
    # Confidence Thresholds for Owner Detection
    CONFIDENCE_THRESHOLD: float = 0.6  # Minimum confidence for positive match
    ENABLE_FORGERY_DETECTION: bool = False  # Disabled for owner detection focus
    
    # Anti-Spoofing (disabled to prioritize identification flow)
    ENABLE_ANTISPOOFING: bool = False
    SPOOFING_THRESHOLD: float = 0.6
    
    # Training Configuration - Focus on Owner Identification Only
    MIN_FORGED_SAMPLES = 0  # Disabled - no forgery detection needed
    ENABLE_FORGERY_TRAINING: bool = False  # Disabled system-wide
    
    # Model Versioning
    ENABLE_MODEL_VERSIONING: bool = True
    MAX_MODEL_VERSIONS: int = 5
    
    # Performance Monitoring
    ENABLE_PERFORMANCE_MONITORING: bool = True
    LOG_LEVEL: str = "INFO"
    
    # Docker Training Configuration
    USE_DOCKER_TRAINING: bool = os.getenv("USE_DOCKER_TRAINING", "false").lower() == "true"
    DOCKER_CONTAINER_NAME: str = os.getenv("DOCKER_CONTAINER_NAME", "ai-training-gpu")
    DOCKER_CONTAINER_IMAGE: str = os.getenv("DOCKER_CONTAINER_IMAGE", "tensorflow/tensorflow:2.15.0-gpu")
    HOST_MODELS_DIR: str = os.getenv("HOST_MODELS_DIR", "./models")
    HOST_DATA_DIR: str = os.getenv("HOST_DATA_DIR", "./training_data")
    
    # AWS GPU Training Configuration (optional, if not using Docker)
    GPU_INSTANCE_TYPE: str = os.getenv("GPU_INSTANCE_TYPE", "g4dn.xlarge")
    GPU_AMI_ID: str = os.getenv("GPU_AMI_ID", "ami-0c55b159cbfafe1f0")
    EC2_KEY_NAME: str = os.getenv("EC2_KEY_NAME", "")
    EC2_SECURITY_GROUP_ID: str = os.getenv("EC2_SECURITY_GROUP_ID", "")
    EC2_SUBNET_ID: str = os.getenv("EC2_SUBNET_ID", "")
    EC2_IAM_INSTANCE_PROFILE: str = os.getenv("EC2_IAM_INSTANCE_PROFILE", "")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", S3_BUCKET)
    EXISTING_GPU_INSTANCE_ID: str = os.getenv("EXISTING_GPU_INSTANCE_ID", "")
    SKIP_GPU_SETUP: bool = os.getenv("SKIP_GPU_SETUP", "false").lower() == "true"
    
    class Config:
        env_file = ".env"

settings = Settings()

# Initialize CPU optimization on import
if settings.USE_CPU_OPTIMIZATION:
    from utils.cpu_optimization import configure_tensorflow_for_cpu
    configure_tensorflow_for_cpu()
