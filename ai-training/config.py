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
    MODEL_BATCH_SIZE = int(os.getenv("MODEL_BATCH_SIZE", 32))
    MODEL_EPOCHS = int(os.getenv("MODEL_EPOCHS", 20))  # Reduced for faster training
    MODEL_LEARNING_RATE = float(os.getenv("MODEL_LEARNING_RATE", 0.0002))  # Further reduced to prevent overfitting

        # CPU Optimization
    USE_CPU_OPTIMIZATION: bool = True
    CPU_THREADS: int = 6  # For Ryzen 5 3400G
    
    # Storage Configuration
    LOCAL_MODELS_DIR = os.getenv("LOCAL_MODELS_DIR", "./models")
    SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "models")
    
    # Training Configuration
    MIN_GENUINE_SAMPLES = int(os.getenv("MIN_GENUINE_SAMPLES", 10))
    MIN_FORGED_SAMPLES = int(os.getenv("MIN_FORGED_SAMPLES", 5))
    MAX_TRAINING_TIME = int(os.getenv("MAX_TRAINING_TIME", 3600))

    # Verification Settings
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.7
    USE_ADAPTIVE_THRESHOLD: bool = False
    
        # Anti-Spoofing (disabled to prioritize identification flow)
    ENABLE_ANTISPOOFING: bool = False
    SPOOFING_THRESHOLD: float = 0.6
    
    # Model Versioning
    ENABLE_MODEL_VERSIONING: bool = True
    MAX_MODEL_VERSIONS: int = 5
    
    # Performance Monitoring
    ENABLE_PERFORMANCE_MONITORING: bool = True
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = Settings()

# Initialize CPU optimization on import
if settings.USE_CPU_OPTIMIZATION:
    from utils.cpu_optimization import configure_tensorflow_for_cpu
    configure_tensorflow_for_cpu()