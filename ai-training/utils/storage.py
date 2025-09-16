import os
import tempfile
from supabase import create_client
from config import settings
import logging

logger = logging.getLogger(__name__)

# Initialize Supabase client (with offline mode support)
try:
    if settings.SUPABASE_URL:
        supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    else:
        supabase = None
        logger.warning("Supabase URL not configured - running in offline mode")
except Exception as e:
    supabase = None
    logger.warning(f"Failed to initialize Supabase client: {e} - running in offline mode")

async def save_to_supabase(local_file_path: str, supabase_path: str) -> str:
    """Upload a file to Supabase Storage"""
    if not supabase:
        raise Exception("Supabase client not available - running in offline mode")
    
    try:
        with open(local_file_path, 'rb') as f:
            file_data = f.read()
        
        response = supabase.storage.from_(settings.SUPABASE_BUCKET).upload(
            supabase_path,
            file_data,
            file_options={
                "contentType": "application/octet-stream",
                "cacheControl": "3600",
                "upsert": "true"
            }
        )
        
        # Supabase Python v2 returns an UploadResponse object
        if hasattr(response, 'error') and response.error is not None:
            raise Exception(f"Upload failed: {response.error}")
        
        logger.info(f"File uploaded to Supabase: {supabase_path}")
        return supabase_path
    
    except Exception as e:
        logger.error(f"Error uploading to Supabase: {e}")
        raise

async def download_from_supabase(supabase_path: str) -> str:
    """Download a file from Supabase Storage to local temp file"""
    if not supabase:
        raise Exception("Supabase client not available - running in offline mode")
    
    try:
        response = supabase.storage.from_(settings.SUPABASE_BUCKET).download(supabase_path)
        
        if response is None:
            raise Exception("Download failed: No data received")
        
        # Create temporary file
        suffix = '.keras' if supabase_path.endswith('.keras') else '.h5'
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        # Supabase python client returns bytes
        data = response if isinstance(response, (bytes, bytearray)) else bytes(response)
        temp_file.write(data)
        temp_file.close()
        
        logger.info(f"File downloaded from Supabase: {supabase_path}")
        return temp_file.name
    
    except Exception as e:
        logger.error(f"Error downloading from Supabase: {e}")
        raise

async def load_model_from_supabase(supabase_path: str):
    """Load a model directly from Supabase Storage into memory without saving to disk"""
    if not supabase:
        raise Exception("Supabase client not available - running in offline mode")
    
    try:
        response = supabase.storage.from_(settings.SUPABASE_BUCKET).download(supabase_path)
        
        if response is None:
            raise Exception("Download failed: No data received")
        
        # Use temporary file with manual cleanup
        import tempfile
        import os
        from tensorflow import keras
        
        # Create a temporary file with the correct extension
        suffix = '.keras' if supabase_path.endswith('.keras') else '.h5'
        
        # Create temporary file that won't be auto-deleted
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = temp_file.name
        
        try:
            # Write data to temporary file
            temp_file.write(response)
            temp_file.flush()
            temp_file.close()  # Close file handle but keep file
            
            # Load model from temporary file
            model = keras.models.load_model(temp_path)
            
            logger.info(f"Model loaded directly from Supabase: {supabase_path}")
            return model
            
        finally:
            # Clean up temporary file after loading
            try:
                os.unlink(temp_path)
            except OSError:
                pass  # File might already be deleted
    
    except Exception as e:
        logger.error(f"Error loading model from Supabase: {e}")
        raise


async def load_model_from_s3(s3_url: str):
    """Load a model directly from S3 into memory without saving to disk"""
    try:
        from utils.s3_storage import download_model_file
        from tensorflow import keras
        import tempfile
        import os
        from urllib.parse import urlparse
        
        # Extract S3 key from URL (support multiple styles)
        # Examples we support:
        #  - https://bucket.s3.region.amazonaws.com/models/type/uuid.keras
        #  - https://s3.region.amazonaws.com/bucket/models/type/uuid.keras
        #  - https://bucket.s3.amazonaws.com/models/type/uuid.keras
        #  - https://cdn.example.com/models/type/uuid.keras (custom domain, full path)
        #  - models/type/uuid.keras (already a key)
        parsed = urlparse(s3_url)
        if parsed.scheme and parsed.netloc:
            # URL with host
            host = parsed.netloc
            path = parsed.path or ""
            # Strip leading '/'
            path = path[1:] if path.startswith('/') else path
            s3_key = path
            # If host is path-style (s3.amazonaws.com or s3.<region>.amazonaws.com), first segment is bucket
            if host == "s3.amazonaws.com" or host.startswith("s3."):
                parts = path.split('/') if path else []
                if parts and parts[0] == settings.S3_BUCKET:
                    s3_key = '/'.join(parts[1:])
            # If host is virtual-hosted (bucket.s3.<region>.amazonaws.com), path is already the key
            # For custom CDN domains, assume full path is the key segment after the domain
        else:
            # Looks like a raw key
            s3_key = s3_url
        
        # Download model data from S3
        model_data = download_model_file(s3_key)
        
        # Create temporary file with the correct extension
        suffix = '.keras' if s3_url.endswith('.keras') else '.h5'
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = temp_file.name
        
        try:
            # Write data to temporary file
            temp_file.write(model_data)
            temp_file.flush()
            temp_file.close()  # Close file handle but keep file
            
            # Load model from temporary file
            model = keras.models.load_model(temp_path)
            
            logger.info(f"Model loaded directly from S3: {s3_url}")
            return model
            
        finally:
            # Clean up temporary file after loading
            try:
                os.unlink(temp_path)
            except OSError:
                pass  # File might already be deleted
    
    except Exception as e:
        logger.error(f"Error loading model from S3: {e}")
        raise

def cleanup_local_file(file_path: str):
    """Clean up a local file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Local file cleaned up: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up local file: {e}")