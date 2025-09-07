import os
import tempfile
from supabase import create_client
from config import settings
import logging

logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

async def save_to_supabase(local_file_path: str, supabase_path: str) -> str:
    """Upload a file to Supabase Storage"""
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

def cleanup_local_file(file_path: str):
    """Clean up a local file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Local file cleaned up: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up local file: {e}")