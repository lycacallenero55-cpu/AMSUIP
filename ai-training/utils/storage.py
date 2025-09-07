import os
import tempfile
import logging
import boto3
from config import settings

logger = logging.getLogger(__name__)

# Initialize AWS S3 client
session = boto3.session.Session(
	aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
	aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
	region_name=os.getenv("AWS_REGION", "us-east-1"),
)
s3_client = session.client("s3")
S3_BUCKET = os.getenv("S3_BUCKET", "signature-ai")

async def save_to_supabase(local_file_path: str, supabase_path: str) -> str:
	"""Upload a file to AWS S3 (compat function name)."""
	try:
		s3_key = supabase_path
		s3_client.upload_file(local_file_path, S3_BUCKET, s3_key)
		logger.info(f"File uploaded to S3: s3://{S3_BUCKET}/{s3_key}")
		return s3_key
	except Exception as e:
		logger.error(f"Error uploading to S3: {e}")
		raise

async def download_from_supabase(supabase_path: str) -> str:
	"""Download a file from S3 to local temp file (compat function name)."""
	try:
		suffix = '.keras' if supabase_path.endswith('.keras') else '.h5'
		temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
		s3_client.download_file(S3_BUCKET, supabase_path, temp_file.name)
		logger.info(f"File downloaded from S3: s3://{S3_BUCKET}/{supabase_path}")
		return temp_file.name
	except Exception as e:
		logger.error(f"Error downloading from S3: {e}")
		raise

async def load_model_from_supabase(supabase_path: str):
	"""Load a model from S3 by downloading to temp then loading with Keras (compat signature)."""
	try:
		from tensorflow import keras
		suffix = '.keras' if supabase_path.endswith('.keras') else '.h5'
		temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
		temp_path = temp_file.name
		temp_file.close()
		s3_client.download_file(S3_BUCKET, supabase_path, temp_path)
		model = keras.models.load_model(temp_path)
		try:
			os.unlink(temp_path)
		except OSError:
			pass
		logger.info(f"Model loaded from S3: s3://{S3_BUCKET}/{supabase_path}")
		return model
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