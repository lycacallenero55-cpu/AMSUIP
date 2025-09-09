import uuid
import mimetypes
from typing import Optional, Tuple

import boto3
from botocore.config import Config as BotoConfig
from boto3.s3.transfer import TransferConfig

from config import settings


_session = boto3.session.Session(
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION,
)
_s3 = _session.client(
    "s3",
    config=BotoConfig(
        retries={"max_attempts": 5, "mode": "standard"},
        max_pool_connections=20,
        read_timeout=15,
        connect_timeout=5,
    ),
)

# Resource and transfer config for faster multipart uploads
_s3_resource = _session.resource("s3")
_transfer_config = TransferConfig(
    multipart_threshold=8 * 1024 * 1024,  # start multipart at 8MB
    multipart_chunksize=8 * 1024 * 1024,  # 8MB parts
    max_concurrency=8,
    use_threads=True,
)


def _resolve_public_base_url() -> str:
    if settings.S3_PUBLIC_BASE_URL:
        return settings.S3_PUBLIC_BASE_URL.rstrip("/")
    # fallback default pattern
    return f"https://{settings.S3_BUCKET}.s3.{settings.AWS_REGION}.amazonaws.com"


def make_key(student_id: int | str, label: str, filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
    safe_label = "genuine" if str(label).lower() == "genuine" else "forged"
    return f"{student_id}/{safe_label}/{uuid.uuid4().hex}.{ext}"


def upload_bytes(
    student_id: int | str,
    label: str,
    filename: str,
    content: bytes,
    content_type: Optional[str] = None,
) -> Tuple[str, str]:
    key = make_key(student_id, label, filename)
    ct = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    acl = "private" if settings.S3_USE_PRESIGNED_GET else "public-read"
    import io
    buf = io.BytesIO(content)
    _s3_resource.Bucket(settings.S3_BUCKET).upload_fileobj(
        Fileobj=buf,
        Key=key,
        ExtraArgs={
            "ContentType": ct,
            "ACL": acl,
            "CacheControl": "max-age=31536000,public",
        },
        Config=_transfer_config,
    )
    url = f"{_resolve_public_base_url()}/{key}"
    return key, url


def create_presigned_post(
    student_id: int | str,
    label: str,
    filename: str,
    content_type: Optional[str] = None,
    expires_seconds: int = 900,
):
    key = make_key(student_id, label, filename)
    fields = {"Content-Type": content_type} if content_type else None
    conditions = []
    if content_type:
        conditions.append({"Content-Type": content_type})
    post = _s3.generate_presigned_post(
        Bucket=settings.S3_BUCKET,
        Key=key,
        Fields=fields,
        Conditions=conditions,
        ExpiresIn=expires_seconds,
    )
    public_url = f"{_resolve_public_base_url()}/{key}"
    return {"key": key, "url": post["url"], "fields": post["fields"], "public_url": public_url}


def create_presigned_get(key: str, expires_seconds: int = 900) -> str:
    return _s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.S3_BUCKET, "Key": key},
        ExpiresIn=expires_seconds,
    )


def delete_key(key: str) -> None:
    _s3.delete_object(Bucket=settings.S3_BUCKET, Key=key)


def upload_model_file(
    model_data: bytes,
    model_type: str,  # 'individual' or 'global'
    model_uuid: str,
    file_extension: str = "keras"
) -> Tuple[str, str]:
    """
    Upload a trained model file to S3 with organized folder structure.
    
    Args:
        model_data: The model file bytes
        model_type: 'individual' or 'global' 
        model_uuid: Unique identifier for the model
        file_extension: File extension (default: keras)
    
    Returns:
        Tuple of (s3_key, s3_url)
    """
    # Updated folder structure: models/{type}/{uuid}.{ext}
    # models/individual/{uuid}.keras for individual student models
    # models/global/{uuid}.keras for global multi-student models
    key = f"models/{model_type}/{model_uuid}.{file_extension}"
    
    content_type = "application/octet-stream"
    if file_extension == "keras":
        content_type = "application/keras"
    elif file_extension == "h5":
        content_type = "application/hdf5"
    elif file_extension == "json":
        content_type = "application/json"
    
    import io
    buf = io.BytesIO(model_data)
    _s3_resource.Bucket(settings.S3_BUCKET).upload_fileobj(
        Fileobj=buf,
        Key=key,
        ExtraArgs={
            "ContentType": content_type,
            "ACL": "private",
            "CacheControl": "max-age=31536000",
        },
        Config=_transfer_config,
    )
    
    url = f"{_resolve_public_base_url()}/{key}"
    return key, url


def download_model_file(s3_key: str) -> bytes:
    """
    Download a model file from S3.
    
    Args:
        s3_key: The S3 key of the model file
    
    Returns:
        The model file bytes
    """
    response = _s3.get_object(Bucket=settings.S3_BUCKET, Key=s3_key)
    return response['Body'].read()


def delete_model_file(s3_key: str) -> None:
    """
    Delete a model file from S3.
    
    Args:
        s3_key: The S3 key of the model file
    """
    _s3.delete_object(Bucket=settings.S3_BUCKET, Key=s3_key)


def download_bytes(s3_key: str) -> bytes:
    """Download arbitrary bytes from S3 for a given key."""
    response = _s3.get_object(Bucket=settings.S3_BUCKET, Key=s3_key)
    return response['Body'].read()


