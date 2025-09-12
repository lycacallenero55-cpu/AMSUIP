# Signature AI Training Backend

AI-powered signature verification training backend using TensorFlow and Supabase.

## Important: Current Mode (Identification-First)

- The system is currently configured to prioritize identifying the most likely owner of a submitted signature.
- Forgery/anti-spoofing checks are disabled for now to focus engineering time on ownership detection accuracy.
- Verification results will return the predicted student and a confidence score based on a fusion of global embeddings and per-student classification.
- Image uploads are permanently stored in AWS S3, and re-uploads of the same image are allowed (no duplicate blocking).

## Features

- **Signature Verification Training**: Train Siamese neural networks for signature verification
- **Real-time Verification**: Verify signatures against trained models
- **Supabase Integration**: Store models and metadata in Supabase
- **RESTful API**: FastAPI-based API for easy integration
- **Image Processing**: Automatic image validation and preprocessing

### Identification Focus Details

- Owner identification uses a hybrid approach:
  - Global model embeddings to compare the submitted signature against learned student centroids
  - Per-student classification head (when available) to predict the best matching class
  - Fusion of both signals to determine the most likely owner and confidence
- Unknown gating is conservative but does not use authenticity checks while the feature is disabled.

## Setup

### 1. Install Python 3.10+
```bash
# Python 3.10+ is required (3.10.11 works perfectly)
# Windows: Download from python.org
# macOS: brew install python@3.10
# Ubuntu: sudo apt install python3.10 python3.10-venv
```

### 2. Create Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env with your Supabase credentials
```

### 5. Run the Server
```bash
python main.py
```

## API Endpoints

### Training
- `POST /api/training/start` - Start training a model
- `GET /api/training/status/{model_id}` - Get training status
- `GET /api/training/models` - Get all trained models

### Verification
- `POST /api/verification/identify` - Identify the most likely owner of a signature (primary endpoint in current mode)
  - Request form-data: `test_file` (image)
  - Response includes: `predicted_student { id, name }`, `confidence`, `score`, `is_unknown`
- `POST /api/verification/verify` - Verify a signature against a specific student (optional)
  - Request form-data: `test_file` (image), optional `student_id`
  - Response includes: `predicted_student`, `is_match`, `confidence`, `is_unknown`

### Uploads & Storage
- `POST /api/uploads/signature` - Upload a signature image for a student
  - Form fields: `student_id`, `label` (`genuine`|`forged`), `file`
  - Behavior: uploads are stored permanently in S3; duplicate images are allowed
- `GET /api/uploads/list?student_id=...` - List persisted signatures for previews

## Usage

### Training a Model
```python
import requests

# Upload images and start training
files = {
    'genuine_files': [open('genuine1.jpg', 'rb'), open('genuine2.jpg', 'rb')],
    'forged_files': [open('forged1.jpg', 'rb')]
}
data = {'student_id': 123}

response = requests.post('http://localhost:8000/api/training/start', files=files, data=data)
```

### Verifying a Signature
```python
# Verify a signature
files = {
    'reference_files': [open('reference.jpg', 'rb')],
    'test_file': open('test_signature.jpg', 'rb')
}
data = {'model_id': 'your-model-id'}

response = requests.post('http://localhost:8000/api/verification/verify', files=files, data=data)
```

## Configuration

Edit `.env` file to configure:
- Supabase credentials
- Model parameters
- Training settings
- Storage options

## Requirements

- Python 3.10+ (3.10.11 works perfectly)
- TensorFlow 2.15+
- Supabase account
- Minimum 4GB RAM
- CPU with AVX support (recommended)

## Configuration Notes

- Identification-first mode is controlled via environment variables in `.env` (defaults shown):
  - `ENABLE_ANTISPOOFING=false`
  - `USE_ADAPTIVE_THRESHOLD=false`
- S3 storage must be configured; example keys:
  - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `S3_BUCKET`, `S3_PUBLIC_BASE_URL`, `S3_USE_PRESIGNED_GET=true`

## Roadmap

- Re-enable and refine forgery/anti-spoofing once identification accuracy is solid and time permits.
- Improve centroid caching and model warm-loading for faster first request.