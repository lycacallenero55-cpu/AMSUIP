# AI Signature Verification System

## Overview

This system provides AI-powered signature verification with focus on owner identification. It uses deep learning with transfer learning and data augmentation to achieve high accuracy even with small datasets (minimum 3 signatures per student).

## Key Features

- **Owner Identification**: Identifies the owner of a signature from trained students
- **Transfer Learning**: Uses MobileNetV2 as backbone for better performance with small datasets
- **Data Augmentation**: Advanced augmentation pipeline for signature-specific variations
- **Real-time Processing**: Fast inference for real-time verification
- **Scalable**: Supports up to 150 students
- **Robust Error Handling**: Comprehensive error handling and fallback mechanisms

## Architecture

### Core Components

1. **SignatureEmbeddingModel**: Main AI model with transfer learning
2. **SignaturePreprocessor**: Advanced image preprocessing pipeline
3. **SignatureAugmentation**: Signature-specific data augmentation
4. **Training Pipeline**: End-to-end training with atomic operations
5. **Verification API**: Real-time signature verification endpoints

### Model Architecture

- **Backbone**: MobileNetV2 (pre-trained on ImageNet)
- **Transfer Learning**: Frozen base layers with custom classification head
- **Progressive Unfreezing**: Two-phase training for optimal performance
- **Data Augmentation**: 3x augmentation with signature-specific transforms

## Installation

### Prerequisites

- Python 3.10.11
- TensorFlow 2.x
- OpenCV
- PIL/Pillow
- NumPy

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export SUPABASE_URL="your_supabase_url"
export SUPABASE_KEY="your_supabase_key"
export AWS_ACCESS_KEY_ID="your_aws_key"
export AWS_SECRET_ACCESS_KEY="your_aws_secret"
export S3_BUCKET="your_s3_bucket"
```

## Usage

### Training a Model

```python
from models.signature_embedding_model import SignatureEmbeddingModel

# Create model
model = SignatureEmbeddingModel(max_students=150)

# Prepare training data
training_data = {
    "student_1": {
        "genuine": [image1, image2, image3],  # PIL Images or numpy arrays
        "forged": []  # Not used for owner identification
    },
    "student_2": {
        "genuine": [image4, image5, image6],
        "forged": []
    }
}

# Train model
result = model.train_classification_only(training_data, epochs=50)
```

### Verifying a Signature

```python
# Load trained model
model.load_models("path/to/model")

# Verify signature
result = model.verify_signature(test_signature_image)

print(f"Predicted student: {result['predicted_student_name']}")
print(f"Confidence: {result['student_confidence']:.3f}")
print(f"Is match: {result['is_match']}")
```

### API Endpoints

#### Training
- `POST /api/training/start` - Start training for a student
- `POST /api/training/start-async` - Start async training
- `POST /api/training/start-gpu-training` - Start GPU training

#### Verification
- `POST /api/verify/identify` - Identify signature owner
- `POST /api/verify/verify` - Verify signature against specific student

#### Sync
- `POST /api/verify/sync-s3-supabase` - Sync S3 with Supabase
- `GET /api/verify/missing-images` - Get students with missing images

## Configuration

### Key Settings

```python
# config.py
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for positive match
ENABLE_FORGERY_DETECTION = False  # Disabled for owner identification focus
MODEL_IMAGE_SIZE = 224  # Input image size
MODEL_BATCH_SIZE = 16  # Training batch size
MODEL_EPOCHS = 50  # Training epochs
MIN_GENUINE_SAMPLES = 3  # Minimum signatures per student
```

### Data Augmentation Settings

```python
# SignatureAugmentation parameters
rotation_range = 10.0  # Rotation angle range
scale_range = (0.9, 1.1)  # Scale range
brightness_range = 0.2  # Brightness variation
noise_std = 3.0  # Noise standard deviation
blur_probability = 0.2  # Blur application probability
```

## Testing

### Run All Tests

```bash
python run_ai_tests.py
```

### Run Specific Tests

```bash
# Image processing tests
pytest tests/test_image_processing.py -v

# Training tests
pytest tests/test_training.py -v

# Integration tests
pytest tests/test_integration.py -v
```

### Test Coverage

The test suite covers:
- Image preprocessing pipeline
- Data augmentation
- Model training and verification
- Error handling
- Integration scenarios

## Performance

### Training Performance

- **Small Dataset (3-5 signatures)**: 2-5 minutes on CPU
- **Medium Dataset (10-20 signatures)**: 5-15 minutes on CPU
- **Large Dataset (50+ signatures)**: 15-60 minutes on CPU
- **GPU Training**: 3-10x faster

### Inference Performance

- **Single Signature**: < 100ms
- **Batch Processing**: < 50ms per signature
- **Memory Usage**: ~500MB for model + 100MB per batch

## Troubleshooting

### Common Issues

1. **Low Accuracy with Small Datasets**
   - Increase augmentation factor
   - Use more conservative augmentation parameters
   - Ensure high-quality signature images

2. **Training Failures**
   - Check image format and quality
   - Verify minimum sample requirements
   - Check memory availability

3. **Verification Errors**
   - Ensure model is properly loaded
   - Check student mappings
   - Verify image preprocessing

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Best Practices

### Data Preparation

1. **Image Quality**: Use high-resolution, clear signature images
2. **Consistency**: Ensure consistent lighting and background
3. **Variety**: Include different writing styles and pressures
4. **Quantity**: Minimum 3 signatures per student, 5+ recommended

### Training

1. **Progressive Training**: Use two-phase training approach
2. **Validation**: Always use validation split
3. **Monitoring**: Monitor training metrics and adjust parameters
4. **Saving**: Save models and mappings after successful training

### Verification

1. **Threshold Tuning**: Adjust confidence threshold based on use case
2. **Preprocessing**: Ensure consistent preprocessing pipeline
3. **Error Handling**: Implement proper error handling and fallbacks

## API Reference

### SignatureEmbeddingModel

#### Methods

- `prepare_training_data(training_data)`: Prepare training data with augmentation
- `train_classification_only(training_data, epochs)`: Train classification model
- `verify_signature(image)`: Verify signature and return prediction
- `save_models(base_path)`: Save trained models
- `load_models(base_path)`: Load trained models

### SignaturePreprocessor

#### Methods

- `preprocess_signature(image)`: Complete preprocessing pipeline
- `_assess_quality(image)`: Assess image quality

### SignatureAugmentation

#### Methods

- `augment_signature(image, is_genuine)`: Apply augmentations
- `augment_batch(images, labels, factor)`: Augment batch of images

## License

This system is part of the AMSUIP project. See the main project license for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test suite for examples
3. Check logs for error details
4. Contact the development team