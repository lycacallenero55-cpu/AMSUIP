"""
End-to-end tests for the AI signature verification system
"""
import pytest
import asyncio
import tempfile
import os
import json
import time
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image
import io
import requests
from fastapi.testclient import TestClient

from main import app
from models.signature_embedding_model import SignatureEmbeddingModel
from utils.s3_storage import S3Storage
from models.database import DatabaseManager


class TestAIE2E:
    """End-to-end tests for the complete AI system"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def sample_signature_images(self):
        """Create sample signature images for testing"""
        images = []
        for i in range(10):
            # Create a simple signature-like image
            img = Image.new('RGB', (200, 100), color='white')
            # Add some random noise to simulate signature strokes
            pixels = np.array(img)
            noise = np.random.randint(0, 50, pixels.shape)
            pixels = np.clip(pixels + noise, 0, 255)
            img = Image.fromarray(pixels.astype('uint8'))
            
            # Save to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            images.append(img_bytes.getvalue())
        
        return images

    @pytest.fixture
    def mock_s3_storage(self):
        """Mock S3 storage for testing"""
        with patch('utils.s3_storage.S3Storage') as mock_s3:
            mock_instance = MagicMock()
            mock_instance.upload_file.return_value = {
                's3_key': 'test/signature.png',
                's3_url': 'https://s3.amazonaws.com/bucket/test/signature.png'
            }
            mock_instance.object_exists.return_value = True
            mock_instance.count_objects_with_prefix.return_value = 5
            mock_instance.get_presigned_url.return_value = 'https://s3.amazonaws.com/bucket/test/signature.png'
            mock_s3.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_database(self):
        """Mock database for testing"""
        with patch('models.database.DatabaseManager') as mock_db:
            mock_instance = MagicMock()
            mock_instance.get_trained_models.return_value = []
            mock_instance.create_trained_model.return_value = {'id': 1}
            mock_instance.get_students.return_value = [
                {'id': 1, 'student_id': 'STU001', 'firstname': 'John', 'surname': 'Doe'}
            ]
            mock_instance.get_student_signatures.return_value = []
            mock_instance.create_student_signature.return_value = {'id': 1}
            mock_db.return_value = mock_instance
            yield mock_instance

    def test_upload_signature_e2e(self, client, sample_signature_images, mock_s3_storage, mock_database):
        """Test complete signature upload flow"""
        # Test signature upload endpoint
        files = {'file': ('signature.png', sample_signature_images[0], 'image/png')}
        data = {'student_id': '1', 'label': 'genuine'}
        
        response = client.post('/api/uploads/signature', files=files, data=data)
        
        assert response.status_code == 200
        result = response.json()
        assert 'record' in result
        assert result['record']['student_id'] == 1
        assert result['record']['label'] == 'genuine'

    def test_training_flow_e2e(self, client, sample_signature_images, mock_s3_storage, mock_database):
        """Test complete training flow"""
        # First, upload some signatures
        for i, img_data in enumerate(sample_signature_images[:5]):
            files = {'file': (f'signature_{i}.png', img_data, 'image/png')}
            data = {'student_id': '1', 'label': 'genuine'}
            response = client.post('/api/uploads/signature', files=files, data=data)
            assert response.status_code == 200

        for i, img_data in enumerate(sample_signature_images[5:7]):
            files = {'file': (f'forged_{i}.png', img_data, 'image/png')}
            data = {'student_id': '1', 'label': 'forged'}
            response = client.post('/api/uploads/signature', files=files, data=data)
            assert response.status_code == 200

        # Start training
        with patch('api.training.SignatureEmbeddingModel') as mock_model_class:
            mock_model = MagicMock()
            mock_model.train_model.return_value = {
                'success': True,
                'accuracy': 0.95,
                'loss': 0.1,
                'val_accuracy': 0.92,
                'val_loss': 0.12
            }
            mock_model.save_model_to_s3.return_value = {
                'success': True,
                's3_key': 'models/test_model.keras',
                's3_url': 'https://s3.amazonaws.com/bucket/models/test_model.keras'
            }
            mock_model_class.return_value = mock_model

            response = client.post('/api/training/start', data={'student_id': '1'})
            assert response.status_code == 200
            result = response.json()
            assert result['success'] is True

    def test_verification_flow_e2e(self, client, sample_signature_images, mock_s3_storage, mock_database):
        """Test complete verification flow"""
        # First, train a model
        with patch('api.training.SignatureEmbeddingModel') as mock_model_class:
            mock_model = MagicMock()
            mock_model.train_model.return_value = {
                'success': True,
                'accuracy': 0.95,
                'loss': 0.1
            }
            mock_model.save_model_to_s3.return_value = {
                'success': True,
                's3_key': 'models/test_model.keras',
                's3_url': 'https://s3.amazonaws.com/bucket/models/test_model.keras'
            }
            mock_model_class.return_value = mock_model

            # Upload signatures and train
            for i, img_data in enumerate(sample_signature_images[:3]):
                files = {'file': (f'signature_{i}.png', img_data, 'image/png')}
                data = {'student_id': '1', 'label': 'genuine'}
                client.post('/api/uploads/signature', files=files, data=data)

            # Train model
            client.post('/api/training/start', data={'student_id': '1'})

        # Now test verification
        with patch('api.verification.load_trained_model') as mock_load_model:
            mock_model = MagicMock()
            mock_model.verify_signature.return_value = {
                'is_match': True,
                'confidence': 0.95,
                'is_genuine': True,
                'authenticity_score': 0.0,
                'predicted_student_id': 1
            }
            mock_load_model.return_value = mock_model

            # Test verification
            files = {'test_file': ('test_signature.png', sample_signature_images[0], 'image/png')}
            response = client.post('/api/verification/identify', files=files)
            
            assert response.status_code == 200
            result = response.json()
            assert result['success'] is True
            assert result['match'] is True
            assert result['score'] > 0.5

    def test_async_training_e2e(self, client, sample_signature_images, mock_s3_storage, mock_database):
        """Test async training flow"""
        # Upload signatures
        for i, img_data in enumerate(sample_signature_images[:5]):
            files = {'file': (f'signature_{i}.png', img_data, 'image/png')}
            data = {'student_id': '1', 'label': 'genuine'}
            client.post('/api/uploads/signature', files=files, data=data)

        # Start async training
        with patch('api.training.SignatureEmbeddingModel') as mock_model_class:
            mock_model = MagicMock()
            mock_model.train_model.return_value = {
                'success': True,
                'accuracy': 0.95,
                'loss': 0.1
            }
            mock_model.save_model_to_s3.return_value = {
                'success': True,
                's3_key': 'models/test_model.keras',
                's3_url': 'https://s3.amazonaws.com/bucket/models/test_model.keras'
            }
            mock_model_class.return_value = mock_model

            response = client.post('/api/training/start-async', data={
                'student_id': '1',
                'training_mode': 'hybrid'
            })
            
            assert response.status_code == 200
            result = response.json()
            assert 'job_id' in result
            assert result['success'] is True

    def test_gpu_training_e2e(self, client, sample_signature_images, mock_s3_storage, mock_database):
        """Test GPU training flow"""
        # Upload signatures
        for i, img_data in enumerate(sample_signature_images[:5]):
            files = {'file': (f'signature_{i}.png', img_data, 'image/png')}
            data = {'student_id': '1', 'label': 'genuine'}
            client.post('/api/uploads/signature', files=files, data=data)

        # Start GPU training
        with patch('api.training.start_gpu_training') as mock_gpu_training:
            mock_gpu_training.return_value = {
                'success': True,
                'job_id': 'gpu-job-123',
                'message': 'GPU training started'
            }

            response = client.post('/api/training/start-gpu-training', data={
                'student_id': '1',
                'use_gpu': 'true',
                'training_mode': 'hybrid'
            })
            
            assert response.status_code == 200
            result = response.json()
            assert result['success'] is True
            assert 'job_id' in result

    def test_progress_streaming_e2e(self, client):
        """Test progress streaming functionality"""
        # Mock job progress
        mock_job = {
            'job_id': 'test-job-123',
            'status': 'running',
            'progress': 50,
            'current_stage': 'training',
            'estimated_time_remaining': 300
        }

        with patch('api.progress.get_job_status') as mock_get_status:
            mock_get_status.return_value = mock_job

            # Test job status endpoint
            response = client.get('/api/progress/job/test-job-123')
            assert response.status_code == 200
            result = response.json()
            assert result['job_id'] == 'test-job-123'
            assert result['status'] == 'running'

    def test_health_check_e2e(self, client):
        """Test health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        result = response.json()
        assert result['status'] == 'healthy'

    def test_error_handling_e2e(self, client):
        """Test error handling in various scenarios"""
        # Test with invalid student ID
        files = {'file': ('signature.png', b'invalid_data', 'image/png')}
        data = {'student_id': 'invalid', 'label': 'genuine'}
        response = client.post('/api/uploads/signature', files=files, data=data)
        assert response.status_code == 422  # Validation error

        # Test verification without trained model
        files = {'test_file': ('test_signature.png', b'test_data', 'image/png')}
        response = client.post('/api/verification/identify', files=files)
        assert response.status_code == 200  # Should handle gracefully
        result = response.json()
        assert result['success'] is False

    def test_small_dataset_training_e2e(self, client, sample_signature_images, mock_s3_storage, mock_database):
        """Test training with small dataset (3+ images)"""
        # Upload minimal dataset
        for i, img_data in enumerate(sample_signature_images[:3]):
            files = {'file': (f'signature_{i}.png', img_data, 'image/png')}
            data = {'student_id': '1', 'label': 'genuine'}
            response = client.post('/api/uploads/signature', files=files, data=data)
            assert response.status_code == 200

        for i, img_data in enumerate(sample_signature_images[3:5]):
            files = {'file': (f'forged_{i}.png', img_data, 'image/png')}
            data = {'student_id': '1', 'label': 'forged'}
            response = client.post('/api/uploads/signature', files=files, data=data)
            assert response.status_code == 200

        # Train with small dataset
        with patch('api.training.SignatureEmbeddingModel') as mock_model_class:
            mock_model = MagicMock()
            mock_model.train_model.return_value = {
                'success': True,
                'accuracy': 0.85,  # Should still achieve reasonable accuracy
                'loss': 0.2
            }
            mock_model.save_model_to_s3.return_value = {
                'success': True,
                's3_key': 'models/small_dataset_model.keras',
                's3_url': 'https://s3.amazonaws.com/bucket/models/small_dataset_model.keras'
            }
            mock_model_class.return_value = mock_model

            response = client.post('/api/training/start', data={'student_id': '1'})
            assert response.status_code == 200
            result = response.json()
            assert result['success'] is True

    def test_forgery_detection_disabled_e2e(self, client, sample_signature_images, mock_s3_storage, mock_database):
        """Test that forgery detection is properly disabled end-to-end"""
        # Train a model
        with patch('api.training.SignatureEmbeddingModel') as mock_model_class:
            mock_model = MagicMock()
            mock_model.train_model.return_value = {
                'success': True,
                'accuracy': 0.95,
                'loss': 0.1
            }
            mock_model.save_model_to_s3.return_value = {
                'success': True,
                's3_key': 'models/test_model.keras',
                's3_url': 'https://s3.amazonaws.com/bucket/models/test_model.keras'
            }
            mock_model_class.return_value = mock_model

            # Upload and train
            for i, img_data in enumerate(sample_signature_images[:3]):
                files = {'file': (f'signature_{i}.png', img_data, 'image/png')}
                data = {'student_id': '1', 'label': 'genuine'}
                client.post('/api/uploads/signature', files=files, data=data)

            client.post('/api/training/start', data={'student_id': '1'})

        # Test verification - should always return is_genuine=True
        with patch('api.verification.load_trained_model') as mock_load_model:
            mock_model = MagicMock()
            mock_model.verify_signature.return_value = {
                'is_match': True,
                'confidence': 0.95,
                'is_genuine': True,  # Always true since forgery detection is disabled
                'authenticity_score': 0.0,  # Always 0 since forgery detection is disabled
                'predicted_student_id': 1
            }
            mock_load_model.return_value = mock_model

            files = {'test_file': ('test_signature.png', sample_signature_images[0], 'image/png')}
            response = client.post('/api/verification/identify', files=files)
            
            assert response.status_code == 200
            result = response.json()
            assert result['success'] is True
            assert result['is_genuine'] is True  # Should always be true

    def test_model_storage_and_retrieval_e2e(self, client, sample_signature_images, mock_s3_storage, mock_database):
        """Test model storage and retrieval"""
        # Train and save a model
        with patch('api.training.SignatureEmbeddingModel') as mock_model_class:
            mock_model = MagicMock()
            mock_model.train_model.return_value = {
                'success': True,
                'accuracy': 0.95,
                'loss': 0.1
            }
            mock_model.save_model_to_s3.return_value = {
                'success': True,
                's3_key': 'models/test_model.keras',
                's3_url': 'https://s3.amazonaws.com/bucket/models/test_model.keras'
            }
            mock_model_class.return_value = mock_model

            # Upload and train
            for i, img_data in enumerate(sample_signature_images[:3]):
                files = {'file': (f'signature_{i}.png', img_data, 'image/png')}
                data = {'student_id': '1', 'label': 'genuine'}
                client.post('/api/uploads/signature', files=files, data=data)

            response = client.post('/api/training/start', data={'student_id': '1'})
            assert response.status_code == 200

        # Test model retrieval
        response = client.get('/api/training/models')
        assert response.status_code == 200
        result = response.json()
        assert 'models' in result

    def test_concurrent_training_e2e(self, client, sample_signature_images, mock_s3_storage, mock_database):
        """Test concurrent training requests"""
        # Upload signatures for multiple students
        for student_id in ['1', '2']:
            for i, img_data in enumerate(sample_signature_images[:3]):
                files = {'file': (f'signature_{i}_{student_id}.png', img_data, 'image/png')}
                data = {'student_id': student_id, 'label': 'genuine'}
                client.post('/api/uploads/signature', files=files, data=data)

        # Start concurrent training
        with patch('api.training.SignatureEmbeddingModel') as mock_model_class:
            mock_model = MagicMock()
            mock_model.train_model.return_value = {
                'success': True,
                'accuracy': 0.95,
                'loss': 0.1
            }
            mock_model.save_model_to_s3.return_value = {
                'success': True,
                's3_key': 'models/test_model.keras',
                's3_url': 'https://s3.amazonaws.com/bucket/models/test_model.keras'
            }
            mock_model_class.return_value = mock_model

            # Start training for both students
            response1 = client.post('/api/training/start', data={'student_id': '1'})
            response2 = client.post('/api/training/start', data={'student_id': '2'})
            
            assert response1.status_code == 200
            assert response2.status_code == 200

    def test_data_validation_e2e(self, client):
        """Test data validation in API endpoints"""
        # Test invalid file type
        files = {'file': ('test.txt', b'not an image', 'text/plain')}
        data = {'student_id': '1', 'label': 'genuine'}
        response = client.post('/api/uploads/signature', files=files, data=data)
        assert response.status_code == 422

        # Test missing required fields
        files = {'file': ('test.png', b'fake image data', 'image/png')}
        data = {'student_id': '1'}  # Missing label
        response = client.post('/api/uploads/signature', files=files, data=data)
        assert response.status_code == 422

        # Test invalid student ID
        files = {'file': ('test.png', b'fake image data', 'image/png')}
        data = {'student_id': 'invalid', 'label': 'genuine'}
        response = client.post('/api/uploads/signature', files=files, data=data)
        assert response.status_code == 422


if __name__ == '__main__':
    pytest.main([__file__, '-v'])