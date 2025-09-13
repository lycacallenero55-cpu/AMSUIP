"""
End-to-end tests for complete AI workflow
"""
import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from ai_training.main import app


class TestE2EWorkflow:
    """End-to-end tests for complete AI workflow"""

    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        return TestClient(app)

    @pytest.fixture
    def sample_signature_image(self):
        """Generate a sample signature image for testing"""
        # Create a simple signature pattern
        img = np.random.rand(28, 28, 1) * 255
        img[10:18, 5:23] = 0  # Horizontal line
        img[5:13, 8:16] = 0   # Vertical line
        return img.astype(np.uint8)

    @pytest.fixture
    def mock_s3_client(self):
        """Mock S3 client for testing"""
        client = Mock()
        client.upload_file = Mock()
        client.download_file = Mock()
        client.list_objects_v2 = Mock()
        return client

    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client for testing"""
        client = Mock()
        client.table = Mock()
        return client

    def test_upload_train_verify_workflow(self, client, sample_signature_image, mock_s3_client, mock_supabase_client):
        """Test complete workflow: upload -> train -> verify"""
        
        # Mock S3 and Supabase responses
        mock_s3_client.upload_file.return_value = None
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'signatures/1/genuine_1.png'},
                {'Key': 'signatures/1/genuine_2.png'},
                {'Key': 'signatures/1/genuine_3.png'},
            ]
        }
        
        mock_supabase_client.table.return_value.select.return_value.execute.return_value = Mock(
            data=[{'id': 1, 'name': 'Test Student', 'genuine_count': 3, 'forged_count': 0}]
        )
        
        mock_supabase_client.table.return_value.insert.return_value.execute.return_value = Mock(
            data=[{'id': 1, 'status': 'completed'}]
        )
        
        # Mock model training
        with patch('ai_training.models.signature_embedding_model.SignatureEmbeddingModel') as mock_model_class:
            mock_model = Mock()
            mock_model_class.return_value = mock_model
            
            # Mock training
            mock_model.train_classification_only.return_value = Mock(
                history={'loss': [0.5, 0.3], 'accuracy': [0.8, 0.9]}
            )
            
            # Mock verification
            mock_model.verify_signature.return_value = {
                'is_match': True,
                'confidence': 0.85,
                'predicted_student_id': 1,
                'is_unknown': False
            }
            
            # Test 1: Upload signature
            with patch('ai_training.api.training.get_s3_client', return_value=mock_s3_client):
                with patch('ai_training.api.training.get_supabase_client', return_value=mock_supabase_client):
                    response = client.post(
                        "/api/training/upload",
                        files={"file": ("test_signature.png", sample_signature_image.tobytes(), "image/png")},
                        data={"student_id": "1", "signature_type": "genuine"}
                    )
                    assert response.status_code == 200
                    assert response.json()["success"] == True

            # Test 2: Train model
            with patch('ai_training.api.training.get_s3_client', return_value=mock_s3_client):
                with patch('ai_training.api.training.get_supabase_client', return_value=mock_supabase_client):
                    response = client.post("/api/training/train/local")
                    assert response.status_code == 200
                    assert response.json()["success"] == True

            # Test 3: Verify signature
            with patch('ai_training.api.verification.get_s3_client', return_value=mock_s3_client):
                with patch('ai_training.api.verification.get_supabase_client', return_value=mock_supabase_client):
                    response = client.post(
                        "/api/verify",
                        files={"file": ("test_signature.png", sample_signature_image.tobytes(), "image/png")},
                        data={"student_id": "1"}
                    )
                    assert response.status_code == 200
                    result = response.json()
                    assert result["success"] == True
                    assert result["is_match"] == True
                    assert result["confidence"] > 0.5

    def test_verify_unknown_signature(self, client, sample_signature_image, mock_s3_client, mock_supabase_client):
        """Test verification of unknown signature returns no_match"""
        
        # Mock S3 and Supabase responses
        mock_s3_client.list_objects_v2.return_value = {'Contents': []}
        mock_supabase_client.table.return_value.select.return_value.execute.return_value = Mock(data=[])
        
        # Mock model verification for unknown signature
        with patch('ai_training.models.signature_embedding_model.SignatureEmbeddingModel') as mock_model_class:
            mock_model = Mock()
            mock_model_class.return_value = mock_model
            
            # Mock verification for unknown signature
            mock_model.verify_signature.return_value = {
                'is_match': False,
                'confidence': 0.3,  # Below threshold
                'predicted_student_id': 0,
                'is_unknown': True
            }
            
            # Test verification of unknown signature
            with patch('ai_training.api.verification.get_s3_client', return_value=mock_s3_client):
                with patch('ai_training.api.verification.get_supabase_client', return_value=mock_supabase_client):
                    response = client.post(
                        "/api/verify",
                        files={"file": ("unknown_signature.png", sample_signature_image.tobytes(), "image/png")},
                        data={"student_id": "999"}  # Non-existent student
                    )
                    assert response.status_code == 200
                    result = response.json()
                    assert result["success"] == True
                    assert result["is_match"] == False
                    assert result["confidence"] < 0.6
                    assert result["predicted_student"]["id"] == 0
                    assert result["predicted_student"]["name"] == "Unknown"
                    assert result["message"] == "No match found"

    def test_identify_signature_workflow(self, client, sample_signature_image, mock_s3_client, mock_supabase_client):
        """Test identify signature workflow"""
        
        # Mock S3 and Supabase responses
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'signatures/1/genuine_1.png'},
                {'Key': 'signatures/2/genuine_1.png'},
            ]
        }
        
        mock_supabase_client.table.return_value.select.return_value.execute.return_value = Mock(
            data=[
                {'id': 1, 'name': 'Student 1', 'genuine_count': 1, 'forged_count': 0},
                {'id': 2, 'name': 'Student 2', 'genuine_count': 1, 'forged_count': 0}
            ]
        )
        
        # Mock model identification
        with patch('ai_training.models.signature_embedding_model.SignatureEmbeddingModel') as mock_model_class:
            mock_model = Mock()
            mock_model_class.return_value = mock_model
            
            # Mock identification
            mock_model.identify_signature_owner.return_value = {
                'is_match': True,
                'confidence': 0.8,
                'predicted_student_id': 1,
                'predicted_student_name': 'Student 1',
                'is_unknown': False
            }
            
            # Test identify signature
            with patch('ai_training.api.verification.get_s3_client', return_value=mock_s3_client):
                with patch('ai_training.api.verification.get_supabase_client', return_value=mock_supabase_client):
                    response = client.post(
                        "/api/identify",
                        files={"file": ("test_signature.png", sample_signature_image.tobytes(), "image/png")}
                    )
                    assert response.status_code == 200
                    result = response.json()
                    assert result["success"] == True
                    assert result["is_match"] == True
                    assert result["confidence"] > 0.5
                    assert result["predicted_student"]["id"] == 1
                    assert result["predicted_student"]["name"] == "Student 1"
                    assert result["message"] == "Match found"

    def test_training_with_small_dataset(self, client, mock_s3_client, mock_supabase_client):
        """Test training with small dataset (3-5 samples per student)"""
        
        # Mock S3 response with small dataset
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'signatures/1/genuine_1.png'},
                {'Key': 'signatures/1/genuine_2.png'},
                {'Key': 'signatures/1/genuine_3.png'},
                {'Key': 'signatures/2/genuine_1.png'},
                {'Key': 'signatures/2/genuine_2.png'},
            ]
        }
        
        mock_supabase_client.table.return_value.select.return_value.execute.return_value = Mock(
            data=[
                {'id': 1, 'name': 'Student 1', 'genuine_count': 3, 'forged_count': 0},
                {'id': 2, 'name': 'Student 2', 'genuine_count': 2, 'forged_count': 0}
            ]
        )
        
        # Mock model training
        with patch('ai_training.models.signature_embedding_model.SignatureEmbeddingModel') as mock_model_class:
            mock_model = Mock()
            mock_model_class.return_value = mock_model
            
            # Mock training with real metrics
            mock_model.train_classification_only.return_value = Mock(
                history={
                    'loss': [0.8, 0.6, 0.4, 0.3, 0.2],
                    'accuracy': [0.6, 0.7, 0.8, 0.85, 0.9],
                    'val_loss': [0.9, 0.7, 0.5, 0.4, 0.3],
                    'val_accuracy': [0.5, 0.6, 0.7, 0.8, 0.85]
                }
            )
            
            # Test training with small dataset
            with patch('ai_training.api.training.get_s3_client', return_value=mock_s3_client):
                with patch('ai_training.api.training.get_supabase_client', return_value=mock_supabase_client):
                    response = client.post("/api/training/train/local")
                    assert response.status_code == 200
                    result = response.json()
                    assert result["success"] == True
                    
                    # Check that training was called with appropriate parameters
                    mock_model.train_classification_only.assert_called_once()
                    call_args = mock_model.train_classification_only.call_args
                    
                    # Check that batch size is adaptive for small dataset
                    assert call_args[1]['batch_size'] <= 8  # Should be small for small dataset
                    assert call_args[1]['epochs'] > 0

    def test_error_handling(self, client, sample_signature_image):
        """Test error handling in various scenarios"""
        
        # Test 1: Invalid file upload
        response = client.post(
            "/api/training/upload",
            files={"file": ("test.txt", b"not an image", "text/plain")},
            data={"student_id": "1", "signature_type": "genuine"}
        )
        assert response.status_code == 400
        
        # Test 2: Missing student ID
        response = client.post(
            "/api/training/upload",
            files={"file": ("test_signature.png", sample_signature_image.tobytes(), "image/png")},
            data={"signature_type": "genuine"}
        )
        assert response.status_code == 400
        
        # Test 3: Invalid student ID
        response = client.post(
            "/api/training/upload",
            files={"file": ("test_signature.png", sample_signature_image.tobytes(), "image/png")},
            data={"student_id": "invalid", "signature_type": "genuine"}
        )
        assert response.status_code == 400

    def test_data_augmentation_in_training(self, client, mock_s3_client, mock_supabase_client):
        """Test that data augmentation is applied during training"""
        
        # Mock S3 response
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'signatures/1/genuine_1.png'},
                {'Key': 'signatures/1/genuine_2.png'},
            ]
        }
        
        mock_supabase_client.table.return_value.select.return_value.execute.return_value = Mock(
            data=[{'id': 1, 'name': 'Student 1', 'genuine_count': 2, 'forged_count': 0}]
        )
        
        # Mock model training
        with patch('ai_training.models.signature_embedding_model.SignatureEmbeddingModel') as mock_model_class:
            mock_model = Mock()
            mock_model_class.return_value = mock_model
            
            # Mock training
            mock_model.train_classification_only.return_value = Mock(
                history={'loss': [0.5, 0.3], 'accuracy': [0.8, 0.9]}
            )
            
            # Test training
            with patch('ai_training.api.training.get_s3_client', return_value=mock_s3_client):
                with patch('ai_training.api.training.get_supabase_client', return_value=mock_supabase_client):
                    response = client.post("/api/training/train/local")
                    assert response.status_code == 200
                    
                    # Check that data augmentation was applied
                    mock_model.train_classification_only.assert_called_once()
                    call_args = mock_model.train_classification_only.call_args
                    
                    # Check that augmentation factor is applied
                    assert 'augmentation_factor' in call_args[1] or 'augment_data' in call_args[1]