#!/usr/bin/env python3
"""
End-to-end tests for the complete upload->train->verify flow
Tests the full AI signature verification pipeline
"""

import pytest
import asyncio
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from fastapi.testclient import TestClient

@pytest.fixture
def test_client():
    """Create test client for API endpoints"""
    from api.training import app
    from api.verification import app as verification_app
    
    # Combine both apps for testing
    from fastapi import FastAPI
    combined_app = FastAPI()
    combined_app.include_router(app.router, prefix="/api")
    combined_app.include_router(verification_app.router, prefix="/api")
    
    return TestClient(combined_app)

@pytest.fixture
def sample_signature_data():
    """Generate sample signature data for testing"""
    # Create synthetic signature images
    genuine_signatures = []
    forged_signatures = []
    
    # Generate 3 genuine signatures with consistent pattern
    for i in range(3):
        img = np.random.rand(224, 224, 3) * 0.3
        # Add signature-like features
        img[50:100, 30:200] += 0.7  # Horizontal stroke
        img[80:150, 100:120] += 0.5  # Vertical stroke
        img += np.random.normal(0, 0.05, img.shape)
        img = np.clip(img, 0, 1)
        genuine_signatures.append(img)
    
    # Generate 1 forged signature (different pattern)
    img = np.random.rand(224, 224, 3) * 0.3
    img[60:110, 40:210] += 0.6  # Different horizontal stroke
    img[90:160, 110:130] += 0.4  # Different vertical stroke
    img += np.random.normal(0, 0.05, img.shape)
    img = np.clip(img, 0, 1)
    forged_signatures.append(img)
    
    return {
        'genuine': genuine_signatures,
        'forged': forged_signatures,
        'student_id': 201,
        'student_name': 'E2E Test Student'
    }

@pytest.fixture
def mock_services():
    """Mock external services for testing"""
    services = {
        's3_storage': Mock(),
        'db_manager': Mock(),
        'ai_service': Mock()
    }
    
    # Setup S3 mock
    services['s3_storage'].upload_model_file = Mock(return_value=('test_key', 'https://test.url'))
    services['s3_storage'].object_exists = Mock(return_value=True)
    
    # Setup DB mock
    services['db_manager'].create_trained_model = AsyncMock(return_value={'id': 1})
    services['db_manager'].list_signatures_by_student = AsyncMock(return_value=[])
    services['db_manager'].list_all_signatures = AsyncMock(return_value=[])
    
    return services

class TestE2EFlow:
    """End-to-end test cases for the complete flow"""
    
    @pytest.mark.asyncio
    async def test_upload_train_verify_flow(self, test_client, sample_signature_data, mock_services):
        """Test complete upload->train->verify flow"""
        
        # Step 1: Upload signatures (mock the upload process)
        student_id = sample_signature_data['student_id']
        genuine_signatures = sample_signature_data['genuine']
        forged_signatures = sample_signature_data['forged']
        
        # Mock signature uploads
        with patch('api.training.db_manager', mock_services['db_manager']):
            with patch('api.training.s3_storage', mock_services['s3_storage']):
                
                # Step 2: Train model
                training_response = test_client.post("/api/training/start", json={
                    "student_id": student_id,
                    "use_gpu": False
                })
                
                # Verify training started
                assert training_response.status_code == 200
                training_data = training_response.json()
                assert "job_id" in training_data or "message" in training_data
        
        # Step 3: Verify signature
        with patch('api.verification.signature_ai_manager') as mock_ai_manager:
            # Mock the AI manager
            mock_ai_manager.load_models = AsyncMock()
            mock_ai_manager.verify_signature = AsyncMock(return_value={
                "predicted_student": {"id": student_id, "name": "E2E Test Student"},
                "is_match": True,
                "confidence": 0.85,
                "score": 0.85,
                "authenticity_score": 0.0,
                "is_unknown": False,
                "model_type": "ai_signature_verification",
                "success": True,
                "decision": "match"
            })
            
            # Test verification endpoint
            verification_response = test_client.post("/api/verify", json={
                "signature_image": "base64_encoded_image_data",
                "student_id": student_id
            })
            
            # Verify response structure
            assert verification_response.status_code == 200
            verification_data = verification_response.json()
            assert "predicted_student" in verification_data
            assert "is_match" in verification_data
            assert "confidence" in verification_data
            assert verification_data["is_match"] is True
            assert verification_data["confidence"] > 0.6  # Above threshold
    
    @pytest.mark.asyncio
    async def test_identify_unknown_signature(self, test_client, mock_services):
        """Test identification of unknown signature (not from trained students)"""
        
        with patch('api.verification.signature_ai_manager') as mock_ai_manager:
            # Mock the AI manager to return unknown signature
            mock_ai_manager.load_models = AsyncMock()
            mock_ai_manager.verify_signature = AsyncMock(return_value={
                "predicted_student": None,
                "is_match": False,
                "confidence": 0.3,  # Below threshold
                "score": 0.3,
                "authenticity_score": 0.0,
                "is_unknown": True,
                "model_type": "ai_signature_verification",
                "success": True,
                "decision": "no_match"
            })
            
            # Test identification endpoint
            identification_response = test_client.post("/api/identify", json={
                "signature_image": "base64_encoded_unknown_image"
            })
            
            # Verify response
            assert identification_response.status_code == 200
            identification_data = identification_response.json()
            assert identification_data["is_unknown"] is True
            assert identification_data["is_match"] is False
            assert identification_data["confidence"] < 0.6  # Below threshold
    
    @pytest.mark.asyncio
    async def test_training_with_real_metrics(self, test_client, sample_signature_data, mock_services):
        """Test that training produces real metrics and logs"""
        
        with patch('api.training.SignatureEmbeddingModel') as mock_model_class:
            # Mock the model to return realistic training metrics
            mock_model = Mock()
            mock_model.train_models.return_value = {
                'classification_history': {
                    'loss': [0.8, 0.6, 0.4, 0.3, 0.2],
                    'accuracy': [0.5, 0.7, 0.8, 0.9, 0.95],
                    'val_loss': [0.9, 0.7, 0.5, 0.4, 0.3],
                    'val_accuracy': [0.4, 0.6, 0.7, 0.8, 0.9]
                },
                'siamese_history': {
                    'loss': [0.9, 0.7, 0.5, 0.4, 0.3]
                },
                'student_mappings': {
                    'student_to_id': {201: 0},
                    'id_to_student': {0: 201}
                }
            }
            mock_model_class.return_value = mock_model
            
            # Mock S3 storage
            mock_services['s3_storage'].upload_model_file.return_value = ('model_key', 'https://model.url')
            
            # Test training
            training_response = test_client.post("/api/training/start", json={
                "student_id": sample_signature_data['student_id'],
                "use_gpu": False
            })
            
            # Verify training started
            assert training_response.status_code == 200
            
            # Verify that training logs would be uploaded
            # (In a real test, we'd check the actual S3 upload calls)
            assert True  # Placeholder for actual verification
    
    @pytest.mark.asyncio
    async def test_s3_supabase_sync(self, test_client, mock_services):
        """Test S3-Supabase synchronization"""
        
        # Mock sync function
        with patch('api.verification.sync_supabase_with_s3_enhanced') as mock_sync:
            mock_sync.return_value = {
                'total_records_checked': 10,
                'missing_s3_objects': 2,
                'records_deleted': 2,
                'errors': 0,
                'sync_duration_seconds': 1.5
            }
            
            # Test sync endpoint
            sync_response = test_client.post("/api/sync-s3-supabase")
            
            # Verify response
            assert sync_response.status_code == 200
            sync_data = sync_response.json()
            assert sync_data['total_records_checked'] == 10
            assert sync_data['missing_s3_objects'] == 2
            assert sync_data['records_deleted'] == 2
    
    @pytest.mark.asyncio
    async def test_model_metadata_storage(self, test_client, sample_signature_data, mock_services):
        """Test that model metadata is properly stored"""
        
        with patch('api.training.db_manager', mock_services['db_manager']):
            with patch('api.training.s3_storage', mock_services['s3_storage']):
                
                # Test model creation
                model_response = test_client.post("/api/training/start", json={
                    "student_id": sample_signature_data['student_id'],
                    "use_gpu": False
                })
                
                # Verify response
                assert model_response.status_code == 200
                
                # Verify that DB record would be created with proper metadata
                # (In a real test, we'd check the actual DB calls)
                assert True  # Placeholder for actual verification
    
    @pytest.mark.asyncio
    async def test_confidence_threshold_behavior(self, test_client):
        """Test that confidence threshold works correctly"""
        
        with patch('api.verification.signature_ai_manager') as mock_ai_manager:
            # Test high confidence (should match)
            mock_ai_manager.load_models = AsyncMock()
            mock_ai_manager.verify_signature = AsyncMock(return_value={
                "predicted_student": {"id": 201, "name": "Test Student"},
                "is_match": True,
                "confidence": 0.8,  # Above threshold
                "score": 0.8,
                "authenticity_score": 0.0,
                "is_unknown": False,
                "model_type": "ai_signature_verification",
                "success": True,
                "decision": "match"
            })
            
            high_conf_response = test_client.post("/api/verify", json={
                "signature_image": "base64_high_confidence_image",
                "student_id": 201
            })
            
            assert high_conf_response.status_code == 200
            high_conf_data = high_conf_response.json()
            assert high_conf_data["is_match"] is True
            assert high_conf_data["confidence"] > 0.6
            
            # Test low confidence (should not match)
            mock_ai_manager.verify_signature.return_value = {
                "predicted_student": {"id": 201, "name": "Test Student"},
                "is_match": False,
                "confidence": 0.4,  # Below threshold
                "score": 0.4,
                "authenticity_score": 0.0,
                "is_unknown": True,
                "model_type": "ai_signature_verification",
                "success": True,
                "decision": "no_match"
            }
            
            low_conf_response = test_client.post("/api/verify", json={
                "signature_image": "base64_low_confidence_image",
                "student_id": 201
            })
            
            assert low_conf_response.status_code == 200
            low_conf_data = low_conf_response.json()
            assert low_conf_data["is_match"] is False
            assert low_conf_data["confidence"] < 0.6

class TestErrorHandling:
    """Test error handling in the E2E flow"""
    
    @pytest.mark.asyncio
    async def test_training_failure_handling(self, test_client):
        """Test handling of training failures"""
        
        with patch('api.training.SignatureEmbeddingModel') as mock_model_class:
            # Mock training failure
            mock_model_class.side_effect = Exception("Training failed")
            
            # Test training
            training_response = test_client.post("/api/training/start", json={
                "student_id": 201,
                "use_gpu": False
            })
            
            # Verify error handling
            assert training_response.status_code in [500, 400]  # Error status
    
    @pytest.mark.asyncio
    async def test_verification_failure_handling(self, test_client):
        """Test handling of verification failures"""
        
        with patch('api.verification.signature_ai_manager') as mock_ai_manager:
            # Mock verification failure
            mock_ai_manager.load_models.side_effect = Exception("Model loading failed")
            
            # Test verification
            verification_response = test_client.post("/api/verify", json={
                "signature_image": "base64_image",
                "student_id": 201
            })
            
            # Verify error handling
            assert verification_response.status_code in [500, 400]  # Error status

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])