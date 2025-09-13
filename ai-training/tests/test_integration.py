"""
Integration tests for the AI training system
"""
import pytest
import asyncio
import tempfile
import os
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image
import io

from models.signature_embedding_model import SignatureEmbeddingModel
from models.global_signature_model import GlobalSignatureModel
from utils.s3_storage import S3Storage
from utils.s3_supabase_sync import sync_supabase_with_s3, ensure_atomic_operations
from models.database import DatabaseManager


class TestAITrainingIntegration:
    """Integration tests for AI training system"""

    @pytest.fixture
    def sample_images(self):
        """Create sample signature images for testing"""
        images = []
        for i in range(5):
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
            mock_db.return_value = mock_instance
            yield mock_instance

    def test_signature_embedding_model_training(self, sample_images):
        """Test signature embedding model training with real data"""
        model = SignatureEmbeddingModel()
        
        # Prepare training data
        genuine_images = sample_images[:3]
        forged_images = sample_images[3:]
        
        # Test data preparation
        training_data = model.prepare_training_data(
            genuine_images=genuine_images,
            forged_images=forged_images,
            student_id=1
        )
        
        assert training_data is not None
        assert len(training_data['genuine']) == 3
        assert len(training_data['forged']) == 2
        
        # Test model training (with reduced epochs for testing)
        with patch('models.signature_embedding_model.tf.keras.callbacks.EarlyStopping') as mock_early_stopping:
            mock_early_stopping.return_value = MagicMock()
            
            result = model.train_model(
                training_data=training_data,
                epochs=2,  # Reduced for testing
                batch_size=2,
                learning_rate=0.001
            )
            
            assert result['success'] is True
            assert 'accuracy' in result
            assert 'loss' in result
            assert result['accuracy'] > 0  # Should have some accuracy
            assert result['loss'] > 0  # Should have some loss

    def test_global_signature_model_training(self, sample_images):
        """Test global signature model training"""
        model = GlobalSignatureModel()
        
        # Prepare training data for multiple students
        student_data = {
            1: {'genuine': sample_images[:2], 'forged': sample_images[2:3]},
            2: {'genuine': sample_images[3:4], 'forged': sample_images[4:5]}
        }
        
        # Test data preparation
        training_data = model.prepare_training_data(student_data)
        
        assert training_data is not None
        assert len(training_data['genuine']) == 3  # 2 + 1
        assert len(training_data['forged']) == 2  # 1 + 1
        
        # Test model training
        with patch('models.global_signature_model.tf.keras.callbacks.EarlyStopping') as mock_early_stopping:
            mock_early_stopping.return_value = MagicMock()
            
            result = model.train_model(
                training_data=training_data,
                epochs=2,  # Reduced for testing
                batch_size=2,
                learning_rate=0.001
            )
            
            assert result['success'] is True
            assert 'accuracy' in result
            assert 'loss' in result

    def test_signature_verification(self, sample_images):
        """Test signature verification with trained model"""
        model = SignatureEmbeddingModel()
        
        # Train a simple model
        training_data = model.prepare_training_data(
            genuine_images=sample_images[:3],
            forged_images=sample_images[3:],
            student_id=1
        )
        
        with patch('models.signature_embedding_model.tf.keras.callbacks.EarlyStopping') as mock_early_stopping:
            mock_early_stopping.return_value = MagicMock()
            
            # Train model
            train_result = model.train_model(
                training_data=training_data,
                epochs=1,  # Minimal training for testing
                batch_size=2,
                learning_rate=0.001
            )
            
            assert train_result['success'] is True
            
            # Test verification
            test_image = sample_images[0]
            verification_result = model.verify_signature(
                test_image=test_image,
                student_id=1,
                threshold=0.5
            )
            
            assert 'is_match' in verification_result
            assert 'confidence' in verification_result
            assert 'is_genuine' in verification_result
            assert verification_result['is_genuine'] is True  # Should be true since forgery detection is disabled

    def test_s3_supabase_sync(self, mock_s3_storage, mock_database):
        """Test S3-Supabase synchronization"""
        # Mock the database manager
        db_manager = mock_database
        
        # Test atomic operations check
        result = asyncio.run(ensure_atomic_operations())
        assert result is True
        
        # Test sync function
        with patch('utils.s3_supabase_sync.db_manager', db_manager):
            result = asyncio.run(sync_supabase_with_s3())
            assert result is not None

    def test_data_augmentation(self, sample_images):
        """Test data augmentation functionality"""
        from models.signature_embedding_model import SignatureAugmentation
        
        aug = SignatureAugmentation()
        
        # Test augmentation on a sample image
        original_img = Image.open(io.BytesIO(sample_images[0]))
        
        # Test rotation
        rotated = aug.rotate_image(original_img, angle=15)
        assert rotated.size == original_img.size
        
        # Test brightness adjustment
        brightened = aug.adjust_brightness(original_img, factor=1.2)
        assert brightened.size == original_img.size
        
        # Test noise addition
        noisy = aug.add_noise(original_img, noise_factor=0.1)
        assert noisy.size == original_img.size

    def test_model_saving_and_loading(self, sample_images, mock_s3_storage):
        """Test model saving and loading functionality"""
        model = SignatureEmbeddingModel()
        
        # Train a model
        training_data = model.prepare_training_data(
            genuine_images=sample_images[:3],
            forged_images=sample_images[3:],
            student_id=1
        )
        
        with patch('models.signature_embedding_model.tf.keras.callbacks.EarlyStopping') as mock_early_stopping:
            mock_early_stopping.return_value = MagicMock()
            
            train_result = model.train_model(
                training_data=training_data,
                epochs=1,
                batch_size=2,
                learning_rate=0.001
            )
            
            assert train_result['success'] is True
            
            # Test model saving
            s3_storage = mock_s3_storage
            save_result = model.save_model_to_s3(
                s3_storage=s3_storage,
                student_id=1,
                model_uuid='test-uuid'
            )
            
            assert save_result['success'] is True
            assert 's3_key' in save_result
            assert 's3_url' in save_result

    def test_training_metrics_callback(self):
        """Test real-time training metrics callback"""
        from utils.training_callback import RealTimeMetricsCallback
        
        # Mock job progress update
        mock_update_progress = MagicMock()
        
        callback = RealTimeMetricsCallback(
            job_id='test-job-123',
            total_epochs=5
        )
        
        # Test callback initialization
        assert callback.job_id == 'test-job-123'
        assert callback.total_epochs == 5
        assert callback.current_epoch == 0

    def test_error_handling(self, sample_images):
        """Test error handling in various scenarios"""
        model = SignatureEmbeddingModel()
        
        # Test with invalid data
        with pytest.raises(Exception):
            model.prepare_training_data(
                genuine_images=[],  # Empty list
                forged_images=[],
                student_id=1
            )
        
        # Test with corrupted image data
        corrupted_images = [b'invalid_image_data']
        
        with pytest.raises(Exception):
            model.prepare_training_data(
                genuine_images=corrupted_images,
                forged_images=[],
                student_id=1
            )

    def test_performance_with_small_dataset(self, sample_images):
        """Test that the system works with small datasets (3+ images)"""
        model = SignatureEmbeddingModel()
        
        # Use minimal dataset (3 genuine, 2 forged)
        training_data = model.prepare_training_data(
            genuine_images=sample_images[:3],
            forged_images=sample_images[3:5],
            student_id=1
        )
        
        # Should not fail with small dataset
        assert training_data is not None
        assert len(training_data['genuine']) == 3
        assert len(training_data['forged']) == 2
        
        # Test training with small dataset
        with patch('models.signature_embedding_model.tf.keras.callbacks.EarlyStopping') as mock_early_stopping:
            mock_early_stopping.return_value = MagicMock()
            
            result = model.train_model(
                training_data=training_data,
                epochs=2,
                batch_size=1,  # Small batch size for small dataset
                learning_rate=0.001
            )
            
            assert result['success'] is True
            # Should achieve some accuracy even with small dataset
            assert result['accuracy'] > 0.1  # At least 10% accuracy

    def test_forgery_detection_disabled(self, sample_images):
        """Test that forgery detection is properly disabled"""
        model = SignatureEmbeddingModel()
        
        # Test verification with forgery detection disabled
        test_image = sample_images[0]
        verification_result = model.verify_signature(
            test_image=test_image,
            student_id=1,
            threshold=0.5
        )
        
        # Should always return is_genuine=True since forgery detection is disabled
        assert verification_result['is_genuine'] is True
        assert verification_result['authenticity_score'] == 0.0

    def test_model_compatibility_python_310(self):
        """Test that the system is compatible with Python 3.10.11"""
        import sys
        
        # Check Python version
        assert sys.version_info.major == 3
        assert sys.version_info.minor >= 10
        
        # Test that key imports work
        import tensorflow as tf
        import numpy as np
        import PIL
        import cv2
        
        # Test that TensorFlow is working
        assert tf.__version__ is not None
        
        # Test basic TensorFlow operations
        x = tf.constant([1, 2, 3])
        y = tf.constant([4, 5, 6])
        z = tf.add(x, y)
        
        assert z.numpy().tolist() == [5, 7, 9]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])