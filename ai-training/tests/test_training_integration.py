"""
Integration tests for AI training pipeline
"""
import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch
from ai_training.models.signature_embedding_model import SignatureEmbeddingModel
from ai_training.utils.augmentation import AdvancedSignatureAugmentation


class TestTrainingIntegration:
    """Integration tests for training pipeline"""

    @pytest.fixture
    def sample_signature_data(self):
        """Generate sample signature data for testing"""
        # Create sample signature images (28x28 grayscale)
        genuine_images = []
        for i in range(5):  # 5 genuine samples per student
            # Create a simple signature pattern
            img = np.random.rand(28, 28, 1) * 255
            # Add some structure to make it look like a signature
            img[10:18, 5:23] = 0  # Horizontal line
            img[5:13, 8:16] = 0   # Vertical line
            genuine_images.append(img.astype(np.uint8))
        
        return {
            'genuine_images': genuine_images,
            'student_ids': [1, 1, 1, 1, 1],  # All from student 1
            'labels': [1, 1, 1, 1, 1]
        }

    def test_data_augmentation_pipeline(self, sample_signature_data):
        """Test that data augmentation creates diverse samples"""
        augmentation = AdvancedSignatureAugmentation()
        
        original_images = sample_signature_data['genuine_images']
        augmented_images = []
        
        for img in original_images:
            # Apply augmentation
            augmented = augmentation.augment_signature(img)
            augmented_images.append(augmented)
        
        # Check that augmented images are different from originals
        for orig, aug in zip(original_images, augmented_images):
            assert not np.array_equal(orig, aug)
            assert aug.shape == orig.shape
            assert aug.dtype == orig.dtype

    def test_model_architecture(self):
        """Test that model architecture is correct"""
        model = SignatureEmbeddingModel()
        
        # Test that model has correct structure
        assert hasattr(model, 'base_model')
        assert hasattr(model, 'embedding_model')
        assert hasattr(model, 'classification_model')
        
        # Test that base model is MobileNetV2
        assert 'mobilenetv2' in model.base_model.name.lower()

    def test_training_with_small_dataset(self, sample_signature_data):
        """Test training with small dataset (5 samples)"""
        model = SignatureEmbeddingModel()
        
        # Prepare training data
        X = np.array(sample_signature_data['genuine_images'])
        y = np.array(sample_signature_data['labels'])
        
        # Test that model can be compiled
        model.compile_embedding_model()
        model.compile_classification_model(num_classes=2)  # 2 classes: student 1 and unknown
        
        # Test that model can be trained (just one epoch for testing)
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.h5')
            
            # Test training
            history = model.train_classification_only(
                X, y, 
                epochs=1, 
                batch_size=2,
                validation_split=0.2,
                model_path=model_path
            )
            
            # Check that training history is recorded
            assert 'loss' in history.history
            assert 'accuracy' in history.history
            assert len(history.history['loss']) == 1

    def test_model_saving_and_loading(self, sample_signature_data):
        """Test that model can be saved and loaded"""
        model = SignatureEmbeddingModel()
        
        # Prepare training data
        X = np.array(sample_signature_data['genuine_images'])
        y = np.array(sample_signature_data['labels'])
        
        # Compile and train model
        model.compile_embedding_model()
        model.compile_classification_model(num_classes=2)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.h5')
            
            # Train model
            model.train_classification_only(
                X, y, 
                epochs=1, 
                batch_size=2,
                model_path=model_path
            )
            
            # Test that model file exists
            assert os.path.exists(model_path)
            assert os.path.getsize(model_path) > 1000  # Model should be substantial
            
            # Test loading model
            loaded_model = SignatureEmbeddingModel()
            loaded_model.load_classification_model(model_path)
            
            # Test that loaded model can make predictions
            test_img = X[0:1]  # First image
            prediction = loaded_model.classification_model.predict(test_img)
            
            assert prediction.shape == (1, 2)  # 2 classes
            assert np.allclose(prediction.sum(axis=1), 1.0)  # Probabilities sum to 1

    def test_adaptive_batch_size(self, sample_signature_data):
        """Test that adaptive batch size works correctly"""
        model = SignatureEmbeddingModel()
        
        # Test with different dataset sizes
        test_cases = [
            (3, 1),   # Very small dataset
            (5, 2),   # Small dataset
            (10, 4),  # Medium dataset
            (20, 8),  # Larger dataset
        ]
        
        for dataset_size, expected_batch_size in test_cases:
            # Create test data
            X = np.random.rand(dataset_size, 28, 28, 1)
            y = np.random.randint(0, 2, dataset_size)
            
            # Calculate adaptive batch size
            adaptive_batch_size = min(32, max(4, len(X) // 4))
            
            assert adaptive_batch_size == expected_batch_size
            assert adaptive_batch_size >= 4
            assert adaptive_batch_size <= 32

    def test_training_metrics_logging(self, sample_signature_data):
        """Test that training metrics are properly logged"""
        model = SignatureEmbeddingModel()
        
        # Prepare training data
        X = np.array(sample_signature_data['genuine_images'])
        y = np.array(sample_signature_data['labels'])
        
        # Compile model
        model.compile_embedding_model()
        model.compile_classification_model(num_classes=2)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test CSV logging
            csv_log_path = os.path.join(temp_dir, 'training_log.csv')
            
            # Train with CSV logging
            history = model.train_classification_only(
                X, y, 
                epochs=2, 
                batch_size=2,
                csv_log_path=csv_log_path
            )
            
            # Check that CSV log was created
            assert os.path.exists(csv_log_path)
            
            # Check that log contains expected columns
            with open(csv_log_path, 'r') as f:
                content = f.read()
                assert 'epoch' in content
                assert 'loss' in content
                assert 'accuracy' in content

    def test_verification_with_trained_model(self, sample_signature_data):
        """Test verification with a trained model"""
        model = SignatureEmbeddingModel()
        
        # Prepare training data
        X = np.array(sample_signature_data['genuine_images'])
        y = np.array(sample_signature_data['labels'])
        
        # Compile and train model
        model.compile_embedding_model()
        model.compile_classification_model(num_classes=2)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.h5')
            
            # Train model
            model.train_classification_only(
                X, y, 
                epochs=1, 
                batch_size=2,
                model_path=model_path
            )
            
            # Test verification
            test_img = X[0]  # First image
            result = model.verify_signature(test_img, student_id=1)
            
            # Check that result has expected structure
            assert 'is_match' in result
            assert 'confidence' in result
            assert 'predicted_student_id' in result
            assert 'is_unknown' in result
            
            # Check that confidence is a valid probability
            assert 0 <= result['confidence'] <= 1