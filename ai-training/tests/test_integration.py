"""
Integration tests for the AI training pipeline
"""
import pytest
import numpy as np
from PIL import Image
import tempfile
import os

from models.signature_embedding_model import SignatureEmbeddingModel
from utils.signature_preprocessing import SignaturePreprocessor, SignatureAugmentation


class TestTrainingIntegration:
    """Test the complete training pipeline"""
    
    def test_model_creation_and_training(self):
        """Test complete model creation and training pipeline"""
        # Create test data
        training_data = {
            "student_1": {
                "genuine": [self._create_test_image() for _ in range(3)],
                "forged": []
            },
            "student_2": {
                "genuine": [self._create_test_image() for _ in range(3)],
                "forged": []
            }
        }
        
        # Create model
        model = SignatureEmbeddingModel(max_students=10)
        
        # Test data preparation
        X, y = model.prepare_training_data(training_data)
        
        assert X.shape[0] > 0
        assert y.shape[0] == X.shape[0]
        assert y.shape[1] == 2  # 2 students
        
        # Test model creation
        classification_head = model.create_classification_head(num_students=2)
        assert classification_head is not None
    
    def test_preprocessing_pipeline(self):
        """Test the complete preprocessing pipeline"""
        preprocessor = SignaturePreprocessor(target_size=224)
        
        # Test with various image types
        test_images = [
            self._create_test_image(),
            self._create_test_image_grayscale(),
            self._create_test_image_rgba()
        ]
        
        for img in test_images:
            processed = preprocessor.preprocess_signature(img)
            assert processed.shape == (224, 224, 3)
            assert processed.dtype == np.uint8
    
    def test_augmentation_pipeline(self):
        """Test the complete augmentation pipeline"""
        augmenter = SignatureAugmentation()
        
        # Create test image
        img = self._create_test_image()
        
        # Test various augmentations
        augmented_images = []
        for _ in range(5):
            aug_img = augmenter.augment_signature(img, is_genuine=True)
            augmented_images.append(aug_img)
        
        # All augmented images should have correct shape
        for aug_img in augmented_images:
            assert aug_img.shape == img.shape
            assert aug_img.dtype == img.dtype
    
    def _create_test_image(self):
        """Create a test RGB image"""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        img[40:60, 40:60] = [0, 0, 0]  # Black square
        return Image.fromarray(img)
    
    def _create_test_image_grayscale(self):
        """Create a test grayscale image"""
        img = np.ones((100, 100), dtype=np.uint8) * 255
        img[40:60, 40:60] = 0  # Black square
        return Image.fromarray(img)
    
    def _create_test_image_rgba(self):
        """Create a test RGBA image"""
        img = np.ones((100, 100, 4), dtype=np.uint8) * 255
        img[40:60, 40:60] = [0, 0, 0, 255]  # Black square
        return Image.fromarray(img)


if __name__ == "__main__":
    pytest.main([__file__])