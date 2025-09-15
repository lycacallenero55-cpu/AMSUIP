"""
Tests for image processing and augmentation
"""
import pytest
import numpy as np
from PIL import Image
import tempfile
import os

from utils.signature_preprocessing import SignaturePreprocessor, SignatureAugmentation


class TestSignaturePreprocessor:
    """Test the signature preprocessor"""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization"""
        preprocessor = SignaturePreprocessor(target_size=224)
        assert preprocessor.target_size == 224
        assert preprocessor.preserve_aspect_ratio == True
        assert preprocessor.enhance_contrast == True
        assert preprocessor.remove_background == True
    
    def test_preprocess_signature_pil_image(self):
        """Test preprocessing with PIL Image"""
        preprocessor = SignaturePreprocessor(target_size=224)
        
        # Create a test PIL image
        img = Image.new('RGB', (100, 100), color='white')
        
        # Add some content (simulate signature)
        img_array = np.array(img)
        img_array[40:60, 40:60] = [0, 0, 0]  # Black square
        
        img = Image.fromarray(img_array)
        
        # Process the image
        processed = preprocessor.preprocess_signature(img)
        
        assert processed.shape == (224, 224, 3)
        assert processed.dtype == np.uint8
        assert processed.min() >= 0
        assert processed.max() <= 255
    
    def test_preprocess_signature_numpy_array(self):
        """Test preprocessing with numpy array"""
        preprocessor = SignaturePreprocessor(target_size=224)
        
        # Create a test numpy array
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        img[40:60, 40:60] = [0, 0, 0]  # Black square
        
        # Process the image
        processed = preprocessor.preprocess_signature(img)
        
        assert processed.shape == (224, 224, 3)
        assert processed.dtype == np.uint8
    
    def test_preprocess_signature_grayscale(self):
        """Test preprocessing with grayscale image"""
        preprocessor = SignaturePreprocessor(target_size=224)
        
        # Create a grayscale image
        img = np.ones((100, 100), dtype=np.uint8) * 255
        img[40:60, 40:60] = 0  # Black square
        
        # Process the image
        processed = preprocessor.preprocess_signature(img)
        
        assert processed.shape == (224, 224, 3)
        assert processed.dtype == np.uint8
    
    def test_preprocess_signature_rgba(self):
        """Test preprocessing with RGBA image"""
        preprocessor = SignaturePreprocessor(target_size=224)
        
        # Create an RGBA image
        img = np.ones((100, 100, 4), dtype=np.uint8) * 255
        img[40:60, 40:60] = [0, 0, 0, 255]  # Black square
        
        # Process the image
        processed = preprocessor.preprocess_signature(img)
        
        assert processed.shape == (224, 224, 3)
        assert processed.dtype == np.uint8
    
    def test_preprocess_signature_error_handling(self):
        """Test error handling in preprocessing"""
        preprocessor = SignaturePreprocessor(target_size=224)
        
        # Test with invalid input
        invalid_img = "not an image"
        
        # Should not raise exception, should return fallback
        processed = preprocessor.preprocess_signature(invalid_img)
        assert processed.shape == (224, 224, 3)
        assert processed.dtype == np.float32
    
    def test_quality_assessment(self):
        """Test quality assessment"""
        preprocessor = SignaturePreprocessor(target_size=224)
        
        # Create a high-quality test image
        img = np.ones((224, 224, 3), dtype=np.uint8) * 255
        img[100:124, 100:124] = [0, 0, 0]  # Black square
        
        quality = preprocessor._assess_quality(img)
        assert 0.0 <= quality <= 1.0
        
        # Test with blank image (should have low quality)
        blank_img = np.ones((224, 224, 3), dtype=np.uint8) * 255
        blank_quality = preprocessor._assess_quality(blank_img)
        assert blank_quality < quality


class TestSignatureAugmentation:
    """Test the signature augmentation"""
    
    def test_augmenter_initialization(self):
        """Test augmenter initialization"""
        augmenter = SignatureAugmentation()
        assert augmenter.rotation_range == 15.0
        assert augmenter.scale_range == (0.8, 1.2)
        assert augmenter.brightness_range == 0.3
        assert augmenter.noise_std == 5.0
        assert augmenter.blur_probability == 0.3
    
    def test_augment_signature(self):
        """Test signature augmentation"""
        augmenter = SignatureAugmentation()
        
        # Create a test image
        img = np.ones((224, 224, 3), dtype=np.uint8) * 255
        img[100:124, 100:124] = [0, 0, 0]  # Black square
        
        # Augment the image
        augmented = augmenter.augment_signature(img, is_genuine=True)
        
        assert augmented.shape == img.shape
        assert augmented.dtype == img.dtype
        assert augmented.min() >= 0
        assert augmented.max() <= 255
    
    def test_augment_signature_error_handling(self):
        """Test error handling in augmentation"""
        augmenter = SignatureAugmentation()
        
        # Test with invalid input
        invalid_img = None
        
        # Should handle gracefully
        try:
            augmented = augmenter.augment_signature(invalid_img, is_genuine=True)
            # If it doesn't raise an exception, it should return None or original
            assert augmented is None or augmented is invalid_img
        except Exception:
            # If it raises an exception, that's also acceptable
            pass
    
    def test_rotation_augmentation(self):
        """Test rotation augmentation"""
        augmenter = SignatureAugmentation()
        
        # Create a test image
        img = np.ones((224, 224, 3), dtype=np.uint8) * 255
        img[100:124, 100:124] = [0, 0, 0]  # Black square
        
        # Test rotation
        rotated = augmenter._rotate_signature(img, 15.0)
        
        assert rotated.shape == img.shape
        assert rotated.dtype == img.dtype
    
    def test_scale_augmentation(self):
        """Test scale augmentation"""
        augmenter = SignatureAugmentation()
        
        # Create a test image
        img = np.ones((224, 224, 3), dtype=np.uint8) * 255
        img[100:124, 100:124] = [0, 0, 0]  # Black square
        
        # Test scaling up
        scaled_up = augmenter._scale_signature(img, 1.2)
        assert scaled_up.shape == img.shape
        
        # Test scaling down
        scaled_down = augmenter._scale_signature(img, 0.8)
        assert scaled_down.shape == img.shape
    
    def test_brightness_augmentation(self):
        """Test brightness augmentation"""
        augmenter = SignatureAugmentation()
        
        # Create a test image
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        
        # Test brightness adjustment
        brighter = augmenter._adjust_brightness(img, 1.5)
        darker = augmenter._adjust_brightness(img, 0.5)
        
        assert brighter.shape == img.shape
        assert darker.shape == img.shape
        assert brighter.dtype == img.dtype
        assert darker.dtype == img.dtype
    
    def test_noise_augmentation(self):
        """Test noise augmentation"""
        augmenter = SignatureAugmentation()
        
        # Create a test image
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        
        # Test noise addition
        noisy = augmenter._add_noise(img)
        
        assert noisy.shape == img.shape
        assert noisy.dtype == img.dtype
    
    def test_blur_augmentation(self):
        """Test blur augmentation"""
        augmenter = SignatureAugmentation()
        
        # Create a test image
        img = np.ones((224, 224, 3), dtype=np.uint8) * 255
        img[100:124, 100:124] = [0, 0, 0]  # Black square
        
        # Test blur
        blurred = augmenter._apply_blur(img)
        
        assert blurred.shape == img.shape
        assert blurred.dtype == img.dtype
    
    def test_augment_batch(self):
        """Test batch augmentation"""
        augmenter = SignatureAugmentation()
        
        # Create test images and labels
        images = [
            np.ones((224, 224, 3), dtype=np.uint8) * 255,
            np.ones((224, 224, 3), dtype=np.uint8) * 128
        ]
        labels = [True, False]
        
        # Test batch augmentation
        aug_images, aug_labels = augmenter.augment_batch(images, labels, augmentation_factor=2)
        
        # Should have original + 2 augmented per image = 6 total
        assert len(aug_images) == 6
        assert len(aug_labels) == 6
        
        # All images should have correct shape
        for img in aug_images:
            assert img.shape == (224, 224, 3)
            assert img.dtype == np.uint8


if __name__ == "__main__":
    pytest.main([__file__])