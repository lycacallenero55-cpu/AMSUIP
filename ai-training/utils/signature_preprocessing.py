"""
Production-Ready Signature Preprocessing Pipeline
Advanced image processing specifically designed for signature verification
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging
from typing import Tuple, Optional, Union, List
import math

logger = logging.getLogger(__name__)

class SignaturePreprocessor:
    """
    Advanced signature preprocessing pipeline with:
    - Signature-specific image enhancement
    - Stroke pattern preservation
    - Noise reduction and cleanup
    - Geometric normalization
    - Quality assessment
    """
    
    def __init__(self, 
                 target_size: int = 224,
                 preserve_aspect_ratio: bool = True,
                 enhance_contrast: bool = True,
                 remove_background: bool = True):
        
        self.target_size = target_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.enhance_contrast = enhance_contrast
        self.remove_background = remove_background
        
        logger.info(f"Initialized SignaturePreprocessor: {target_size}x{target_size} output")
    
    def preprocess_signature(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Complete signature preprocessing pipeline
        """
        try:
            # Convert to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Ensure RGB format
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = image[:, :, :3]  # Remove alpha channel
            
            # Step 1: Background removal and cleanup
            if self.remove_background:
                image = self._remove_background(image)
            
            # Step 2: Signature enhancement
            if self.enhance_contrast:
                image = self._enhance_signature(image)
            
            # Step 3: Geometric normalization
            image = self._geometric_normalization(image)
            
            # Step 4: Final resizing and normalization
            image = self._final_resize(image)
            
            # Step 5: Quality validation
            quality_score = self._assess_quality(image)
            if quality_score < 0.3:
                logger.warning(f"Low quality signature detected: {quality_score:.3f}")
            
            return image
            
        except Exception as e:
            logger.error(f"Signature preprocessing failed: {e}")
            # Fallback to simple processing
            return self._fallback_preprocessing(image)
    
    def _remove_background(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced background removal for signatures
        """
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Adaptive thresholding for better background removal
        # Use Otsu's method with additional processing
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if signature is dark on light background
        if np.mean(binary) > 127:
            binary = 255 - binary
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Remove small noise
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (image.shape[0] * image.shape[1]) * 0.001  # 0.1% of image area
        
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                cv2.fillPoly(binary, [contour], 0)
        
        # Create mask and apply to original image
        mask = binary.astype(np.uint8)
        
        # Apply mask to each channel
        result = image.copy()
        for i in range(3):
            result[:, :, i] = np.where(mask == 0, 255, result[:, :, i])
        
        return result
    
    def _enhance_signature(self, image: np.ndarray) -> np.ndarray:
        """
        Signature-specific enhancement
        """
        # Convert to PIL for enhancement
        pil_image = Image.fromarray(image)
        
        # Enhance contrast specifically for signatures
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.5)  # Increase contrast
        
        # Enhance sharpness to make strokes clearer
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.3)
        
        # Convert back to numpy
        enhanced = np.array(pil_image)
        
        # Additional OpenCV enhancements
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def _geometric_normalization(self, image: np.ndarray) -> np.ndarray:
        """
        Geometric normalization for consistent signature orientation
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Find signature contours
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(binary) > 127:
            binary = 255 - binary
        
        # Find the largest contour (main signature)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        # Correct angle if needed
        if angle < -45:
            angle = 90 + angle
        
        # Rotate image to correct orientation
        if abs(angle) > 1:  # Only rotate if significant angle
            h, w = image.shape[:2]
            center = (float(w // 2), float(h // 2))  # Ensure center is float tuple
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                 flags=cv2.INTER_CUBIC, 
                                 borderMode=cv2.BORDER_CONSTANT, 
                                 borderValue=255)
        
        return image
    
    def _final_resize(self, image: np.ndarray) -> np.ndarray:
        """
        Final resizing with proper aspect ratio handling
        """
        h, w = image.shape[:2]
        
        if self.preserve_aspect_ratio:
            # Calculate scaling factor to fit in target size
            scale = min(self.target_size / w, self.target_size / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize with high-quality interpolation
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Create square canvas and center the signature
            canvas = np.full((self.target_size, self.target_size, 3), 255, dtype=np.uint8)
            
            # Calculate position to center the signature
            y_offset = (self.target_size - new_h) // 2
            x_offset = (self.target_size - new_w) // 2
            
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            
            return canvas
        else:
            # Direct resize to target size
            return cv2.resize(image, (self.target_size, self.target_size), 
                            interpolation=cv2.INTER_CUBIC)
    
    def _assess_quality(self, image: np.ndarray) -> float:
        """
        Assess signature quality for validation
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate various quality metrics
        
        # 1. Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Contrast (standard deviation)
        contrast = np.std(gray)
        
        # 3. Signature density (non-white pixels)
        non_white = np.sum(gray < 240)  # Pixels that are not white
        density = non_white / (gray.shape[0] * gray.shape[1])
        
        # 4. Edge density (Canny edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 5. Structural complexity (entropy)
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Combine metrics into quality score
        quality_score = (
            min(laplacian_var / 1000, 1.0) * 0.3 +  # Sharpness
            min(contrast / 100, 1.0) * 0.2 +         # Contrast
            min(density * 10, 1.0) * 0.2 +           # Density
            min(edge_density * 20, 1.0) * 0.2 +      # Edge density
            min(entropy / 8, 1.0) * 0.1              # Entropy
        )
        
        return float(quality_score)
    
    def _fallback_preprocessing(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Fallback preprocessing for error cases
        """
        try:
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Simple resize and normalization
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            image = cv2.resize(image, (self.target_size, self.target_size), 
                             interpolation=cv2.INTER_CUBIC)
            
            # Normalize to [0, 1]
            if image.max() > 1.0:
                image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            logger.error(f"Fallback preprocessing failed: {e}")
            # Return a blank white image as last resort
            return np.ones((self.target_size, self.target_size, 3), dtype=np.float32)


class SignatureAugmentation:
    """
    Advanced signature augmentation for training data generation
    """
    
    def __init__(self, 
                 rotation_range: float = 15.0,
                 scale_range: Tuple[float, float] = (0.8, 1.2),
                 brightness_range: float = 0.3,
                 noise_std: float = 5.0,
                 blur_probability: float = 0.3):
        
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.brightness_range = brightness_range
        self.noise_std = noise_std
        self.blur_probability = blur_probability
        
        logger.info("Initialized SignatureAugmentation")
    
    def augment_signature(self, image: np.ndarray, is_genuine: bool = True) -> np.ndarray:
        """
        Apply signature-specific augmentations
        """
        augmented = image.copy()
        
        # Apply augmentations with different intensities for genuine vs forged
        intensity = 1.0 if is_genuine else 0.6
        
        # Rotation (simulate natural writing variations)
        if np.random.random() < 0.8 * intensity:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            augmented = self._rotate_signature(augmented, angle)
        
        # Scale (simulate different writing sizes)
        if np.random.random() < 0.7 * intensity:
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            augmented = self._scale_signature(augmented, scale)
        
        # Brightness (simulate different lighting conditions)
        if np.random.random() < 0.6 * intensity:
            factor = np.random.uniform(1 - self.brightness_range, 1 + self.brightness_range)
            augmented = self._adjust_brightness(augmented, factor)
        
        # Noise (simulate camera/scanning noise)
        if np.random.random() < 0.4 * intensity:
            augmented = self._add_noise(augmented)
        
        # Blur (simulate camera focus issues)
        if np.random.random() < self.blur_probability * intensity:
            augmented = self._apply_blur(augmented)
        
        # Elastic distortion (simulate paper texture)
        if np.random.random() < 0.3 * intensity:
            augmented = self._elastic_distortion(augmented)
        
        # Perspective distortion (simulate camera angle)
        if np.random.random() < 0.2 * intensity:
            augmented = self._perspective_distortion(augmented)
        
        # Pen pressure simulation (vary line thickness)
        if np.random.random() < 0.3 * intensity:
            augmented = self._simulate_pen_pressure(augmented)
        
        return augmented
    
    def _rotate_signature(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate signature with proper boundary handling"""
        h, w = image.shape[:2]
        center = (float(w // 2), float(h // 2))  # Ensure center is float tuple
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new boundaries
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # Adjust rotation matrix for translation
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=255)
        
        # Resize back to original size
        return cv2.resize(rotated, (w, h), interpolation=cv2.INTER_CUBIC)
    
    def _scale_signature(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Scale signature with proper centering"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        if scale > 1.0:
            # Center crop
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            return resized[start_h:start_h + h, start_w:start_w + w]
        else:
            # Pad with white
            result = np.full((h, w, 3), 255, dtype=np.uint8)
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            result[start_h:start_h + new_h, start_w:start_w + new_w] = resized
            return result
    
    def _adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust brightness with gamma correction"""
        # Use gamma correction for more realistic brightness changes
        gamma = 1.0 / factor if factor > 0 else 1.0
        inv_gamma = 1.0 / gamma
        
        # Create lookup table
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in range(256)]).astype(np.uint8)
        
        # Apply LUT to each channel separately to avoid OpenCV LUT errors
        result = image.copy()
        for c in range(image.shape[2]):
            result[:, :, c] = cv2.LUT(image[:, :, c], table)
        
        return result
    
    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add realistic noise"""
        noise = np.random.normal(0, self.noise_std, image.shape)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)
    
    def _apply_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply various blur types"""
        blur_type = np.random.choice(['gaussian', 'motion'])
        
        if blur_type == 'gaussian':
            kernel_size = np.random.choice([3, 5])
            sigma = np.random.uniform(0.5, 1.5)
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        else:  # motion blur
            kernel_size = np.random.choice([5, 7])
            angle = np.random.uniform(0, 180)
            return self._motion_blur(image, kernel_size, angle)
    
    def _motion_blur(self, image: np.ndarray, kernel_size: int, angle: float) -> np.ndarray:
        """Apply directional motion blur"""
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        center = (float(kernel_size // 2), float(kernel_size // 2))  # Ensure center is float tuple
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
        
        return cv2.filter2D(image, -1, kernel)
    
    def _elastic_distortion(self, image: np.ndarray) -> np.ndarray:
        """Apply elastic distortion for natural variations"""
        random_state = np.random.RandomState(None)
        shape = image.shape[:2]
        
        # Generate random displacement fields
        dx = cv2.GaussianBlur(
            (random_state.rand(*shape) * 2 - 1), 
            (17, 17), 
            6.0
        ) * 8.0
        
        dy = cv2.GaussianBlur(
            (random_state.rand(*shape) * 2 - 1), 
            (17, 17), 
            6.0
        ) * 8.0
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        return cv2.remap(image, map_x, map_y, 
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)
    
    def augment_batch(self, images: List[np.ndarray], labels: List[bool], 
                     augmentation_factor: int = 3) -> Tuple[List[np.ndarray], List[bool]]:
        """
        Augment a batch of signatures
        """
        augmented_images = []
        augmented_labels = []
        
        for image, label in zip(images, labels):
            # Always include original
            augmented_images.append(image)
            augmented_labels.append(label)
            
            # Create augmented versions
            for _ in range(augmentation_factor):
                aug_img = self.augment_signature(image, is_genuine=label)
                augmented_images.append(aug_img)
                augmented_labels.append(label)
        
        return augmented_images, augmented_labels
    
    def _perspective_distortion(self, image: np.ndarray) -> np.ndarray:
        """Apply perspective distortion to simulate camera angle"""
        h, w = image.shape[:2]
        
        # Define source points (corners of the image)
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Define destination points with slight perspective distortion
        max_offset = min(w, h) * 0.1
        dst_points = np.float32([
            [np.random.uniform(-max_offset, max_offset), np.random.uniform(-max_offset, max_offset)],
            [w + np.random.uniform(-max_offset, max_offset), np.random.uniform(-max_offset, max_offset)],
            [w + np.random.uniform(-max_offset, max_offset), h + np.random.uniform(-max_offset, max_offset)],
            [np.random.uniform(-max_offset, max_offset), h + np.random.uniform(-max_offset, max_offset)]
        ])
        
        # Get perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply perspective transformation
        distorted = cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        
        return distorted
    
    def _simulate_pen_pressure(self, image: np.ndarray) -> np.ndarray:
        """Simulate varying pen pressure by adjusting line thickness"""
        # Convert to grayscale for morphological operations
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Create varying kernel sizes to simulate different pen pressures
        kernel_sizes = [1, 2, 3]
        weights = [0.3, 0.5, 0.2]  # Probability weights for each kernel size
        
        # Randomly select kernel size based on weights
        kernel_size = np.random.choice(kernel_sizes, p=weights)
        
        if kernel_size > 1:
            # Dilate to make lines thicker (simulate more pressure)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            gray = cv2.dilate(gray, kernel, iterations=1)
        
        # Convert back to RGB
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        return result