from PIL import Image, ImageEnhance
import io
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

def validate_image(file) -> bool:
    """Validate if uploaded file is a valid image"""
    try:
        # Check file size (max 10MB)
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            return False
        
        # Try to open as image
        file.file.seek(0)
        image_data = file.file.read()
        file.file.seek(0)
        
        image = Image.open(io.BytesIO(image_data))
        image.verify()
        
        return True
    
    except Exception as e:
        logger.error(f"Image validation failed: {e}")
        return False

def preprocess_image(image: Image.Image, target_size: int) -> Image.Image:
    """Preprocess image for model input with normalization, deskew, crop, pad, and resize.

    Steps:
    - Convert to grayscale
    - Enhance contrast
    - Deskew (estimate angle from binary and rotate)
    - Crop to tight bounding box
    - Pad to square and center
    - Resize to target and convert to RGB
    """
    try:
        # 1) Convert to grayscale
        gray = image.convert('L')

        # 2) Enhance contrast
        gray = ImageEnhance.Contrast(gray).enhance(1.5)

        # 3) Deskew + 4) Crop using OpenCV
        np_img = np.array(gray)
        # Binarize (Otsu)
        _, thresh = cv2.threshold(np_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Invert so signature strokes are foreground
        inv = 255 - thresh

        # Estimate skew via minAreaRect on foreground points
        coords = np.column_stack(np.where(inv > 0))
        angle = 0.0
        if coords.size > 0:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
        # Rotate to deskew
        (h, w) = np_img.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        de_skew = cv2.warpAffine(np_img, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # Re-binarize and crop to bounding box
        _, bw = cv2.threshold(de_skew, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inv2 = 255 - bw
        ys, xs = np.where(inv2 > 0)
        if ys.size > 0 and xs.size > 0:
            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()
            cropped = de_skew[y1:y2 + 1, x1:x2 + 1]
        else:
            cropped = de_skew

        # 5) Pad to square and center with margin
        h2, w2 = cropped.shape[:2]
        side = max(h2, w2)
        margin = int(0.05 * side)
        canvas = np.full((side + 2 * margin, side + 2 * margin), 255, dtype=np.uint8)
        y_off = (canvas.shape[0] - h2) // 2
        x_off = (canvas.shape[1] - w2) // 2
        canvas[y_off:y_off + h2, x_off:x_off + w2] = cropped

        # 6) Resize to target and convert to RGB
        out = cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_AREA)
        out_pil = Image.fromarray(out).convert('RGB')
        
        # FIXED: Return PIL Image for consistency with signature preprocessing
        # The signature preprocessing pipeline expects PIL Image input
        return out_pil

    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        # Fallback: simple resize and convert
        try:
            fallback = image.convert('RGB').resize((target_size, target_size), Image.Resampling.LANCZOS)
            logger.warning("Using fallback preprocessing")
            return fallback
        except Exception as fallback_error:
            logger.error(f"Fallback preprocessing also failed: {fallback_error}")
            raise

def get_image_quality_score(image: Image.Image) -> float:
    """Calculate a basic image quality score"""
    try:
        # Convert to grayscale for analysis
        gray = image.convert('L')
        
        # Calculate variance (higher variance = more detail)
        import numpy as np
        img_array = np.array(gray)
        variance = np.var(img_array)
        
        # Calculate improved quality score combining variance and sharpness
        sharpness = cv2.Laplacian(img_array, cv2.CV_64F).var()
        
        # Combine sharpness and variance for better quality assessment
        quality_score = min((sharpness/1000) * (variance/5000), 1.0)
        
        return float(quality_score)
    
    except Exception as e:
        logger.error(f"Quality score calculation failed: {e}")
        return 0.0