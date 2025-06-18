import numpy as np
from PIL import Image as PILImage
import cv2
import io
import logging
from typing import Dict
from django.core.files import File
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def generate_prediction_outputs_yolo(model, original_image_path: str, confidence_threshold: float,
                                   picture_filename: str) -> Dict[str, io.BytesIO]:
    """
    Generate original and overlay images using YOLO's native methods.
    
    Args:
        model: YOLO model instance
        original_image_path: Path to the original image file
        confidence_threshold: Confidence threshold for YOLO predictions
        picture_filename: Base filename for generating output names
    
    Returns:
        Dictionary with BytesIO objects for each image type:
        - 'original': Original image
        - 'overlay': Original with green overlay on detected regions (YOLO style)
    """
    try:
        logger.info(f"Generating YOLO prediction outputs for: {picture_filename}")
        
        image = cv2.imread(original_image_path)
        if image is None:
            raise ValueError(f"Image at path {original_image_path} could not be loaded.")
        
        # Run YOLO prediction
        results = model(image, conf=confidence_threshold)

        # Use ultralytics' built-in plot() method for a robust overlay
        # This returns a BGR numpy array with masks and boxes drawn
        overlay_image_bgr = results[0].plot(labels=False, conf=False)

        # --- Create original image output ---
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_pil = PILImage.fromarray(original_rgb)
        original_bytes = io.BytesIO()
        original_pil.save(original_bytes, format='PNG')
        original_bytes.seek(0)

        # --- Create overlay image output ---
        overlay_rgb = cv2.cvtColor(overlay_image_bgr, cv2.COLOR_BGR2RGB)
        overlay_pil = PILImage.fromarray(overlay_rgb)
        overlay_bytes = io.BytesIO()
        overlay_pil.save(overlay_bytes, format='PNG')
        overlay_bytes.seek(0)
        
        outputs = {
            'original': original_bytes,
            'overlay': overlay_bytes,
        }
        
        logger.info(f"Successfully generated YOLO prediction outputs for: {picture_filename}")
        return outputs
        
    except Exception as e:
        logger.error(f"Error generating YOLO prediction outputs for {picture_filename}: {str(e)}")
        raise


def generate_prediction_outputs_unet(original_image_path: str, mask_image: PILImage.Image, 
                                    picture_filename: str) -> Dict[str, io.BytesIO]:
    """
    Generate original and overlay images for UNet predictions.
    
    Args:
        original_image_path: Path to the original image file
        mask_image: Predicted mask as PIL Image
        picture_filename: Base filename for generating output names
    
    Returns:
        Dictionary with BytesIO objects for each image type:
        - 'original': Original image
        - 'overlay': Original with green overlay on detected regions  
    """
    try:
        logger.info(f"Generating UNet prediction outputs for: {picture_filename}")
        
        # Load original image
        original_image = PILImage.open(original_image_path).convert('RGB')
        
        # Convert mask to numpy array
        mask_array = np.array(mask_image.convert('L'))
        
        # Create overlay image (similar to YOLO method)
        original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        overlay_image = original_cv.copy()
        
        # Apply green overlay where mask is detected
        overlay_image[mask_array > 0] = [0, 255, 0]  # BGR format
        
        # Convert back to RGB
        overlay_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
        overlay_pil = PILImage.fromarray(overlay_rgb)
        
        # Convert images to BytesIO objects
        outputs = {}
        
        # Original image
        original_bytes = io.BytesIO()
        original_image.save(original_bytes, format='PNG')
        original_bytes.seek(0)
        outputs['original'] = original_bytes
        
        # Overlay image
        overlay_bytes = io.BytesIO()
        overlay_pil.save(overlay_bytes, format='PNG')
        overlay_bytes.seek(0)
        outputs['overlay'] = overlay_bytes
        
        logger.info(f"Successfully generated UNet prediction outputs for: {picture_filename}")
        return outputs
        
    except Exception as e:
        logger.error(f"Error generating UNet prediction outputs for {picture_filename}: {str(e)}")
        raise


def generate_prediction_outputs(original_image_path: str, mask_image: PILImage.Image, 
                               picture_filename: str, model=None, confidence_threshold: float = None) -> Dict[str, io.BytesIO]:
    """
    Generate original and overlay images from prediction results.
    Automatically detects if YOLO model is used and uses appropriate method.
    
    Args:
        original_image_path: Path to the original image file
        mask_image: Predicted mask as PIL Image (used for UNet)
        picture_filename: Base filename for generating output names
        model: Model instance (if YOLO, will use YOLO method)
        confidence_threshold: Confidence threshold (for YOLO)
    
    Returns:
        Dictionary with BytesIO objects for each image type:
        - 'original': Original image
        - 'overlay': Original with green overlay on detected regions  
    """
    # Check if this is a YOLO model
    if model is not None and isinstance(model, YOLO):
        return generate_prediction_outputs_yolo(model, original_image_path, confidence_threshold, picture_filename)
    else:
        return generate_prediction_outputs_unet(original_image_path, mask_image, picture_filename)


def save_prediction_outputs(outputs: Dict[str, io.BytesIO], base_filename: str, 
                          upload_to_prefix: str = 'predictions/') -> Dict[str, File]:
    """
    Save prediction outputs as Django File objects.
    
    Args:
        outputs: Dictionary of BytesIO objects from generate_prediction_outputs
        base_filename: Base filename (without extension)
        upload_to_prefix: Prefix for upload path
    
    Returns:
        Dictionary of Django File objects ready for model field assignment
    """
    try:
        files = {}
        
        # Generate filenames for each output type
        filenames = {
            'original': f"{base_filename}_original.png",
            'overlay': f"{base_filename}_overlay.png"
        }
        
        # Create Django File objects
        for output_type, byte_stream in outputs.items():
            filename = filenames[output_type]
            django_file = File(byte_stream, name=filename)
            files[output_type] = django_file
            
        logger.info(f"Successfully created Django File objects for: {base_filename}")
        return files
        
    except Exception as e:
        logger.error(f"Error saving prediction outputs for {base_filename}: {str(e)}")
        raise 