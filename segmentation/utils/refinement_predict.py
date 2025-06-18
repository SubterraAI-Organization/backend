import numpy as np
import torch
from PIL import Image as PILImage
from torchvision.transforms.v2 import functional as F
from ultralytics import YOLO
import logging

from .predict import predict
from .masks import threshold

logger = logging.getLogger(__name__)


def apply_refinement_mask(original_image_array: np.ndarray, initial_mask_array: np.ndarray) -> np.ndarray:
    """
    Apply initial mask to original image, masking out detected areas with black pixels.
    
    Args:
        original_image_array: Original image as numpy array (H, W, 3)
        initial_mask_array: Initial mask as numpy array (H, W) with values 0-255
    
    Returns:
        Masked image array where detected areas are set to black
    """
    # Normalize mask to 0-1 range
    mask_normalized = initial_mask_array.astype(np.float32) / 255.0
    
    # Expand mask to 3 channels
    mask_3channel = np.stack([mask_normalized, mask_normalized, mask_normalized], axis=2)
    
    # Apply inverse mask (mask out detected areas with black)
    # Where mask is 1 (detected), set to black (0)
    # Where mask is 0 (not detected), keep original pixel values
    masked_image = original_image_array * (1 - mask_3channel)
    
    return masked_image.astype(np.uint8)


def combine_masks(initial_mask: np.ndarray, refined_mask: np.ndarray, combination_method: str = 'additive') -> np.ndarray:
    """
    Combine initial and refined masks intelligently.
    
    Args:
        initial_mask: Initial prediction mask (H, W) with values 0-255
        refined_mask: Refined prediction mask (H, W) with values 0-255
        combination_method: Method to combine masks ('additive', 'max', 'weighted')
    
    Returns:
        Combined mask as numpy array
    """
    # Normalize masks to 0-1 range
    initial_norm = initial_mask.astype(np.float32) / 255.0
    refined_norm = refined_mask.astype(np.float32) / 255.0
    
    if combination_method == 'additive':
        # Add masks and clamp to [0, 1]
        combined = np.clip(initial_norm + refined_norm, 0, 1)
    elif combination_method == 'max':
        # Take maximum of both masks
        combined = np.maximum(initial_norm, refined_norm)
    elif combination_method == 'weighted':
        # Weighted combination (favor initial prediction more)
        combined = 0.7 * initial_norm + 0.3 * refined_norm
        combined = np.clip(combined, 0, 1)
    else:
        # Default to additive
        combined = np.clip(initial_norm + refined_norm, 0, 1)
    
    # Convert back to 0-255 range
    return (combined * 255).astype(np.uint8)


def refine_prediction(model, image_path: str, initial_mask: PILImage.Image, 
                     confidence_threshold: float = None, area_threshold: int = 15,
                     combination_method: str = 'additive') -> PILImage.Image:
    """
    Refine prediction by re-predicting on non-root areas of the image.
    
    Args:
        model: The segmentation model (UNET or YOLO)
        image_path: Path to the original image
        initial_mask: Initial prediction mask as PIL Image
        confidence_threshold: Confidence threshold for refined prediction
        area_threshold: Area threshold for filtering small regions
        combination_method: Method to combine initial and refined masks
    
    Returns:
        Refined mask as PIL Image
    """
    try:
        logger.info(f"Starting refinement for image: {image_path}")
        print(f"Starting refinement for image: {image_path}")
        
        # Load original image
        original_image = PILImage.open(image_path).convert('RGB')
        original_array = np.array(original_image)
        
        # Convert initial mask to numpy array
        initial_mask_array = np.array(initial_mask.convert('L'))
        
        # Create masked image (mask out initially detected areas)
        masked_image_array = apply_refinement_mask(original_array, initial_mask_array)
        
        # Save masked image temporarily for prediction
        masked_image_pil = PILImage.fromarray(masked_image_array)
        
        # Create temporary file path for masked image
        import tempfile
        import os
        temp_dir = tempfile.gettempdir()
        temp_filename = f"masked_{os.path.basename(image_path)}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        try:
            masked_image_pil.save(temp_path)
            
            # Use lower confidence threshold for refinement to catch more subtle features
            refined_confidence = confidence_threshold * 0.8 if confidence_threshold else None
            
            # Run prediction on masked image
            refined_mask = predict(model, temp_path, area_threshold, confidence_threshold=refined_confidence)
            refined_mask_array = np.array(refined_mask.convert('L'))
            
            # Combine initial and refined masks
            combined_mask_array = combine_masks(initial_mask_array, refined_mask_array, combination_method)
            
            # Apply area threshold to final combined mask
            combined_mask_array = threshold(combined_mask_array, area_threshold)
            
            # Convert back to PIL Image
            final_mask = PILImage.fromarray(combined_mask_array, mode='L')
            
            logger.info(f"Refinement completed successfully for image: {image_path}")
            print(f"Refinement completed successfully for image: {image_path}")
            
            return final_mask
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"Error during refinement for image {image_path}: {str(e)}")
        print(f"Error during refinement for image {image_path}: {str(e)}")
        # Return original mask if refinement fails
        return initial_mask


def predict_with_refinement(model, image_path: str, area_threshold: int = 15,
                          confidence_threshold: float = None, use_refinement: bool = True,
                          combination_method: str = 'additive') -> PILImage.Image:
    """
    Main function to predict with optional refinement.
    
    Args:
        model: The segmentation model
        image_path: Path to the input image
        area_threshold: Area threshold for filtering small regions
        confidence_threshold: Confidence threshold for predictions
        use_refinement: Whether to apply refinement
        combination_method: Method to combine masks if refinement is used
    
    Returns:
        Final prediction mask as PIL Image
    """
    try:
        logger.info(f"Starting prediction with refinement={use_refinement} for image: {image_path}")
        print(f"Starting prediction with refinement={use_refinement} for image: {image_path}")
        
        # Get initial prediction
        initial_mask = predict(model, image_path, area_threshold, confidence_threshold)
        
        if not use_refinement:
            logger.info(f"Returning initial prediction (no refinement) for image: {image_path}")
            return initial_mask
        
        # Apply refinement
        refined_mask = refine_prediction(
            model, image_path, initial_mask, 
            confidence_threshold, area_threshold, combination_method
        )
        
        logger.info(f"Completed prediction with refinement for image: {image_path}")
        return refined_mask
        
    except Exception as e:
        logger.error(f"Error in predict_with_refinement for image {image_path}: {str(e)}")
        print(f"Error in predict_with_refinement for image {image_path}: {str(e)}")
        # Fallback to basic prediction
        return predict(model, image_path, area_threshold, confidence_threshold) 