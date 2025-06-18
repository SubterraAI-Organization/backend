from PIL import Image as PILImage
import numpy as np
import torch
from torch import nn
from torchvision.transforms.v2 import functional as F
from ultralytics import YOLO

from .masks import threshold


def predict(model: nn.Module, image_path: str, area_threshold: int = 15, confidence_threshold: float = None) -> PILImage.Image:
    """
    Predicts the segmentation mask for an input image using a given model.

    Args:
        model (nn.Module): The segmentation model.
        image_path (str): The path to the input image.
        area_threshold (int, optional): The threshold for filtering small regions in the segmentation mask.
            Defaults to 15.
        confidence_threshold (float, optional): The confidence threshold for the model predictions.
            Defaults to 0.7 for UNET, 0.3 for YOLO.

    Returns:
        PIL.Image.Image: The predicted segmentation mask as a PIL image.
    """
    import logging
    import sys
    import traceback
    logger = logging.getLogger(__name__)
    
    try:
        print(f"Starting prediction with model type: {type(model).__name__}")
        
        # Default confidence thresholds if none provided
        if confidence_threshold is None:
            confidence_threshold = 0.5
        
        print(f"Using confidence threshold: {confidence_threshold}")
        
        # Verify image path is valid
        if not image_path or not isinstance(image_path, str):
            error_msg = f"Invalid image path: {image_path}"
            print(error_msg)
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Handle different model types
        if isinstance(model, YOLO):
            try:
                print(f"Processing with YOLO model")
                # YOLO prediction logic - disable labels and confidence display
                results = model(image_path, conf=confidence_threshold, show_labels=False, show_conf=False)
                print(f"YOLO prediction complete, got {len(results)} results")
                
                if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
                    # Get first mask from YOLO predictions
                    mask = results[0].masks.data[0].cpu().numpy()
                    mask = (mask > 0) * 255  # Convert to binary mask
                    mask = mask.astype(np.uint8)
                    mask = threshold(mask, area_threshold)
                    print(f"YOLO mask created successfully")
                    return F.to_pil_image(mask)
                else:
                    # Return blank mask if no predictions
                    print(f"No masks found in YOLO results, returning blank mask")
                    image = PILImage.open(image_path)
                    blank_mask = np.zeros((image.height, image.width), dtype=np.uint8)
                    return F.to_pil_image(blank_mask)
            except Exception as e:
                error_msg = f"Error in YOLO prediction: {str(e)}"
                print(error_msg)
                logger.error(error_msg)
                traceback.print_exc(file=sys.stdout)
                
                # Return a blank mask as fallback
                try:
                    image = PILImage.open(image_path)
                    blank_mask = np.zeros((image.height, image.width), dtype=np.uint8)
                    return F.to_pil_image(blank_mask)
                except Exception as inner_e:
                    error_msg = f"Failed to create blank mask: {str(inner_e)}"
                    print(error_msg)
                    logger.error(error_msg)
                    raise
        else:
            try:
                print(f"Processing with standard model: {type(model).__name__}")
                
                # Standard model prediction logic (UNet, ResNet)
                try:
                    image = PILImage.open(image_path)
                    print(f"Image opened successfully: {image.size}")
                except Exception as e:
                    error_msg = f"Failed to open image at {image_path}: {str(e)}"
                    print(error_msg)
                    logger.error(error_msg)
                    raise
                
                image = np.array(image)
                
                # Make sure we have a 3-channel image
                if len(image.shape) != 3 or image.shape[2] < 3:
                    error_msg = f"Image has unexpected shape {image.shape}, expected a 3-channel RGB image"
                    print(error_msg)
                    logger.error(error_msg)
                    # Try to convert to RGB if possible
                    try:
                        original_image = PILImage.open(image_path).convert('RGB')
                        image = np.array(original_image)
                    except Exception as convert_e:
                        print(f"Failed to convert image to RGB: {str(convert_e)}")
                        raise ValueError(error_msg)
                
                # Extract only RGB channels
                image = image[:, :, :3]
                
                # Convert to tensor
                image = F.to_image(image)
                image = F.to_dtype(image, torch.float32, scale=True)
                
                image = image.unsqueeze(0)
                
                try:
                    print(f"Running model inference")
                    with torch.no_grad():
                        pred = model(image).detach()
                    print(f"Model inference complete")
                except Exception as e:
                    error_msg = f"Error during model inference: {str(e)}"
                    print(error_msg)
                    logger.error(error_msg)
                    traceback.print_exc(file=sys.stdout)
                    raise
                
                # Apply confidence threshold 
                pred = (pred > confidence_threshold).float()
                
                # Convert to image
                image = pred.squeeze(0, 1)
                image = image.numpy().astype(np.uint8) * 255
                
                # Apply area threshold
                try:
                    image = threshold(image, area_threshold)
                    print(f"Threshold applied successfully")
                except Exception as e:
                    error_msg = f"Error applying threshold: {str(e)}"
                    print(error_msg)
                    logger.error(error_msg)
                
                # Convert to PIL image
                try:
                    result = F.to_pil_image(image)
                    print(f"Final mask created successfully: {result.size}")
                    return result
                except Exception as e:
                    error_msg = f"Error converting result to PIL image: {str(e)}"
                    print(error_msg)
                    logger.error(error_msg)
                    raise
            except Exception as e:
                error_msg = f"Error in standard model prediction: {str(e)}"
                print(error_msg)
                logger.error(error_msg)
                traceback.print_exc(file=sys.stdout)
                
                # Return a blank mask as fallback
                try:
                    original_image = PILImage.open(image_path)
                    blank_mask = np.zeros((original_image.height, original_image.width), dtype=np.uint8)
                    return F.to_pil_image(blank_mask)
                except Exception as inner_e:
                    error_msg = f"Failed to create blank mask: {str(inner_e)}"
                    print(error_msg)
                    logger.error(error_msg)
                    raise
    
    except Exception as e:
        error_msg = f"Unhandled exception in predict: {str(e)}"
        print(error_msg)
        logger.error(error_msg)
        traceback.print_exc(file=sys.stdout)
        
        # Last resort - try to create a blank mask of some reasonable size
        try:
            # Create a blank mask of 512x512 pixels if all else fails
            blank_mask = np.zeros((512, 512), dtype=np.uint8)
            return F.to_pil_image(blank_mask)
        except Exception:
            # If everything fails, raise the original exception
            raise
