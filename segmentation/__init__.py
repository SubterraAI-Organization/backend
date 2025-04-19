from .utils import file_management, masks
from .utils.root_analysis import calculate_metrics
from .utils.predict import predict
import os
import sys
import logging
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional, Any

logger = logging.getLogger(__name__)

def predict(model: Any, image_path: Union[str, Path], area_threshold: int = 0, confidence_threshold: Optional[float] = None) -> Image.Image:
    """
    Predict a mask for an image using a model.
    
    Args:
        model: The model to use for prediction
        image_path: Path to the image or Path object
        area_threshold: Minimum area size to include in the mask
        confidence_threshold: Confidence threshold for YOLO models
        
    Returns:
        PIL Image containing the mask
    """
    try:
        print(f"Starting prediction for image: {image_path}")
        logger.info(f"Starting prediction for image: {image_path}")
        print(f"Model type: {type(model).__name__}")
        logger.info(f"Model type: {type(model).__name__}")
        
        # Check if the image exists
        image_path = Path(image_path)
        if not image_path.exists():
            error_msg = f"Image not found at {image_path}"
            print(error_msg)
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        print(f"Image found at {image_path}, loading...")
        logger.info(f"Image found at {image_path}, loading...")
        
        # Determine model type for specialized handling
        model_type = 'yolo' if 'YOLO' in type(model).__name__ else 'unet'
        print(f"Detected model type: {model_type}")
        logger.info(f"Detected model type: {model_type}")
        
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        print(f"Image loaded, size: {image.size}")
        logger.info(f"Image loaded, size: {image.size}")
        
        # Process with different model types
        if model_type == 'yolo':
            try:
                print(f"Processing with YOLO model, confidence_threshold: {confidence_threshold}")
                logger.info(f"Processing with YOLO model, confidence_threshold: {confidence_threshold}")
                
                # YOLO models use their own prediction method
                results = model.predict(
                    str(image_path), 
                    conf=confidence_threshold if confidence_threshold is not None else 0.3
                )
                
                mask = results[0].plot(masks=True)[:, :, ::-1]  # BGR to RGB
                mask = Image.fromarray(mask)
                print(f"YOLO prediction completed, mask size: {mask.size}")
                logger.info(f"YOLO prediction completed, mask size: {mask.size}")
                return mask
                
            except Exception as e:
                error_msg = f"Error in YOLO prediction: {str(e)}"
                print(error_msg)
                logger.error(error_msg)
                import traceback
                traceback.print_exc(file=sys.stdout)
                raise
        else:
            try:
                print(f"Processing with UNet model")
                logger.info(f"Processing with UNet model")
                
                # Convert image to tensor and normalize
                from torchvision import transforms
                
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                input_tensor = transform(image).unsqueeze(0)
                print(f"Image converted to tensor, shape: {input_tensor.shape}")
                logger.info(f"Image converted to tensor, shape: {input_tensor.shape}")
                
                # Run inference
                with torch.no_grad():
                    print("Running model inference...")
                    logger.info("Running model inference...")
                    output = model(input_tensor)
                
                print(f"Model output shape: {output.shape}")
                logger.info(f"Model output shape: {output.shape}")
                
                # Convert output to mask
                mask_np = output.squeeze().cpu().numpy()
                
                # Apply confidence threshold if provided
                if confidence_threshold is not None:
                    print(f"Applying confidence threshold: {confidence_threshold}")
                    logger.info(f"Applying confidence threshold: {confidence_threshold}")
                    mask_np = (mask_np > confidence_threshold).astype(np.uint8) * 255
                else:
                    print("Using default threshold (0.5) for mask creation")
                    logger.info("Using default threshold (0.5) for mask creation")
                    mask_np = (mask_np > 0.5).astype(np.uint8) * 255
                
                # Apply area threshold if needed
                if area_threshold > 0:
                    print(f"Applying area threshold: {area_threshold}")
                    logger.info(f"Applying area threshold: {area_threshold}")
                    from skimage import measure
                    
                    labeled = measure.label(mask_np)
                    props = measure.regionprops(labeled)
                    
                    for prop in props:
                        if prop.area < area_threshold:
                            mask_np[labeled == prop.label] = 0
                
                mask = Image.fromarray(mask_np)
                
                # Resize to match original image
                mask = mask.resize(image.size, Image.NEAREST)
                print(f"UNet prediction completed, mask size: {mask.size}")
                logger.info(f"UNet prediction completed, mask size: {mask.size}")
                
                return mask
            except Exception as e:
                error_msg = f"Error in UNet prediction: {str(e)}"
                print(error_msg)
                logger.error(error_msg)
                import traceback
                traceback.print_exc(file=sys.stdout)
                raise
                
    except Exception as e:
        error_msg = f"Unhandled exception in predict function: {str(e)}"
        print(error_msg)
        logger.error(error_msg)
        import traceback
        traceback.print_exc(file=sys.stdout)
        raise

def calculate_metrics(mask: np.ndarray, px_to_mm_ratio: float) -> dict:
    """
    Calculate various metrics from a prediction mask.
    
    Args:
        mask: Numpy array containing the mask
        px_to_mm_ratio: Ratio to convert pixels to millimeters
        
    Returns:
        Dictionary containing the calculated metrics
    """
    import logging
    logger = logging.getLogger(__name__)
    
    print(f"Calculating metrics with px_to_mm_ratio: {px_to_mm_ratio}")
    logger.info(f"Calculating metrics with px_to_mm_ratio: {px_to_mm_ratio}")
    
    try:
        from skimage import measure
        
        # Basic metrics
        labeled = measure.label(mask)
        props = measure.regionprops(labeled)
        
        # Root count is the number of connected regions
        root_count = len(props)
        print(f"Found {root_count} roots in mask")
        logger.info(f"Found {root_count} roots in mask")
        
        # Calculate metrics from region properties
        if props:
            # Total metrics
            total_area_px = sum(prop.area for prop in props)
            total_area_mm = total_area_px * (px_to_mm_ratio ** 2)
            
            # Average diameter - using equivalent diameter from area
            avg_diameter_px = sum(prop.equivalent_diameter for prop in props) / root_count
            avg_diameter_mm = avg_diameter_px * px_to_mm_ratio
            
            # Approximating length as the major axis of the ellipse that fits the region
            total_length_px = sum(prop.major_axis_length for prop in props)
            total_length_mm = total_length_px * px_to_mm_ratio
            
            # Volume estimated as cylinder (area * length)
            total_volume_mm3 = total_area_mm * total_length_mm
            
            metrics = {
                'root_count': root_count,
                'average_root_diameter': round(avg_diameter_mm, 4),
                'total_root_length': round(total_length_mm, 4),
                'total_root_area': round(total_area_mm, 4),
                'total_root_volume': round(total_volume_mm3, 4),
            }
        else:
            # No roots found
            print("No roots found in mask")
            logger.info("No roots found in mask")
            metrics = {
                'root_count': 0,
                'average_root_diameter': 0.0,
                'total_root_length': 0.0,
                'total_root_area': 0.0,
                'total_root_volume': 0.0,
            }
            
        logger.info(f"Calculated metrics: {metrics}")
        print(f"Calculated metrics: {metrics}")
        return metrics
        
    except Exception as e:
        error_msg = f"Error calculating metrics: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        import traceback
        traceback.print_exc(file=sys.stdout)
        
        # Return zeros for all metrics on error
        return {
            'root_count': 0,
            'average_root_diameter': 0.0,
            'total_root_length': 0.0,
            'total_root_area': 0.0,
            'total_root_volume': 0.0,
        }
