from typing import Optional
import io
import numpy as np
from pathlib import Path
import torch
from django.db.models.query import QuerySet
from django.core.files import File

from segmentation import predict, calculate_metrics
from segmentation.utils.refinement_predict import predict_with_refinement
from segmentation.utils.multi_image_output import generate_prediction_outputs, save_prediction_outputs
from processing.models import Picture, Mask
from processing.models import ModelType


def predict_image(image: Picture, model_type: ModelType, area_threshold: Optional[int] = 0, 
                 confidence_threshold: Optional[float] = None, use_refinement: bool = False,
                 refinement_method: str = 'additive') -> Mask:
    """
    Process a single image with the specified model and create a mask object.
    
    Args:
        image: Picture object to process
        model_type: The type of model to use (UNET, YOLO, etc.)
        area_threshold: Optional threshold for minimum area size
        confidence_threshold: Optional threshold for prediction confidence
        use_refinement: Whether to apply refinement prediction
        refinement_method: Method for combining masks ('additive', 'max', 'weighted')
    
    Returns:
        Created Mask object
    """
    import sys, traceback
    import logging
    import os
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting prediction for image ID: {image.id} - {image.filename} with model {model_type}")
        print(f"Starting prediction for image ID: {image.id} - {image.filename} with model {model_type}")
        
        # Use model-specific default confidence thresholds if none provided
        if confidence_threshold is None:
            confidence_threshold = 0.7 if model_type == ModelType.UNET else 0.3
            
        logger.info(f"Using confidence threshold: {confidence_threshold}")
        print(f"Using confidence threshold: {confidence_threshold}")
            
        try:
            # Check if model files exist with absolute path
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            unet_path = os.path.join(base_dir, 'segmentation', 'models', 'saved_models', 'unet_saved.pth')
            yolo_path = os.path.join(base_dir, 'segmentation', 'models', 'saved_models', 'yolo_saved.pt')
            
            logger.info(f"Base directory: {base_dir}")
            logger.info(f"UNET path exists: {os.path.exists(unet_path)}, path: {unet_path}")
            logger.info(f"YOLO path exists: {os.path.exists(yolo_path)}, path: {yolo_path}")
            
            print(f"Base directory: {base_dir}")
            print(f"UNET path exists: {os.path.exists(unet_path)}, path: {unet_path}")
            print(f"YOLO path exists: {os.path.exists(yolo_path)}, path: {yolo_path}")
            
            model = model_type.get_model(in_channels=3, out_channels=1)
            logger.info(f"Successfully created model instance: {type(model)}")
            print(f"Successfully created model instance: {type(model)}")
            
            match model_type:
                case ModelType.UNET:
                    try:
                        checkpoint_path = Path(unet_path)
                        if not checkpoint_path.exists():
                            logger.error(f"Model checkpoint not found at: {checkpoint_path}")
                            print(f"Model checkpoint not found at: {checkpoint_path}")
                            raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")
                            
                        print(f"Loading checkpoint from {checkpoint_path}")
                        # Try to load with map_location to CPU if available
                        try:
                            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                            print(f"Checkpoint loaded with map_location=cpu")
                        except:
                            checkpoint = torch.load(checkpoint_path)
                            print(f"Checkpoint loaded without map_location")
                        
                        print(f"Checkpoint type: {type(checkpoint)}")
                        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                            print(f"Found state_dict in checkpoint, extracting...")
                            checkpoint = checkpoint['state_dict']
                            
                        # Print some keys to validate checkpoint structure
                        if isinstance(checkpoint, dict):
                            key_list = list(checkpoint.keys())[:5]
                            print(f"First few checkpoint keys: {key_list}")
                            
                            # Check for model. prefix
                            has_model_prefix = any(k.startswith('model.') for k in key_list)
                            if has_model_prefix:
                                print("Removing 'model.' prefix from checkpoint keys")
                                new_checkpoint = {}
                                for k, v in checkpoint.items():
                                    new_key = k.replace('model.', '')
                                    new_checkpoint[new_key] = v
                                checkpoint = new_checkpoint
                        
                        # Try to load the state dict
                        try:
                            model.load_state_dict(checkpoint)
                            print("Successfully loaded checkpoint into model")
                        except Exception as load_err:
                            print(f"Error loading checkpoint into model: {str(load_err)}")
                            # Try a more lenient load
                            try:
                                model.load_state_dict(checkpoint, strict=False)
                                print("Successfully loaded checkpoint with strict=False")
                            except Exception as lenient_err:
                                print(f"Error loading checkpoint with strict=False: {str(lenient_err)}")
                                raise
                        
                        model.eval()
                        logger.info("Successfully loaded UNET model checkpoint")
                        print("Successfully loaded UNET model checkpoint and set to eval mode")
                    except Exception as e:
                        logger.error(f"Failed to load UNET model checkpoint: {str(e)}")
                        print(f"Failed to load UNET model checkpoint: {str(e)}")
                        traceback.print_exc(file=sys.stdout)
                        raise
                        
                case ModelType.YOLO:
                    try:
                        checkpoint_path = Path(yolo_path)
                        if not checkpoint_path.exists():
                            logger.error(f"Model checkpoint not found at: {checkpoint_path}")
                            print(f"Model checkpoint not found at: {checkpoint_path}")
                            raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")
                            
                        print(f"Loading YOLO model from {checkpoint_path}")
                        try:
                            from ultralytics import YOLO
                            model = YOLO(checkpoint_path)
                            print(f"YOLO model loaded successfully using ultralytics")
                        except Exception as yolo_err:
                            print(f"Error loading YOLO model: {str(yolo_err)}")
                            # Try alternate loading
                            model = model(checkpoint_path)
                            print(f"YOLO model loaded using alternate method")
                        
                        logger.info("Successfully loaded YOLO model checkpoint")
                        print("Successfully loaded YOLO model checkpoint")
                    except Exception as e:
                        logger.error(f"Failed to load YOLO model checkpoint: {str(e)}")
                        print(f"Failed to load YOLO model checkpoint: {str(e)}")
                        traceback.print_exc(file=sys.stdout)
                        raise
                        
                case _:
                    logger.error(f"Unsupported model type: {model_type}")
                    print(f"Unsupported model type: {model_type}")
                    raise ValueError(f"Unsupported model type: {model_type}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            print(f"Failed to initialize model: {str(e)}")
            traceback.print_exc(file=sys.stdout)
            raise

        # Check if image file exists and is readable
        if not image.image:
            logger.error(f"Image file is missing for image ID: {image.id}")
            print(f"Image file is missing for image ID: {image.id}")
            raise ValueError(f"Image file is missing for image ID: {image.id}")

        try:
            image_path = image.image.path
            print(f"Processing image at path: {image_path}")
            
            # Use refinement prediction if requested
            if use_refinement:
                print(f"Using refinement prediction with method: {refinement_method}")
                mask = predict_with_refinement(
                    model, image_path, area_threshold, 
                    confidence_threshold=confidence_threshold, 
                    use_refinement=True,
                    combination_method=refinement_method
                )
            else:
                print(f"Using standard prediction")
                mask = predict(model, image_path, area_threshold, confidence_threshold=confidence_threshold)
                
            logger.info(f"Successfully predicted mask for image ID: {image.id}")
            print(f"Successfully predicted mask for image ID: {image.id}")
        except Exception as e:
            logger.error(f"Failed to predict mask for image ID {image.id}: {str(e)}")
            print(f"Failed to predict mask for image ID {image.id}: {str(e)}")
            traceback.print_exc(file=sys.stdout)
            raise

        try:
            # Generate multi-image outputs BEFORE normalizing the mask
            print(f"Generating multi-image outputs for image ID: {image.id}")
            try:
                outputs = generate_prediction_outputs(
                    image_path, mask, image.filename_noext, 
                    model=model, confidence_threshold=confidence_threshold
                )
                files = save_prediction_outputs(outputs, image.filename_noext)
                logger.info(f"Successfully generated multi-image outputs for image ID: {image.id}")
                print(f"Successfully generated multi-image outputs for image ID: {image.id}")
            except Exception as e:
                logger.error(f"Failed to generate multi-image outputs for image ID {image.id}: {str(e)}")
                print(f"Failed to generate multi-image outputs for image ID {image.id}: {str(e)}")
                # Continue with basic mask if multi-image generation fails
                files = {'original': None, 'overlay': None}
            
            # Now normalize the mask for metrics calculation
            mask_arr = np.array(mask) / 255
            mask_arr = mask_arr.astype(np.uint8)

            # Count number of detected objects
            num_objects = np.max(mask_arr)
            print(f"Detected {num_objects} objects in mask")
            logger.info(f"Detected {num_objects} objects in mask")
            
            # Add a check to see if the mask is empty
            if np.sum(mask_arr) == 0:
                print(f"WARNING: Empty mask detected for image ID {image.id}")
                logger.warning(f"Empty mask detected for image ID {image.id}")

            metrics = calculate_metrics(mask_arr, 0.2581)
            logger.info(f"Metrics calculated for image ID {image.id}: {metrics}")
            print(f"Metrics calculated for image ID {image.id}: {metrics}")
            
            mask_byte_arr = io.BytesIO()
            mask.save(mask_byte_arr, format='PNG')
            mask_byte_arr.seek(0)  # Rewind to beginning

            mask_file = File(mask_byte_arr, name=f'{image.filename_noext}_mask.png')
            
            # Create and save the mask with new fields
            mask_obj = Mask(
                picture=image, 
                image=mask_file, 
                threshold=area_threshold,
                use_refinement=use_refinement,
                refinement_method=refinement_method,
                original_image=files.get('original'),
                overlay_image=files.get('overlay'),
                **metrics
            )
            mask_obj.save()
            
            logger.info(f"Successfully created and saved mask for image ID: {image.id}")
            print(f"Successfully created and saved mask for image ID: {image.id}")
            
            return mask_obj
        except Exception as e:
            logger.error(f"Failed to create mask object for image ID {image.id}: {str(e)}")
            print(f"Failed to create mask object for image ID {image.id}: {str(e)}")
            traceback.print_exc(file=sys.stdout)
            raise
            
    except Exception as e:
        logger.error(f"Unhandled exception in predict_image: {str(e)}")
        print(f"Unhandled exception in predict_image: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        raise


def bulk_predict_images(images: QuerySet[Picture], model_type: ModelType, area_threshold: Optional[int] = 0, 
                       confidence_threshold: Optional[float] = None, use_refinement: bool = False,
                       refinement_method: str = 'additive') -> None:
    """
    Process multiple images with the specified model and create mask objects for each.
    
    Args:
        images: QuerySet of Picture objects to process
        model_type: The type of model to use (UNET, YOLO, etc.)
        area_threshold: Optional threshold for minimum area size
        confidence_threshold: Optional threshold for prediction confidence
    
    Returns:
        List of created Mask objects
    """
    import sys, traceback
    import logging
    import os
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting bulk prediction for {len(images)} images with model {model_type}")
        print(f"Starting bulk prediction for {len(images)} images with model {model_type}")
        
        # Use model-specific default confidence thresholds if none provided
        if confidence_threshold is None:
            confidence_threshold = 0.7 if model_type == ModelType.UNET else 0.3
            
        logger.info(f"Using confidence threshold: {confidence_threshold}")
        print(f"Using confidence threshold: {confidence_threshold}")
        
        # Load the model based on model_type
        try:
            # Check if model files exist with absolute path
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            unet_path = os.path.join(base_dir, 'segmentation', 'models', 'saved_models', 'unet_saved.pth')
            yolo_path = os.path.join(base_dir, 'segmentation', 'models', 'saved_models', 'yolo_saved.pt')
            
            logger.info(f"Base directory: {base_dir}")
            logger.info(f"UNET path exists: {os.path.exists(unet_path)}, path: {unet_path}")
            logger.info(f"YOLO path exists: {os.path.exists(yolo_path)}, path: {yolo_path}")
            
            print(f"Base directory: {base_dir}")
            print(f"UNET path exists: {os.path.exists(unet_path)}, path: {unet_path}")
            print(f"YOLO path exists: {os.path.exists(yolo_path)}, path: {yolo_path}")
            
            model = model_type.get_model(in_channels=3, out_channels=1)
            logger.info(f"Successfully created model instance: {type(model)}")
            print(f"Successfully created model instance: {type(model)}")
            
            match model_type:
                case ModelType.UNET:
                    try:
                        checkpoint_path = Path(unet_path)
                        if not checkpoint_path.exists():
                            logger.error(f"Model checkpoint not found at: {checkpoint_path}")
                            print(f"Model checkpoint not found at: {checkpoint_path}")
                            raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")
                        
                        print(f"Loading checkpoint from {checkpoint_path}")
                        # Try to load with map_location to CPU if available
                        try:
                            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                            print(f"Checkpoint loaded with map_location=cpu")
                        except:
                            checkpoint = torch.load(checkpoint_path)
                            print(f"Checkpoint loaded without map_location")
                        
                        print(f"Checkpoint type: {type(checkpoint)}")
                        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                            print(f"Found state_dict in checkpoint, extracting...")
                            checkpoint = checkpoint['state_dict']
                            
                        # Print some keys to validate checkpoint structure
                        if isinstance(checkpoint, dict):
                            key_list = list(checkpoint.keys())[:5]
                            print(f"First few checkpoint keys: {key_list}")
                            
                            # Check for model. prefix
                            has_model_prefix = any(k.startswith('model.') for k in key_list)
                            if has_model_prefix:
                                print("Removing 'model.' prefix from checkpoint keys")
                                new_checkpoint = {}
                                for k, v in checkpoint.items():
                                    new_key = k.replace('model.', '')
                                    new_checkpoint[new_key] = v
                                checkpoint = new_checkpoint
                        
                        # Try to load the state dict
                        try:
                            model.load_state_dict(checkpoint)
                            print("Successfully loaded checkpoint into model")
                        except Exception as load_err:
                            print(f"Error loading checkpoint into model: {str(load_err)}")
                            # Try a more lenient load
                            try:
                                model.load_state_dict(checkpoint, strict=False)
                                print("Successfully loaded checkpoint with strict=False")
                            except Exception as lenient_err:
                                print(f"Error loading checkpoint with strict=False: {str(lenient_err)}")
                                raise
                        
                        model.eval()
                        logger.info("Successfully loaded UNET model checkpoint")
                        print("Successfully loaded UNET model checkpoint and set to eval mode")
                    except Exception as e:
                        logger.error(f"Failed to load UNET model checkpoint: {str(e)}")
                        print(f"Failed to load UNET model checkpoint: {str(e)}")
                        traceback.print_exc(file=sys.stdout)
                        raise
                        
                case ModelType.YOLO:
                    try:
                        checkpoint_path = Path(yolo_path)
                        if not checkpoint_path.exists():
                            logger.error(f"Model checkpoint not found at: {checkpoint_path}")
                            print(f"Model checkpoint not found at: {checkpoint_path}")
                            raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")
                         
                        print(f"Loading YOLO model from {checkpoint_path}")
                        try:
                            from ultralytics import YOLO
                            model = YOLO(checkpoint_path)
                            print(f"YOLO model loaded successfully using ultralytics")
                        except Exception as yolo_err:
                            print(f"Error loading YOLO model: {str(yolo_err)}")
                            # Try alternate loading
                            model = model(checkpoint_path)
                            print(f"YOLO model loaded using alternate method")
                        
                        logger.info("Successfully loaded YOLO model checkpoint")
                        print("Successfully loaded YOLO model checkpoint")
                    except Exception as e:
                        logger.error(f"Failed to load YOLO model checkpoint: {str(e)}")
                        print(f"Failed to load YOLO model checkpoint: {str(e)}")
                        traceback.print_exc(file=sys.stdout)
                        raise
                        
                case _:
                    logger.error(f"Unsupported model type: {model_type}")
                    print(f"Unsupported model type: {model_type}")
                    raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            print(f"Failed to initialize model: {str(e)}")
            traceback.print_exc(file=sys.stdout)
            raise

        masks = []
        for i, image in enumerate(images):
            try:
                logger.info(f"Processing image {i+1}/{len(images)}: {image.id} - {image.filename}")
                print(f"Processing image {i+1}/{len(images)}: {image.id} - {image.filename}")
                
                # Check if image file exists and is readable
                if not image.image:
                    logger.error(f"Image file is missing for image ID: {image.id}")
                    print(f"Image file is missing for image ID: {image.id}")
                    continue
                    
                image_path = image.image.path
                print(f"Processing image at path: {image_path}")
                
                # Pass confidence_threshold to predict function
                try:
                    mask = predict(model, image_path, area_threshold, confidence_threshold=confidence_threshold)
                    logger.info(f"Successfully predicted mask for image ID: {image.id}")
                    print(f"Successfully predicted mask for image ID: {image.id}")
                except Exception as e:
                    logger.error(f"Failed to predict mask for image ID {image.id}: {str(e)}")
                    print(f"Failed to predict mask for image ID {image.id}: {str(e)}")
                    traceback.print_exc(file=sys.stdout)
                    continue

                try:
                    # Generate multi-image outputs BEFORE normalizing the mask
                    print(f"Generating multi-image outputs for image ID: {image.id}")
                    try:
                        outputs = generate_prediction_outputs(
                            image_path, mask, image.filename_noext, 
                            model=model, confidence_threshold=confidence_threshold
                        )
                        files = save_prediction_outputs(outputs, image.filename_noext)
                        logger.info(f"Successfully generated multi-image outputs for image ID: {image.id}")
                        print(f"Successfully generated multi-image outputs for image ID: {image.id}")
                    except Exception as e:
                        logger.error(f"Failed to generate multi-image outputs for image ID {image.id}: {str(e)}")
                        print(f"Failed to generate multi-image outputs for image ID {image.id}: {str(e)}")
                        # Continue with basic mask if multi-image generation fails
                        files = {'original': None, 'overlay': None}
                    
                    # Now normalize the mask for metrics calculation
                    mask_arr = np.array(mask) / 255
                    mask_arr = mask_arr.astype(np.uint8)
                    
                    # Count number of detected objects
                    num_objects = np.max(mask_arr)
                    print(f"Detected {num_objects} objects in mask")
                    logger.info(f"Detected {num_objects} objects in mask")
                    
                    # Add a check to see if the mask is empty
                    if np.sum(mask_arr) == 0:
                        print(f"WARNING: Empty mask detected for image ID {image.id}")
                        logger.warning(f"Empty mask detected for image ID {image.id}")
                    
                    metrics = calculate_metrics(mask_arr, 0.2581)
                    logger.info(f"Metrics calculated for image ID {image.id}: {metrics}")
                    print(f"Metrics calculated for image ID {image.id}: {metrics}")
                    
                    mask_byte_arr = io.BytesIO()
                    mask.save(mask_byte_arr, format='PNG')
                    mask_byte_arr.seek(0)  # Rewind to beginning
                    
                    mask_file = File(mask_byte_arr, name=f'{image.filename_noext}_mask.png')
                    
                    # Create and save the mask with new fields
                    mask_obj = Mask(
                        picture=image, 
                        image=mask_file, 
                        threshold=area_threshold,
                        use_refinement=use_refinement,
                        refinement_method=refinement_method,
                        original_image=files.get('original'),
                        overlay_image=files.get('overlay'),
                        **metrics
                    )
                    mask_obj.save()
                    
                    logger.info(f"Successfully created and saved mask for image ID: {image.id}")
                    print(f"Successfully created and saved mask for image ID: {image.id}")
                    
                    masks.append(mask_obj)
                except Exception as e:
                    logger.error(f"Failed to process mask for image ID {image.id}: {str(e)}")
                    print(f"Failed to process mask for image ID {image.id}: {str(e)}")
                    traceback.print_exc(file=sys.stdout)
                    continue
            except Exception as e:
                logger.error(f"Failed to process image ID {image.id}: {str(e)}")
                print(f"Failed to process image ID {image.id}: {str(e)}")
                traceback.print_exc(file=sys.stdout)
                continue

        try:
            if masks:
                created_masks = Mask.objects.bulk_create(masks)
                logger.info(f"Successfully created {len(created_masks)} mask objects")
                print(f"Successfully created {len(created_masks)} mask objects")
                return created_masks
            else:
                logger.warning("No masks were created during bulk prediction")
                print("No masks were created during bulk prediction")
                return []
        except Exception as e:
            logger.error(f"Failed to bulk create masks: {str(e)}")
            print(f"Failed to bulk create masks: {str(e)}")
            traceback.print_exc(file=sys.stdout)
            raise
            
    except Exception as e:
        logger.error(f"Unhandled exception in bulk_predict_images: {str(e)}")
        print(f"Unhandled exception in bulk_predict_images: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        raise


def update_mask(original_mask: Mask, model_type: ModelType, area_threshold: Optional[int] = 0, confidence_threshold: Optional[float] = None) -> Mask:
    """
    Update an existing mask using a different model or threshold.
    
    Args:
        original_mask: The existing mask to update
        model_type: The type of model to use (UNET, YOLO, etc.)
        area_threshold: Optional threshold for minimum area size
        confidence_threshold: Optional threshold for prediction confidence
    
    Returns:
        Updated Mask object
    """
    import sys, traceback
    import logging
    import os
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Updating mask ID: {original_mask.id} with model {model_type}")
        print(f"Updating mask ID: {original_mask.id} with model {model_type}")
        
        # Use model-specific default confidence thresholds if none provided
        if confidence_threshold is None:
            confidence_threshold = 0.7 if model_type == ModelType.UNET else 0.3
            
        logger.info(f"Using confidence threshold: {confidence_threshold}")
        print(f"Using confidence threshold: {confidence_threshold}")
        
        try:
            # Check if model files exist with absolute path
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            unet_path = os.path.join(base_dir, 'segmentation', 'models', 'saved_models', 'unet_saved.pth')
            yolo_path = os.path.join(base_dir, 'segmentation', 'models', 'saved_models', 'yolo_saved.pt')
            
            logger.info(f"Base directory: {base_dir}")
            logger.info(f"UNET path exists: {os.path.exists(unet_path)}, path: {unet_path}")
            logger.info(f"YOLO path exists: {os.path.exists(yolo_path)}, path: {yolo_path}")
            
            print(f"Base directory: {base_dir}")
            print(f"UNET path exists: {os.path.exists(unet_path)}, path: {unet_path}")
            print(f"YOLO path exists: {os.path.exists(yolo_path)}, path: {yolo_path}")
            
            model = model_type.get_model(in_channels=3, out_channels=1)
            logger.info(f"Successfully created model instance: {type(model)}")
            print(f"Successfully created model instance: {type(model)}")
            
            match model_type:
                case ModelType.UNET:
                    try:
                        checkpoint_path = Path(unet_path)
                        if not checkpoint_path.exists():
                            logger.error(f"Model checkpoint not found at: {checkpoint_path}")
                            print(f"Model checkpoint not found at: {checkpoint_path}")
                            raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")
                            
                        print(f"Loading checkpoint from {checkpoint_path}")
                        # Try to load with map_location to CPU if available
                        try:
                            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                            print(f"Checkpoint loaded with map_location=cpu")
                        except:
                            checkpoint = torch.load(checkpoint_path)
                            print(f"Checkpoint loaded without map_location")
                        
                        print(f"Checkpoint type: {type(checkpoint)}")
                        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                            print(f"Found state_dict in checkpoint, extracting...")
                            checkpoint = checkpoint['state_dict']
                            
                        # Print some keys to validate checkpoint structure
                        if isinstance(checkpoint, dict):
                            key_list = list(checkpoint.keys())[:5]
                            print(f"First few checkpoint keys: {key_list}")
                            
                            # Check for model. prefix
                            has_model_prefix = any(k.startswith('model.') for k in key_list)
                            if has_model_prefix:
                                print("Removing 'model.' prefix from checkpoint keys")
                                new_checkpoint = {}
                                for k, v in checkpoint.items():
                                    new_key = k.replace('model.', '')
                                    new_checkpoint[new_key] = v
                                checkpoint = new_checkpoint
                        
                        # Try to load the state dict
                        try:
                            model.load_state_dict(checkpoint)
                            print("Successfully loaded checkpoint into model")
                        except Exception as load_err:
                            print(f"Error loading checkpoint into model: {str(load_err)}")
                            # Try a more lenient load
                            try:
                                model.load_state_dict(checkpoint, strict=False)
                                print("Successfully loaded checkpoint with strict=False")
                            except Exception as lenient_err:
                                print(f"Error loading checkpoint with strict=False: {str(lenient_err)}")
                                raise
                        
                        model.eval()
                        logger.info("Successfully loaded UNET model checkpoint")
                        print("Successfully loaded UNET model checkpoint and set to eval mode")
                    except Exception as e:
                        logger.error(f"Failed to load UNET model checkpoint: {str(e)}")
                        print(f"Failed to load UNET model checkpoint: {str(e)}")
                        traceback.print_exc(file=sys.stdout)
                        raise
                        
                case ModelType.YOLO:
                    try:
                        checkpoint_path = Path(yolo_path)
                        if not checkpoint_path.exists():
                            logger.error(f"Model checkpoint not found at: {checkpoint_path}")
                            print(f"Model checkpoint not found at: {checkpoint_path}")
                            raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")
                            
                        print(f"Loading YOLO model from {checkpoint_path}")
                        try:
                            from ultralytics import YOLO
                            model = YOLO(checkpoint_path)
                            print(f"YOLO model loaded successfully using ultralytics")
                        except Exception as yolo_err:
                            print(f"Error loading YOLO model: {str(yolo_err)}")
                            # Try alternate loading
                            model = model(checkpoint_path)
                            print(f"YOLO model loaded using alternate method")
                        
                        logger.info("Successfully loaded YOLO model checkpoint")
                        print("Successfully loaded YOLO model checkpoint")
                    except Exception as e:
                        logger.error(f"Failed to load YOLO model checkpoint: {str(e)}")
                        print(f"Failed to load YOLO model checkpoint: {str(e)}")
                        traceback.print_exc(file=sys.stdout)
                        raise
                        
                case _:
                    logger.error(f"Unsupported model type: {model_type}")
                    print(f"Unsupported model type: {model_type}")
                    raise ValueError(f"Unsupported model type: {model_type}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            print(f"Failed to initialize model: {str(e)}")
            traceback.print_exc(file=sys.stdout)
            raise
        
        try:
            # Check if the picture exists
            picture = original_mask.picture
            if not picture or not picture.image:
                logger.error(f"Picture is missing for mask ID: {original_mask.id}")
                print(f"Picture is missing for mask ID: {original_mask.id}")
                raise ValueError(f"Picture is missing for mask ID: {original_mask.id}")
                
            image_path = picture.image.path
            print(f"Processing image at path: {image_path}")
            
            # Pass confidence_threshold to predict function
            new_mask = predict(model, image_path, area_threshold, confidence_threshold=confidence_threshold)
            logger.info(f"Successfully predicted new mask for picture ID: {picture.id}")
            print(f"Successfully predicted new mask for picture ID: {picture.id}")
            
            mask_arr = np.array(new_mask) / 255
            mask_arr = mask_arr.astype(np.uint8)
            
            # Count number of detected objects
            num_objects = np.max(mask_arr)
            print(f"Detected {num_objects} objects in new mask")
            logger.info(f"Detected {num_objects} objects in new mask")
            
            # Add a check to see if the mask is empty
            if np.sum(mask_arr) == 0:
                print(f"WARNING: Empty mask detected for picture ID {picture.id}")
                logger.warning(f"Empty mask detected for picture ID {picture.id}")
            
            metrics = calculate_metrics(mask_arr, 0.2581)
            logger.info(f"Metrics calculated for picture ID {picture.id}: {metrics}")
            print(f"Metrics calculated for picture ID {picture.id}: {metrics}")
            
            mask_byte_arr = io.BytesIO()
            new_mask.save(mask_byte_arr, format='PNG')
            mask_byte_arr.seek(0)  # Rewind to beginning
            
            mask_file = File(mask_byte_arr, name=f'{picture.filename_noext}_mask.png')
            
            # Update the mask object
            original_mask.image = mask_file
            original_mask.threshold = area_threshold
            original_mask.root_count = metrics['root_count']
            original_mask.average_root_diameter = metrics['average_root_diameter']
            original_mask.total_root_length = metrics['total_root_length']
            original_mask.total_root_area = metrics['total_root_area']
            original_mask.total_root_volume = metrics['total_root_volume']
            original_mask.save()
            
            logger.info(f"Successfully updated mask ID: {original_mask.id}")
            print(f"Successfully updated mask ID: {original_mask.id}")
            
            return original_mask
        except Exception as e:
            logger.error(f"Failed to update mask ID {original_mask.id}: {str(e)}")
            print(f"Failed to update mask ID {original_mask.id}: {str(e)}")
            traceback.print_exc(file=sys.stdout)
            raise
            
    except Exception as e:
        logger.error(f"Unhandled exception in update_mask: {str(e)}")
        print(f"Unhandled exception in update_mask: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        raise
