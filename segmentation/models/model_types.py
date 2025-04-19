from enum import Enum
import os
import sys
from pathlib import Path
import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)

class ModelType(Enum):
    UNET = 'unet'
    YOLO = 'yolo'
    
    def get_model(self, in_channels=3, out_channels=1, **kwargs):
        """
        Returns the model instance for this model type.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            **kwargs: Additional keyword arguments for model creation
            
        Returns:
            Model instance
        """
        import logging
        import traceback
        logger = logging.getLogger(__name__)
        
        print(f"Getting model for {self.name} with in_channels={in_channels}, out_channels={out_channels}")
        logger.info(f"Getting model for {self.name} with in_channels={in_channels}, out_channels={out_channels}")
        
        if self == ModelType.UNET:
            try:
                # Import the UNet model
                from .unet import UNet
                
                print(f"Successfully imported UNet from .unet")
                logger.info(f"Successfully imported UNet")
                
                model = UNet(in_channels=in_channels, out_channels=out_channels)
                print(f"UNet model created successfully")
                logger.info(f"UNet model created successfully")
                return model
            except Exception as e:
                error_msg = f"Error creating UNet model: {str(e)}"
                print(error_msg)
                logger.error(error_msg)
                traceback.print_exc(file=sys.stdout)
                raise
        elif self == ModelType.YOLO:
            try:
                print(f"Attempting to create YOLO model placeholder")
                logger.info(f"Attempting to create YOLO model placeholder")
                
                # Create a placeholder model that will be replaced when loading the weights
                class YOLOPlaceholder(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.placeholder = nn.Conv2d(in_channels, out_channels, 1)
                        
                    def __call__(self, checkpoint_path):
                        """This will be called when loading the model with the checkpoint"""
                        try:
                            from ultralytics import YOLO
                            print(f"Using ultralytics.YOLO to load model from {checkpoint_path}")
                            return YOLO(checkpoint_path)
                        except Exception as e:
                            print(f"Failed to load YOLO using ultralytics: {str(e)}")
                            print(f"Attempting alternate loading method")
                            # Fallback
                            raise e
                            
                    def forward(self, x):
                        # This is a placeholder
                        return self.placeholder(x)
                
                model = YOLOPlaceholder()
                print(f"YOLO placeholder model created successfully")
                logger.info(f"YOLO placeholder model created successfully")
                return model
            except Exception as e:
                error_msg = f"Error creating YOLO model: {str(e)}"
                print(error_msg)
                logger.error(error_msg)
                traceback.print_exc(file=sys.stdout)
                raise
        else:
            error_msg = f"Unknown model type: {self.name}"
            print(error_msg)
            logger.error(error_msg)
            raise ValueError(error_msg)
