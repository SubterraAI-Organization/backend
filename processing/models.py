import os

from django.db import models
from django.contrib.auth.models import User
from django_prometheus.models import ExportModelOperationsMixin


class Dataset(ExportModelOperationsMixin('dataset'), models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True, null=True)
    owner = models.ForeignKey('auth.User', related_name='datasets', on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    public = models.BooleanField(default=False)


class Picture(ExportModelOperationsMixin('picture'), models.Model):
    dataset = models.ForeignKey('processing.Dataset', related_name='pictures', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='images/', editable=False)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    @property
    def filename(self) -> str:
        return os.path.basename(self.image.name)

    @property
    def filename_noext(self) -> str:
        return os.path.splitext(self.filename)[0]

    @property
    def owner(self) -> User:
        return self.dataset.owner

    @property
    def public(self) -> bool:
        return self.dataset.public


class Mask(ExportModelOperationsMixin('mask'), models.Model):
    picture = models.OneToOneField('processing.Picture', related_name='mask', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='masks/', editable=False)
    threshold = models.IntegerField(default=0)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    root_count = models.IntegerField(default=0)
    average_root_diameter = models.FloatField(default=0)
    total_root_length = models.FloatField(default=0)
    total_root_area = models.FloatField(default=0)
    total_root_volume = models.FloatField(default=0)

    @property
    def filename(self) -> str:
        return os.path.basename(self.image.name)

    @property
    def filename_noext(self) -> str:
        return os.path.splitext(self.filename)[0]

    @property
    def owner(self) -> User:
        return self.picture.owner

    @property
    def public(self) -> bool:
        return self.picture.public


class Model(ExportModelOperationsMixin('model'), models.Model):
    UNET = 'unet'
    RESNET18 = 'resnet18'
    RESNET34 = 'resnet34'
    RESNET50 = 'resnet50'
    RESNET101 = 'resnet101'
    RESNET152 = 'resnet152'

    choices = {
        (UNET, 'UNet'),
        (RESNET18, 'ResNet18'),
        (RESNET34, 'ResNet34'),
        (RESNET50, 'ResNet50'),
        (RESNET101, 'ResNet101'),
        (RESNET152, 'ResNet152'),
    }

    name = models.CharField(max_length=200)
    model_type = models.CharField(max_length=50, choices=choices, default=UNET)
    model_weights = models.FileField(upload_to='models/', editable=False)
    description = models.TextField(blank=True, null=True)
    owner = models.ForeignKey('auth.User', related_name='models', on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    public = models.BooleanField(default=False)


class ModelType(models.TextChoices):
    """
    Enum for different types of segmentation models.
    """
    UNET = 'UNET', 'U-Net'
    YOLO = 'YOLO', 'YOLO'
    
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
        import os
        import sys
        import traceback
        logger = logging.getLogger(__name__)
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        print(f"Getting model for {self.name} with in_channels={in_channels}, out_channels={out_channels}")
        logger.info(f"Getting model for {self.name} with in_channels={in_channels}, out_channels={out_channels}")
        
        if self == ModelType.UNET:
            try:
                from segmentation.models.unet import UNet
                print(f"Successfully imported UNet from segmentation.models.unet")
                logger.info(f"Successfully imported UNet from segmentation.models.unet")
                
                model = UNet(in_channels=in_channels, out_channels=out_channels, **kwargs)
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
                from torch import nn
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
                            # Fallback to other methods or raise the error
                            raise e
                
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
