from enum import Enum
from torch import nn

from ultralytics import YOLO
from .unet import UNet
from .resnet import ResNet


class ModelType(Enum):
    UNET = 'unet'
    RESNET18 = 'resnet18'
    RESNET34 = 'resnet34'
    RESNET50 = 'resnet50'
    RESNET101 = 'resnet101'
    RESNET152 = 'resnet152'
    DETECTRON = 'detectron'
    YOLO = 'yolo'

    def __str__(self) -> str:
        return self.value

    def get_model(self, in_channels: int, out_channels: int, dropout: float = 0.2) -> nn.Module:
        match self:
            case ModelType.UNET:
                return UNet(in_channels, out_channels)
            case model if model.name.startswith('RESNET'):
                return ResNet(
                    in_channels,
                    out_channels,
                    num_layers=int(self.name[6:]),
                    dropout=dropout,
                )
            case ModelType.YOLO:
                return YOLO
            case ModelType.DETECTRON:
                pass
