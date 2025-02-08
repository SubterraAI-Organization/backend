from typing import Optional
import io
import numpy as np
from pathlib import Path
import torch
from django.db.models.query import QuerySet
from django.core.files import File

from segmentation import predict, calculate_metrics
from processing.models import Picture, Mask

from segmentation.models import ModelType


def predict_image(image: Picture, area_threshold: Optional[int] = 0) -> None:
    model = ModelType.UNET.get_model(in_channels=3, out_channels=1)

    checkpoint = torch.load(
        Path('segmentation/models/saved_models/unet_saved.pth'))

    model.load_state_dict(checkpoint)
    model.eval()

    mask = predict(model, image.image, area_threshold)

    mask_arr = np.array(mask) / 255
    mask_arr = mask_arr.astype(np.uint8)

    metrics = calculate_metrics(mask_arr, 0.2581)

    mask_byte_arr = io.BytesIO()
    mask.save(mask_byte_arr, format='PNG')

    mask = File(mask_byte_arr, name=f'{image.filename_noext}_mask.png')
    mask = Mask(picture=image, image=mask, threshold=area_threshold, **metrics)
    mask.save()

    return mask


def bulk_predict_images(images: QuerySet[Picture],
                        area_threshold: Optional[int] = 0) -> None:
    model = ModelType.UNET.get_model(in_channels=3, out_channels=1)

    checkpoint = torch.load(
        Path('segmentation/models/saved_models/unet_saved.pth'))

    model.load_state_dict(checkpoint)
    model.eval()

    masks = []
    for image in images:
        mask = predict(model, image.image, area_threshold)

        mask_arr = np.array(mask) / 255
        mask_arr = mask_arr.astype(np.uint8)

        metrics = calculate_metrics(mask_arr, 0.2581)

        mask_byte_arr = io.BytesIO()
        mask.save(mask_byte_arr, format='PNG')

        mask = File(mask_byte_arr, name=f'{image.filename_noext}_mask.png')
        masks.append(
            Mask(picture=image,
                 image=mask,
                 threshold=area_threshold,
                 **metrics))

    masks = Mask.objects.bulk_create(masks)

    return masks


def update_mask(original_mask: Mask,
                area_threshold: Optional[int] = 0) -> Mask:
    model = ModelType.UNET.get_model(in_channels=3, out_channels=1)

    checkpoint = torch.load(
        Path('segmentation/models/saved_models/unet_saved.pth'))

    model.load_state_dict(checkpoint)
    model.eval()

    new_mask = predict(model, original_mask.image, area_threshold)

    mask_arr = np.array(new_mask) / 255
    mask_arr = mask_arr.astype(np.uint8)

    metrics = calculate_metrics(mask_arr, 0.2581)

    mask_byte_arr = io.BytesIO()
    new_mask.save(mask_byte_arr, format='PNG')

    mask = File(mask_byte_arr, name=f'{original_mask.filename_noext}_mask.png')
    original_mask.image = mask
    original_mask.threshold = area_threshold
    original_mask.root_count = metrics['root_count']
    original_mask.average_root_diameter = metrics['average_root_diameter']
    original_mask.total_root_length = metrics['total_root_length']
    original_mask.total_root_area = metrics['total_root_area']
    original_mask.total_root_volume = metrics['total_root_volume']
    original_mask.save()

    return mask
