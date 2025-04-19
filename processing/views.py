import io
import os
import zipfile
import json
import numpy as np
from PIL import Image as PILImage

from django.http import HttpResponse, HttpRequest
from django.db.models import Q
from django.db.models.query import QuerySet
from django.core.files import File
from django_q.tasks import async_task, result
from rest_framework import viewsets, permissions, status
from rest_framework.serializers import Serializer
from rest_framework.decorators import action
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, extend_schema_view, OpenApiParameter, OpenApiTypes

from processing.models import Dataset, Picture, Mask, Model, ModelType
from processing.serializers import DatasetSerializer, PictureSerializer, MaskSerializer, LabelMeSerializer, ModelSerializer
from processing.permissions import IsOwnerOrReadOnly
from segmentation import masks, calculate_metrics


@extend_schema(tags=['datasets'])
@extend_schema_view(
    list=extend_schema(summary='List all datasets'),
    create=extend_schema(summary='Create a new dataset'),
    retrieve=extend_schema(summary='Retrieve a dataset'),
    partial_update=extend_schema(summary='Update a dataset'),
    destroy=extend_schema(summary='Delete a dataset'),
)
class DatasetViewSet(viewsets.ModelViewSet):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    permission_classes = []  # Removed auth permissions
    http_method_names = ['get', 'post', 'patch', 'delete']

    def create(self, request: HttpRequest) -> Response:
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        # Get or create a default user for all datasets in development
        from django.contrib.auth import get_user_model
        User = get_user_model()
        
        user, _ = User.objects.get_or_create(
            username='system',
            defaults={'email': 'system@example.com', 'password': 'unsecured'}
        )
        
        # Create the dataset with the system user as owner
        dataset = serializer.save(owner=user)
        
        # Set the dataset as public by default for easier development
        dataset.public = True
        dataset.save()
        
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def get_queryset(self) -> QuerySet[Dataset]:
        # Simplified to return all datasets
        return Dataset.objects.all()

class PictureViewSet(viewsets.ModelViewSet):
    queryset = Picture.objects.all()
    serializer_class = PictureSerializer
    permission_classes = []  # Removed auth permissions
    http_method_names = ['get', 'post', 'delete']

    def create(self, request: HttpRequest, dataset_pk: int = None) -> Response:
        try:
            # Removed owner filter
            dataset = Dataset.objects.get(pk=dataset_pk)
        except Dataset.DoesNotExist:
            return Response({'detail': 'Dataset not found.'}, status=status.HTTP_404_NOT_FOUND)

        # Verify that an image was provided in the request
        if 'image' not in request.data or not request.data['image']:
            return Response({'detail': 'No image file provided.'}, status=status.HTTP_400_BAD_REQUEST)

        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        # Explicitly save with the dataset attached
        picture = serializer.save(dataset=dataset)
        
        # Log success for debugging
        print(f"Successfully created picture with ID: {picture.id} for dataset ID: {dataset_pk}")
        
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @extend_schema(summary='Delete multiple images',
                   parameters=[OpenApiParameter(name='ids', type=str, location='query', required=True)])
    def bulk_destroy(self, request: HttpRequest, dataset_pk: int = None, format: str = None) -> Response:
        image_ids = request.query_params.get('ids')
        
        if not image_ids:
            return Response({'detail': 'No image ids provided.'}, status=status.HTTP_400_BAD_REQUEST)
            
        # Filter out empty strings before converting to integers
        ids = [int(id) for id in image_ids.split(',') if id.strip()]
        
        if not ids:
            return Response({'detail': 'No valid image ids provided.'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Handle anonymous or null user    
        if not hasattr(request, 'user') or request.user is None:
            return Response({'detail': 'Authentication required.'}, status=status.HTTP_401_UNAUTHORIZED)
        
        # Try to get the dataset first
        try:
            dataset = Dataset.objects.get(pk=dataset_pk)
            
            # Check if user has permission to modify this dataset
            if dataset.owner != request.user:
                return Response({'detail': 'You do not have permission to modify this dataset.'}, 
                               status=status.HTTP_403_FORBIDDEN)
        except Dataset.DoesNotExist:
            return Response({'detail': 'Dataset not found.'}, status=status.HTTP_404_NOT_FOUND)
            
        # Now get the images
        images = self.queryset.filter(id__in=ids, dataset=dataset_pk)

        if len(ids) != len(images):
            return Response({'detail': 'Some images do not exist.'}, status=status.HTTP_404_NOT_FOUND)

        images.delete()

        return Response(status=status.HTTP_204_NO_CONTENT)

    @extend_schema(tags=['masks'],
                   summary='Predict masks for multiple images',
                   parameters=[
                       OpenApiParameter(name='ids', type=str, location='query', required=True),
                       OpenApiParameter(name='confidence_threshold', type=float, location='query', required=False),
                   ],
                   responses={201: None})
    @action(detail=False, methods=['post'], url_path='predict', serializer_class=MaskSerializer)
    def bulk_predict(self, request: HttpRequest, dataset_pk: int = None) -> Response:
        # Check if image id parameter exists
        ids = request.query_params.get('ids')
        if not ids:
            return Response({'detail': 'Missing "ids" parameter.'}, status=status.HTTP_400_BAD_REQUEST)

        # Get model type from request data
        model_type_str = request.data.get('model_type', 'unet')
        confidence_threshold = request.data.get('confidence_threshold', None)
        threshold = request.data.get('threshold', 0)
        
        try:
            # Try to convert to uppercase to match enum
            model_type = ModelType[model_type_str.upper()]
            print(f"Using model type: {model_type}")
        except (KeyError, AttributeError):
            print(f"Invalid model type: {model_type_str}, defaulting to UNET")
            model_type = ModelType.UNET

        # Parse ids
        try:
            ids = [int(id) for id in ids.split(',')]
        except ValueError:
            return Response({'detail': 'Invalid "ids" parameter.'}, status=status.HTTP_400_BAD_REQUEST)

        # Get images
        images = Picture.objects.filter(id__in=ids, dataset_id=dataset_pk)
        if not images:
            return Response({'detail': 'No images found.'}, status=status.HTTP_404_NOT_FOUND)
            
        print(f"Found {len(images)} images to process with {model_type}")
        
        # Process all images directly
        try:
            from processing.services import predict_image
            
            print(f"Processing {len(images)} images directly")
            
            successful_count = 0
            for image in images:
                try:
                    print(f"Processing image {image.id}: {image.filename}")
                    result = predict_image(image, model_type, area_threshold=threshold, 
                                          confidence_threshold=confidence_threshold)
                    if result:
                        successful_count += 1
                        print(f"Successfully processed image {image.id}, mask ID: {result.id}")
                    else:
                        print(f"Failed to process image {image.id}: No result returned")
                except Exception as img_err:
                    print(f"Error processing image {image.id}: {str(img_err)}")
                    import traceback
                    traceback.print_exc()
            
            print(f"Finished processing images. Success: {successful_count}/{len(images)}")
            return Response(status=status.HTTP_201_CREATED)
            
        except Exception as e:
            print(f"Error in bulk processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return Response({'detail': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @extend_schema(tags=['masks'],
                   summary='Delete predictions for multiple images',
                   parameters=[OpenApiParameter(name='ids', type=str, location='query', required=True)])
    @bulk_predict.mapping.delete
    def bulk_destroy_predictions(self, request: HttpRequest, dataset_pk: int = None) -> Response:
        image_ids = request.query_params.get('ids')

        if image_ids is None:
            return Response({'detail': 'No image ids provided.'}, status=status.HTTP_400_BAD_REQUEST)

        # Filter out empty strings before converting to integers
        ids = [int(id) for id in image_ids.split(',') if id.strip()]
        
        if not ids:
            return Response({'detail': 'No valid image ids provided.'}, status=status.HTTP_400_BAD_REQUEST)
            
        # Use get_queryset which already handles user authentication properly    
        images = self.get_queryset().filter(dataset=dataset_pk, id__in=ids, mask__isnull=False)

        if len(ids) != len(images):
            return Response({'detail': 'Some images do not exist or you do not have permission to access them.'}, status=status.HTTP_400_BAD_REQUEST)

        predictions = Mask.objects.filter(picture__dataset=dataset_pk, picture__id__in=ids)
        
        # Make sure user has permission to delete these masks
        if not all(mask.picture.dataset.public or 
                  (hasattr(request, 'user') and request.user is not None and mask.picture.dataset.owner == request.user) 
                  for mask in predictions):
            return Response({'detail': 'You do not have permission to delete some of these masks.'}, 
                           status=status.HTTP_403_FORBIDDEN)
        
        predictions.delete()

        return Response(status=status.HTTP_204_NO_CONTENT)

    def get_queryset(self) -> QuerySet[Picture]:
        # Check if request.user exists and is not anonymous
        if not hasattr(self.request, 'user') or self.request.user is None or self.request.user.is_anonymous:
            # For anonymous users or when user is None, only return public datasets
            return self.queryset.filter(dataset=self.kwargs['dataset_pk'], dataset__public=True)

        # For authenticated users, return both their datasets and public datasets
        is_owner_or_public = Q(dataset__owner=self.request.user) | Q(dataset__public=True)
        return self.queryset.filter(is_owner_or_public, dataset=self.kwargs['dataset_pk'])


@extend_schema(tags=['masks'])
@extend_schema_view(list=extend_schema(summary='List all predictions for an image'),
                    create=extend_schema(summary='Predict a mask for an image'),
                    retrieve=extend_schema(summary='Retrieve a prediction'),
                    partial_update=extend_schema(summary='Update a prediction'),
                    destroy=extend_schema(summary='Delete a prediction'))
class MaskViewSet(viewsets.ModelViewSet):
    serializer_class = MaskSerializer
    permission_classes = []  # Removed auth permissions
    queryset = Mask.objects.all()
    http_method_names = ['get', 'post', 'delete', 'patch']

    def list(self, request: HttpRequest, dataset_pk: int = None, image_pk: int = None) -> Response:
        queryset = self.get_queryset().filter(picture=image_pk)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def create(self, request, dataset_pk=None, image_pk=None) -> Response:
        # Handle anonymous or null user    
        if not hasattr(request, 'user') or request.user is None:
            return Response({'detail': 'Authentication required.'}, status=status.HTTP_401_UNAUTHORIZED)
        
        try:
            original = Picture.objects.get(pk=image_pk, dataset__owner=request.user)
        except Picture.DoesNotExist:
            return Response({'detail': 'Image not found.'}, status=status.HTTP_404_NOT_FOUND)

        if hasattr(original, 'mask') and original.mask is not None:
            return Response({'detail': 'Prediction already exists for this image.'}, status=status.HTTP_400_BAD_REQUEST)

        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        area_threshold = serializer.validated_data['threshold']

        # Get model type string and validate it
        model_type_str = request.data.get('model_type', 'unet')
        if not model_type_str:
            model_type_str = 'unet'  # Default to UNET
            
        # Convert to ModelType enum
        try:
            # Try to convert to uppercase to match enum
            model_type = ModelType[model_type_str.upper()]
            print(f"Using model type: {model_type}")
        except (KeyError, AttributeError):
            print(f"Invalid model type: {model_type_str}, defaulting to UNET")
            model_type = ModelType.UNET
        
        # Extract confidence threshold from request
        confidence_threshold = request.data.get('confidence_threshold')
        if confidence_threshold is not None:
            confidence_threshold = float(confidence_threshold)

        # Log what we're processing
        print(f"Processing image {image_pk} with model type {model_type}, area threshold {area_threshold}, and confidence threshold {confidence_threshold}")

        try:
            # Use direct processing instead of async for better debugging
            from processing.services import predict_image
            new_image = predict_image(original, model_type, area_threshold, confidence_threshold)
            if new_image is None:
                return Response({'detail': 'Error creating mask'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            serializer = self.get_serializer(new_image)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()
            return Response({'detail': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def partial_update(self, request: HttpRequest, dataset_pk: int = None, image_pk: int = None, pk: int = None) -> Response:
        # Handle anonymous or null user    
        if not hasattr(request, 'user') or request.user is None:
            return Response({'detail': 'Authentication required.'}, status=status.HTTP_401_UNAUTHORIZED)
            
        try:
            original_mask = Mask.objects.get(pk=pk, picture__dataset__owner=request.user)
        except Mask.DoesNotExist:
            return Response({'detail': 'Prediction not found.'}, status=status.HTTP_404_NOT_FOUND)

        serializer = self.get_serializer(data=request.data, partial=True)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        if serializer.validated_data.get('threshold', None) is None:
            return Response({'detail': 'Threshold not provided.'}, status=status.HTTP_400_BAD_REQUEST)
        area_threshold = serializer.validated_data['threshold']

        # Get model type string and validate it
        model_type_str = request.data.get('model_type', 'unet')
        if not model_type_str:
            model_type_str = 'unet'  # Default to UNET
            
        # Convert to ModelType enum
        try:
            # Try to convert to uppercase to match enum
            model_type = ModelType[model_type_str.upper()]
            print(f"Using model type: {model_type}")
        except (KeyError, AttributeError):
            print(f"Invalid model type: {model_type_str}, defaulting to UNET")
            model_type = ModelType.UNET
        
        # Extract confidence threshold from request
        confidence_threshold = request.data.get('confidence_threshold')
        if confidence_threshold is not None:
            confidence_threshold = float(confidence_threshold)

        # Log what we're processing
        print(f"Updating mask {pk} with model type {model_type}, area threshold {area_threshold}, and confidence threshold {confidence_threshold}")

        try:
            # Use direct processing instead of async for better debugging
            from processing.services import update_mask
            updated_mask = update_mask(original_mask, model_type, area_threshold, confidence_threshold)
            if updated_mask is None:
                return Response({'detail': 'Error updating mask'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            serializer = self.get_serializer(updated_mask)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            print(f"Error updating mask: {str(e)}")
            import traceback
            traceback.print_exc()
            return Response({'detail': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @extend_schema(request=LabelMeSerializer, responses={200: MaskSerializer}, summary='Create a mask using LabelMe')
    @action(detail=False, methods=['post'], url_path='labelme', serializer_class=LabelMeSerializer)
    def create_labelme(self, request: HttpRequest, dataset_pk: int = None, image_pk: int = None) -> Response:
        # Handle anonymous or null user    
        if not hasattr(request, 'user') or request.user is None:
            return Response({'detail': 'Authentication required.'}, status=status.HTTP_401_UNAUTHORIZED)
            
        serializer = self.get_serializer(data=request.data, partial=True)

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            original = Picture.objects.get(pk=image_pk, dataset__owner=request.user)
        except Picture.DoesNotExist:
            return Response({'detail': 'Image not found.'}, status=status.HTTP_404_NOT_FOUND)

        if hasattr(original, 'mask') and original.mask is not None:
            return Response({'detail': 'Prediction already exists for this image.'}, status=status.HTTP_400_BAD_REQUEST)

        if serializer.validated_data.get('json', None) is None:
            return Response({'detail': 'JSON not provided.'}, status=status.HTTP_400_BAD_REQUEST)

        image = PILImage.open(original.image)
        labelme_data = json.loads(serializer.validated_data['json'].read().decode('utf-8'))

        mask = masks.from_labelme(np.array(image), labelme_data)

        mask_arr = np.array(mask) / 255
        mask_arr = mask_arr.astype(np.uint8)
        metrics = calculate_metrics(mask_arr, 0.2581)

        mask_image = PILImage.fromarray(mask_arr)
        mask_byte_arr = io.BytesIO()
        mask_image.save(mask_byte_arr, format='PNG')
        mask = File(mask_byte_arr, name=original.filename)

        instance = serializer.save(picture=original, image=mask, **metrics)
        instance_serializer = MaskSerializer(instance)

        return Response(instance_serializer.data, status=status.HTTP_201_CREATED)

    @extend_schema(responses={(200, 'application/octet-stream'): OpenApiTypes.BINARY}, summary='Export a mask to LabelMe')
    @action(detail=True, methods=['get'], url_path='labelme')
    def export_labelme(self, request: HttpRequest, dataset_pk: int = None, image_pk: int = None, pk: int = None) -> HttpResponse:
        try:
            prediction = self.get_queryset().get(pk=pk)
        except Mask.DoesNotExist:
            return Response({'detail': 'Prediction not found.'}, status=status.HTTP_404_NOT_FOUND)

        image = PILImage.open(prediction.picture.image)
        mask = PILImage.open(prediction.image).convert('L')

        mask_arr = np.array(mask) / 255
        mask_arr = mask_arr.astype(np.uint8)

        labelme_data = masks.to_labelme(prediction.picture.filename, mask_arr)

        outfile = io.BytesIO()
        with zipfile.ZipFile(outfile, 'w') as zf:
            with zf.open('labelme.json', 'w') as f:
                f.write(labelme_data.encode('utf-8'))

            with zf.open(os.path.basename(prediction.picture.filename), 'w') as f:
                image.save(f, format='PNG')

        response = HttpResponse(outfile.getvalue(), content_type='application/octet-stream')
        response['Content-Disposition'] = f'attachment; filename={prediction.picture.filename_noext}_labelme.zip'

        return response

    def get_queryset(self) -> QuerySet[Mask]:
        # Check if request.user exists and is not anonymous
        if not hasattr(self.request, 'user') or self.request.user is None or self.request.user.is_anonymous:
            # For anonymous users or when user is None, only return masks from public datasets
            return self.queryset.filter(picture=self.kwargs['image_pk'], picture__dataset__public=True)

        # For authenticated users, return both their masks and masks from public datasets
        is_owner_or_public = Q(picture__dataset__owner=self.request.user) | Q(picture__dataset__public=True)

        return self.queryset.filter(
            is_owner_or_public,
            picture=self.kwargs['image_pk'],
        )

    def get_serializer_class(self) -> Serializer:
        if self.action == 'create_labelme':
            return LabelMeSerializer
        return MaskSerializer


@extend_schema(tags=['models'])
@extend_schema_view(list=extend_schema(summary='List all models'),
                    create=extend_schema(summary='Upload a new model'),
                    retrieve=extend_schema(summary='Retrieve a model'),
                    partial_update=extend_schema(summary='Update a model'),
                    destroy=extend_schema(summary='Delete a model'))
class ModelViewSet(viewsets.ModelViewSet):
    serializer_class = ModelSerializer
    permission_classes = []  # Removed auth permissions
    queryset = Model.objects.all()
    http_method_names = ['get', 'post', 'patch', 'delete']

    def create(self, request: HttpRequest) -> Response:
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        # Remove owner assignment
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def get_queryset(self) -> QuerySet[Model]:
        # Return all models, regardless of user authentication status
        return Model.objects.all()
