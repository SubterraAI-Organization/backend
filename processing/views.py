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

from processing.models import Dataset, Picture, Mask, Model
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
    permission_classes = [permissions.IsAuthenticatedOrReadOnly, IsOwnerOrReadOnly]
    http_method_names = ['get', 'post', 'patch', 'delete']

    def create(self, request: HttpRequest) -> Response:
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        serializer.save(owner=request.user)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def get_queryset(self) -> QuerySet[Dataset]:
        if self.request.user.is_anonymous:
            return self.queryset.filter(public=True)

        is_owner_or_public = Q(owner=self.request.user) | Q(public=True)

        return self.queryset.filter(is_owner_or_public)


@extend_schema(tags=['pictures'])
@extend_schema_view(
    list=extend_schema(summary='List all images in a dataset'),
    create=extend_schema(summary='Upload a new image'),
    retrieve=extend_schema(summary='Retrieve an image'),
    destroy=extend_schema(summary='Delete an image')
)
class PictureViewSet(viewsets.ModelViewSet):
    queryset = Picture.objects.all()
    serializer_class = PictureSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly, IsOwnerOrReadOnly]
    http_method_names = ['get', 'post', 'delete']

    def create(self, request: HttpRequest, dataset_pk: int = None) -> Response:
        try:
            dataset = Dataset.objects.get(pk=dataset_pk, owner=request.user)
        except Dataset.DoesNotExist:
            return Response({'detail': 'Dataset not found.'}, status=status.HTTP_404_NOT_FOUND)

        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        serializer.save(dataset=dataset)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @extend_schema(summary='Delete multiple images',
                   parameters=[OpenApiParameter(name='ids', type=str, location='query', required=True)])
    def bulk_destroy(self, request: HttpRequest, dataset_pk: int = None, format: str = None) -> Response:
        ids = [int(id) for id in request.query_params.get('ids').split(',')]
        images = self.queryset.filter(id__in=ids, dataset=dataset_pk, dataset__owner=request.user)

        if len(ids) != len(images):
            return Response({'detail': 'Some images do not exist.'}, status=status.HTTP_404_NOT_FOUND)

        images.delete()

        return Response(status=status.HTTP_204_NO_CONTENT)

    @extend_schema(tags=['masks'], summary='Predict masks for multiple images', parameters=[OpenApiParameter(
        name='ids', type=str, location='query', required=True)], responses={201: None})
    @action(detail=False, methods=['post'], url_path='predict', serializer_class=MaskSerializer)
    def bulk_predict(self, request: HttpRequest, dataset_pk: int = None) -> Response:
        image_ids = request.query_params.get('ids')

        if image_ids is None:
            return Response({'detail': 'No image ids provided.'}, status=status.HTTP_400_BAD_REQUEST)

        ids = [int(id) for id in image_ids.split(',')]
        images = self.get_queryset().filter(dataset=dataset_pk, id__in=ids)

        if len(ids) != len(images):
            return Response({'detail': 'Some images do not exist.'}, status=status.HTTP_400_BAD_REQUEST)

        for image in images:
            if hasattr(image, 'mask') and image.mask is not None:
                return Response({'detail': 'Mask already exists for some images.'}, status=status.HTTP_400_BAD_REQUEST)

        area_threshold = int(request.data.get('threshold', 15))

        async_task('processing.services.bulk_predict_images', images, area_threshold)

        return Response(status=status.HTTP_201_CREATED)

    @extend_schema(tags=['masks'], summary='Delete predictions for multiple images',
                   parameters=[OpenApiParameter(name='ids', type=str, location='query', required=True)])
    @bulk_predict.mapping.delete
    def bulk_destroy_predictions(self, request: HttpRequest, dataset_pk: int = None) -> Response:
        image_ids = request.query_params.get('ids')

        if image_ids is None:
            return Response({'detail': 'No image ids provided.'}, status=status.HTTP_400_BAD_REQUEST)

        ids = [int(id) for id in image_ids.split(',')]
        images = self.get_queryset().filter(dataset=dataset_pk, id__in=ids, mask__isnull=False)

        if len(ids) != len(images):
            return Response({'detail': 'Some images do not exist.'}, status=status.HTTP_400_BAD_REQUEST)

        predictions = Mask.objects.filter(picture__dataset=dataset_pk, picture__id__in=ids)
        predictions.delete()

        return Response(status=status.HTTP_204_NO_CONTENT)

    def get_queryset(self) -> QuerySet[Picture]:
        if self.request.user.is_anonymous:
            return self.queryset.filter(dataset=self.kwargs['dataset_pk'], dataset__public=True)

        is_owner_or_public = Q(dataset__owner=self.request.user) | Q(dataset__public=True)

        return self.queryset.filter(is_owner_or_public, dataset=self.kwargs['dataset_pk'])


@extend_schema(tags=['masks'])
@extend_schema_view(
    list=extend_schema(summary='List all predictions for an image'),
    create=extend_schema(summary='Predict a mask for an image'),
    retrieve=extend_schema(summary='Retrieve a prediction'),
    partial_update=extend_schema(summary='Update a prediction'),
    destroy=extend_schema(summary='Delete a prediction')
)
class MaskViewSet(viewsets.ModelViewSet):
    serializer_class = MaskSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly, IsOwnerOrReadOnly]
    queryset = Mask.objects.all()
    http_method_names = ['get', 'post', 'delete', 'patch']

    def list(self, request: HttpRequest, dataset_pk: int = None, image_pk: int = None) -> Response:
        queryset = self.get_queryset().filter(picture=image_pk)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def create(self, request, dataset_pk=None, image_pk=None) -> Response:
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

        task_id = async_task('processing.services.predict_image', original, area_threshold)
        new_image = result(task_id)

        serializer = self.get_serializer(new_image)

        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def partial_update(self, request: HttpRequest, dataset_pk: int = None,
                       image_pk: int = None, pk: int = None) -> Response:
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

        task_id = async_task('processing.services.update_mask', original_mask, area_threshold)
        serializer = self.get_serializer(result(task_id))

        return Response(serializer.data, status=status.HTTP_200_OK)

    @extend_schema(request=LabelMeSerializer, responses={200: MaskSerializer}, summary='Create a mask using LabelMe')
    @action(detail=False, methods=['post'], url_path='labelme', serializer_class=LabelMeSerializer)
    def create_labelme(self, request: HttpRequest, dataset_pk: int = None, image_pk: int = None) -> Response:
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

    @extend_schema(responses={(200, 'application/octet-stream'): OpenApiTypes.BINARY},
                   summary='Export a mask to LabelMe')
    @action(detail=True, methods=['get'], url_path='labelme')
    def export_labelme(self, request: HttpRequest, dataset_pk: int = None,
                       image_pk: int = None, pk: int = None) -> HttpResponse:
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

        response = HttpResponse(
            outfile.getvalue(), content_type='application/octet-stream')
        response['Content-Disposition'] = f'attachment; filename={prediction.picture.filename_noext}_labelme.zip'

        return response

    def get_queryset(self) -> QuerySet[Mask]:
        if self.request.user.is_anonymous:
            return self.queryset.filter(picture=self.kwargs['image_pk'], picture__dataset__public=True)

        is_owner_or_public = Q(picture__dataset__owner=self.request.user) | Q(picture__dataset__public=True)

        return self.queryset.filter(is_owner_or_public, picture=self.kwargs['image_pk'],)

    def get_serializer_class(self) -> Serializer:
        if self.action == 'create_labelme':
            return LabelMeSerializer
        return MaskSerializer


@extend_schema(tags=['models'])
@extend_schema_view(
    list=extend_schema(summary='List all models'),
    create=extend_schema(summary='Upload a new model'),
    retrieve=extend_schema(summary='Retrieve a model'),
    partial_update=extend_schema(summary='Update a model'),
    destroy=extend_schema(summary='Delete a model')
)
class ModelViewSet(viewsets.ModelViewSet):
    serializer_class = ModelSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    queryset = Model.objects.all()
    http_method_names = ['get', 'post', 'patch', 'delete']

    def create(self, request: HttpRequest) -> Response:
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        serializer.save(owner=request.user)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def get_queryset(self) -> QuerySet[Model]:
        if self.request.user.is_anonymous:
            return self.queryset.filter(public=True)

        return self.queryset.filter(Q(owner=self.request.user) | Q(public=True))
