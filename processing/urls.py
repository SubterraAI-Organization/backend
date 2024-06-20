from django.urls import path
from django.urls.conf import include
from rest_framework_nested.routers import SimpleRouter, NestedSimpleRouter

from processing.routers import BulkNestedRouter
from processing import views

router = SimpleRouter()
router.register('datasets', views.DatasetViewSet, basename='datasets')
router.register('models', views.ModelViewSet, basename='models')

image_router = BulkNestedRouter(router, 'datasets', lookup='dataset')
image_router.register('images', views.PictureViewSet, basename='images')

mask_router = NestedSimpleRouter(image_router, 'images', lookup='image')
mask_router.register('masks', views.MaskViewSet, basename='masks')


app_name = 'segmentation'
urlpatterns = [
    path('api/', include(router.urls)),
    path('api/', include(image_router.urls)),
    path('api/', include(mask_router.urls)),
]
