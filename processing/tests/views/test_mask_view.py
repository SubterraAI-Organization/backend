import io
import os
import json
import shutil
import tempfile
from PIL import Image as PILImage
from urllib.parse import urlparse
from rest_framework.test import APIRequestFactory, force_authenticate, APITestCase
from django.contrib.auth.models import User
from django.core.files import File
from django.test import override_settings

from processing.models import Dataset, Picture, Mask
from processing.views import MaskViewSet

MEDIA_ROOT = tempfile.mkdtemp()


class TestMaskListEndpoint(APITestCase):
    def setUp(self) -> None:
        self.client = APIRequestFactory()
        os.makedirs(MEDIA_ROOT, exist_ok=True)

        self.user = User.objects.create_user(username='test', password='test')
        self.public_dataset = Dataset.objects.create(name='public', description='public', owner=self.user, public=True)
        self.private_dataset = Dataset.objects.create(
            name='private', description='private', owner=self.user, public=False)

        image = PILImage.new('RGB', (100, 100), color='red')
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')

        self.public_picture = Picture.objects.create(
            dataset=self.public_dataset, image=File(image_bytes, name='test.png'))
        self.private_picture = Picture.objects.create(
            dataset=self.private_dataset, image=File(image_bytes, name='test.png'))
        self.public_mask = Mask.objects.create(picture=self.public_picture,
                                               image=File(image_bytes, name='test_mask.png'))
        self.private_mask = Mask.objects.create(picture=self.private_picture,
                                                image=File(image_bytes, name='test_mask.png'))

    def tearDown(self) -> None:
        shutil.rmtree(MEDIA_ROOT)

    def test_returns_public_masks_when_anonymous(self) -> None:
        request = self.client.get('masks/')
        view = MaskViewSet.as_view({'get': 'list'})
        response = view(request, image_pk=self.public_picture.id)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)

    def test_returns_private_masks_when_owner(self) -> None:
        request = self.client.get('masks/')
        force_authenticate(request, user=self.user)
        view = MaskViewSet.as_view({'get': 'list'})
        response = view(request, image_pk=self.private_picture.id)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)

    def test_does_not_return_private_masks_when_anonymous(self) -> None:
        request = self.client.get('masks/')
        view = MaskViewSet.as_view({'get': 'list'})
        response = view(request, image_pk=self.private_picture.id)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 0)

    def test_does_not_return_private_masks_when_not_owner(self) -> None:
        request = self.client.get('masks/')
        force_authenticate(request, user=User.objects.create_user(username='test2', password='test2'))
        view = MaskViewSet.as_view({'get': 'list'})
        response = view(request, image_pk=self.private_picture.id)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 0)


class TestMaskRetrieveEndpoint(APITestCase):
    def setUp(self) -> None:
        self.client = APIRequestFactory()
        os.makedirs(MEDIA_ROOT, exist_ok=True)

        self.user = User.objects.create_user(username='test', password='test')
        self.public_dataset = Dataset.objects.create(name='public', description='public', owner=self.user, public=True)
        self.private_dataset = Dataset.objects.create(
            name='private', description='private', owner=self.user, public=False)

        image = PILImage.new('RGB', (100, 100), color='red')
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')

        self.public_picture = Picture.objects.create(
            dataset=self.public_dataset, image=File(image_bytes, name='test.png'))
        self.private_picture = Picture.objects.create(
            dataset=self.private_dataset, image=File(image_bytes, name='test.png'))
        self.public_mask = Mask.objects.create(picture=self.public_picture,
                                               image=File(image_bytes, name='test_mask.png'))
        self.private_mask = Mask.objects.create(picture=self.private_picture,
                                                image=File(image_bytes, name='test_mask.png'))

    def tearDown(self) -> None:
        shutil.rmtree(MEDIA_ROOT)

    def test_all_fields_are_present(self) -> None:
        request = self.client.get(f'masks/{self.public_mask.id}/')
        force_authenticate(request, user=self.user)
        view = MaskViewSet.as_view({'get': 'retrieve'})
        response = view(request, dataset_pk=self.public_dataset.id,
                        image_pk=self.public_picture.id, pk=self.public_mask.id)

        self.assertEqual(response.status_code, 200)

        expected_data = {
            'id': self.public_mask.id,
            'picture': self.public_picture.id,
            'threshold': self.public_mask.threshold,
            'root_count': self.public_mask.root_count,
            'average_root_diameter': self.public_mask.average_root_diameter,
            'total_root_length': self.public_mask.total_root_length,
            'total_root_area': self.public_mask.total_root_area,
            'total_root_volume': self.public_mask.total_root_volume,
        }

        parsed_url = urlparse(response.data['image'])
        self.assertDictContainsSubset(expected_data, response.data)
        self.assertIn('created', response.data)
        self.assertIn('updated', response.data)
        self.assertTrue(parsed_url.scheme in ['http', 'https'])

    def test_returns_public_mask_when_anonymous(self) -> None:
        request = self.client.get(f'masks/{self.public_mask.id}/')
        view = MaskViewSet.as_view({'get': 'retrieve'})
        response = view(request, dataset_pk=self.public_dataset.id,
                        image_pk=self.public_picture.id, pk=self.public_mask.id)

        self.assertEqual(response.status_code, 200)

    def test_returns_private_mask_when_owner(self) -> None:
        request = self.client.get(f'masks/{self.private_mask.id}/')
        force_authenticate(request, user=self.user)
        view = MaskViewSet.as_view({'get': 'retrieve'})
        response = view(request, dataset_pk=self.private_dataset.id,
                        image_pk=self.private_picture.id, pk=self.private_mask.id)

        self.assertEqual(response.status_code, 200)

    def test_does_not_return_private_mask_when_anonymous(self) -> None:
        request = self.client.get(f'masks/{self.private_mask.id}/')
        view = MaskViewSet.as_view({'get': 'retrieve'})
        response = view(request, dataset_pk=self.private_dataset.id,
                        image_pk=self.private_picture.id, pk=self.private_mask.id)

        self.assertEqual(response.status_code, 404)

    def test_does_not_return_private_mask_when_not_owner(self) -> None:
        request = self.client.get(f'masks/{self.private_mask.id}/')
        force_authenticate(request, user=User.objects.create_user(username='test2', password='test2'))
        view = MaskViewSet.as_view({'get': 'retrieve'})
        response = view(request, dataset_pk=self.private_dataset.id,
                        image_pk=self.private_picture.id, pk=self.private_mask.id)

        self.assertEqual(response.status_code, 404)


class TestMaskCreateEndpoint(APITestCase):
    def setUp(self) -> None:
        self.client = APIRequestFactory()
        os.makedirs(MEDIA_ROOT, exist_ok=True)

        self.user = User.objects.create_user(username='test', password='test')
        self.dataset = Dataset.objects.create(name='test', description='test', owner=self.user)

        image = PILImage.new('RGB', (100, 100), color='red')
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')

        self.picture = Picture.objects.create(dataset=self.dataset, image=File(image_bytes, name='test.png'))

    def tearDown(self) -> None:
        shutil.rmtree(MEDIA_ROOT)

    def test_creates_mask(self) -> None:
        request = self.client.post('masks/')
        force_authenticate(request, user=self.user)
        view = MaskViewSet.as_view({'post': 'create'})
        response = view(request, dataset_pk=self.dataset.id, image_pk=self.picture.id)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(Mask.objects.filter(pk=response.data['id']).count(), 1)

    def test_cannot_create_picture_while_anonymous(self) -> None:
        request = self.client.post('masks/')
        view = MaskViewSet.as_view({'post': 'create'})
        response = view(request, dataset_pk=self.dataset.id, image_pk=self.picture.id)

        self.assertEqual(response.status_code, 401)
        self.assertEqual(Mask.objects.filter(picture=self.picture).count(), 0)

    def test_cannot_create_picture_while_not_owner(self) -> None:
        request = self.client.post('masks/')
        force_authenticate(request, user=User.objects.create_user(username='test2', password='test2'))
        view = MaskViewSet.as_view({'post': 'create'})
        response = view(request, dataset_pk=self.dataset.id, image_pk=self.picture.id)

        self.assertEqual(response.status_code, 404)
        self.assertEqual(Mask.objects.filter(picture=self.picture).count(), 0)


class TestMaskUpdateEndpoint(APITestCase):
    def setUp(self) -> None:
        self.client = APIRequestFactory()
        os.makedirs(MEDIA_ROOT, exist_ok=True)

        self.user = User.objects.create_user(username='test', password='test')
        self.dataset = Dataset.objects.create(name='test', description='test', owner=self.user)

        image = PILImage.new('RGB', (100, 100), color='red')
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')

        self.picture = Picture.objects.create(dataset=self.dataset, image=File(image_bytes, name='test.png'))
        self.mask = Mask.objects.create(picture=self.picture, image=File(image_bytes, name='test_mask.png'))

    def tearDown(self) -> None:
        shutil.rmtree(MEDIA_ROOT)
        self.client = APIRequestFactory()

    def test_updates_mask(self) -> None:
        data = {'threshold': 5}
        request = self.client.patch(f'masks/{self.mask.id}/', data)
        force_authenticate(request, user=self.user)
        view = MaskViewSet.as_view({'patch': 'partial_update'})

        response = view(request, dataset_pk=self.dataset.id, image_pk=self.picture.id, pk=self.mask.id)
        self.mask.refresh_from_db()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.mask.threshold, 5)

    def test_cannot_update_mask_while_anonymous(self) -> None:
        request = self.client.patch(f'masks/{self.mask.id}/')
        view = MaskViewSet.as_view({'patch': 'partial_update'})
        response = view(request, dataset_pk=self.dataset.id, image_pk=self.picture.id, pk=self.mask.id)

        self.mask.refresh_from_db()

        self.assertEqual(response.status_code, 401)
        self.assertNotEqual(self.mask.threshold, 5)

    def test_cannot_update_mask_while_not_owner(self) -> None:
        request = self.client.patch(f'masks/{self.mask.id}/')
        force_authenticate(request, user=User.objects.create_user(username='test2', password='test2'))
        view = MaskViewSet.as_view({'patch': 'partial_update'})
        response = view(request, dataset_pk=self.dataset.id, image_pk=self.picture.id, pk=self.mask.id)

        self.mask.refresh_from_db()

        self.assertEqual(response.status_code, 404)
        self.assertNotEqual(self.mask.threshold, 5)


class TestMaskDeleteEndpoint(APITestCase):
    def setUp(self) -> None:
        self.client = APIRequestFactory()
        os.makedirs(MEDIA_ROOT, exist_ok=True)

        self.user = User.objects.create_user(username='test', password='test')
        self.dataset = Dataset.objects.create(name='test', description='test', owner=self.user)

        image = PILImage.new('RGB', (100, 100), color='red')
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')

        self.picture = Picture.objects.create(dataset=self.dataset, image=File(image_bytes, name='test.png'))
        self.mask = Mask.objects.create(picture=self.picture, image=File(image_bytes, name='test_mask.png'))

    def tearDown(self) -> None:
        shutil.rmtree(MEDIA_ROOT)
        self.client = APIRequestFactory()

    def test_deletes_picture(self) -> None:
        request = self.client.delete(f'masks/{self.mask.id}/')
        force_authenticate(request, user=self.user)
        view = MaskViewSet.as_view({'delete': 'destroy'})
        response = view(request, dataset_pk=self.dataset.id, image_pk=self.picture.id, pk=self.mask.id)

        self.assertEqual(response.status_code, 204)
        with self.assertRaises(Mask.DoesNotExist):
            Mask.objects.get(id=self.mask.id)

    def test_cannot_delete_picture_while_anonymous(self) -> None:
        request = self.client.delete(f'masks/{self.mask.id}/')
        view = MaskViewSet.as_view({'delete': 'destroy'})
        response = view(request, dataset_pk=self.dataset.id, image_pk=self.picture.id, pk=self.mask.id)

        self.assertEqual(response.status_code, 401)
        self.assertEqual(Mask.objects.filter(pk=self.mask.id).count(), 1)

    def test_cannot_delete_picture_while_not_owner(self) -> None:
        request = self.client.delete(f'masks/{self.mask.id}/')
        force_authenticate(request, user=User.objects.create_user(username='test2', password='test2'))
        view = MaskViewSet.as_view({'delete': 'destroy'})
        response = view(request, dataset_pk=self.dataset.id, image_pk=self.picture.id, pk=self.mask.id)

        self.assertEqual(response.status_code, 404)
        self.assertEqual(Mask.objects.filter(pk=self.mask.id).count(), 1)


class TestMaskCreateLabelmeEndpoint(APITestCase):
    def setUp(self) -> None:
        self.client = APIRequestFactory()
        os.makedirs(MEDIA_ROOT, exist_ok=True)

        self.user = User.objects.create_user(username='test', password='test')
        self.dataset = Dataset.objects.create(name='test', description='test', owner=self.user)

        image = PILImage.new('RGB', (100, 100), color='red')
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')

        self.picture = Picture.objects.create(dataset=self.dataset, image=File(image_bytes, name='test.png'))

    def tearDown(self) -> None:
        shutil.rmtree(MEDIA_ROOT)

    def test_creates_mask(self) -> None:
        json_data = {
            'shapes': []
        }
        tmp_file = tempfile.NamedTemporaryFile(suffix='.json')
        tmp_file.write(json.dumps(json_data).encode('utf-8'))
        tmp_file.seek(0)

        data = {'json': tmp_file}
        request = self.client.post('masks/labelme/', data)
        force_authenticate(request, user=self.user)
        view = MaskViewSet.as_view({'post': 'create_labelme'})
        response = view(request, dataset_pk=self.dataset.id, image_pk=self.picture.id)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(Mask.objects.filter(pk=self.picture.id).count(), 1)

    def test_throws_if_no_json(self) -> None:
        request = self.client.post('masks/labelme/')
        force_authenticate(request, user=self.user)
        view = MaskViewSet.as_view({'post': 'create_labelme'})
        response = view(request, dataset_pk=self.dataset.id, image_pk=self.picture.id)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(Mask.objects.filter(picture=self.picture).count(), 0)

    def test_cannot_create_mask_while_anonymous(self) -> None:
        request = self.client.post('masks/labelme/')
        view = MaskViewSet.as_view({'post': 'create_labelme'})
        response = view(request, dataset_pk=self.dataset.id, image_pk=self.picture.id)

        self.assertEqual(response.status_code, 401)
        self.assertEqual(Mask.objects.filter(picture=self.picture).count(), 0)

    def test_cannot_create_mask_while_not_owner(self) -> None:
        request = self.client.post('masks/labelme/')
        force_authenticate(request, user=User.objects.create_user(username='test2', password='test2'))
        view = MaskViewSet.as_view({'post': 'create_labelme'})
        response = view(request, dataset_pk=self.dataset.id, image_pk=self.picture.id)

        self.assertEqual(response.status_code, 404)
        self.assertEqual(Mask.objects.filter(picture=self.picture).count(), 0)


class TestMaskExportLabelmeEndpoint(APITestCase):
    def setUp(self) -> None:
        self.client = APIRequestFactory()
        os.makedirs(MEDIA_ROOT, exist_ok=True)

        self.user = User.objects.create_user(username='test', password='test')
        self.public_dataset = Dataset.objects.create(name='public', description='public', owner=self.user, public=True)
        self.private_dataset = Dataset.objects.create(
            name='private', description='private', owner=self.user, public=False)

        image = PILImage.new('RGB', (100, 100), color='red')
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')

        self.public_picture = Picture.objects.create(
            dataset=self.public_dataset, image=File(image_bytes, name='test.png'))
        self.private_picture = Picture.objects.create(
            dataset=self.private_dataset, image=File(image_bytes, name='test.png'))

        self.public_mask = Mask.objects.create(picture=self.public_picture,
                                               image=File(image_bytes, name='test_mask.png'))
        self.private_mask = Mask.objects.create(picture=self.private_picture,
                                                image=File(image_bytes, name='test_mask.png'))

    def tearDown(self) -> None:
        shutil.rmtree(MEDIA_ROOT)

    def test_can_export_labelme_while_anonymous(self) -> None:
        request = self.client.get(f'masks/{self.public_mask.id}/labelme/')

        view = MaskViewSet.as_view({'get': 'export_labelme'})
        response = view(request, dataset_pk=self.public_dataset.id,
                        image_pk=self.public_picture.id, pk=self.public_mask.id)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/octet-stream')

    def test_can_export_labelme_while_owner(self) -> None:
        request = self.client.get(f'masks/{self.private_mask.id}/labelme/')
        force_authenticate(request, user=self.user)

        view = MaskViewSet.as_view({'get': 'export_labelme'})
        response = view(request, dataset_pk=self.private_dataset.id,
                        image_pk=self.private_picture.id, pk=self.private_mask.id)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/octet-stream')

    def test_does_not_return_private_pictures_when_anonymous(self) -> None:
        request = self.client.get(f'masks/{self.private_mask.id}/labelme/')
        view = MaskViewSet.as_view({'get': 'export_labelme'})
        response = view(request, dataset_pk=self.private_dataset.id, image_pk=self.private_picture.id,
                        pk=self.private_mask.id)

        self.assertEqual(response.status_code, 404)

    def test_does_not_return_private_pictures_when_not_owner(self) -> None:
        request = self.client.get(f'masks/{self.private_mask.id}/labelme/')
        force_authenticate(request, user=User.objects.create_user(username='test2', password='test2'))
        view = MaskViewSet.as_view({'get': 'export_labelme'})
        response = view(request, dataset_pk=self.private_dataset.id, image_pk=self.private_picture.id,
                        pk=self.private_mask.id)

        self.assertEqual(response.status_code, 404)


@override_settings(MEDIA_ROOT=MEDIA_ROOT)
class TestMaskViewSet(APITestCase):
    def setUp(self) -> None:
        self.user = User.objects.create_user(username='test', password='test')
        self.dataset = Dataset.objects.create(name='test', description='test', owner=self.user)

        image = PILImage.new('RGB', (100, 100), color='red')
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')

        self.picture = Picture.objects.create(dataset=self.dataset, image=File(image_bytes, name='test.png'))
        self.mask = Mask.objects.create(picture=self.picture, image=File(image_bytes, name='test_mask.png'))

        self.picture_no_mask = Picture.objects.create(dataset=self.dataset, image=File(image_bytes, name='test.png'))
        self.client = APIRequestFactory()

    def test_list_endpoint(self) -> None:
        request = self.client.get('masks/')
        force_authenticate(request, user=self.user)

        view = MaskViewSet.as_view({'get': 'list'})
        response = view(request, image_pk=self.picture.id)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)

    def test_retrieve_endpoint(self) -> None:
        request = self.client.get('masks/')
        force_authenticate(request, user=self.user)

        view = MaskViewSet.as_view({'get': 'retrieve'})
        response = view(request, dataset_pk=self.dataset.id, image_pk=self.picture.id, pk=self.mask.id)

        self.assertEqual(response.status_code, 200)

        expected_data = {
            'id': self.mask.id,
            'picture': self.picture.id,
            'threshold': self.mask.threshold
        }
        self.assertDictContainsSubset(expected_data, response.data)
        self.assertIn('root_count', response.data)
        self.assertIn('average_root_diameter', response.data)
        self.assertIn('total_root_length', response.data)
        self.assertIn('total_root_area', response.data)
        self.assertIn('total_root_volume', response.data)

        parsed_url = urlparse(response.data['image'])
        self.assertTrue(parsed_url.scheme in ['http', 'https'])

    def test_create_endpoint(self) -> None:
        data = {'threshold': 0}
        request = self.client.post('masks/', data)
        force_authenticate(request, user=self.user)

        view = MaskViewSet.as_view({'post': 'create'})
        response = view(request, dataset_pk=self.dataset.id, image_pk=self.picture_no_mask.id)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(Mask.objects.filter(pk=response.data['id']).count(), 1)

    def test_update_endpoint(self) -> None:
        data = {'threshold': 5}
        request = self.client.patch(f'masks/{self.mask.id}/', data)
        force_authenticate(request, user=self.user)

        view = MaskViewSet.as_view({'patch': 'partial_update'})
        response = view(request, dataset_pk=self.dataset.id, image_pk=self.picture.id, pk=self.mask.id)
        self.assertEqual(response.status_code, 200)

        self.mask.refresh_from_db()
        self.assertEqual(self.mask.threshold, 5)

    def test_delete_endpoint(self) -> None:
        request = self.client.delete(f'masks/{self.mask.id}/')
        force_authenticate(request, user=self.user)

        view = MaskViewSet.as_view({'delete': 'destroy'})
        response = view(request, dataset_pk=self.dataset.id, image_pk=self.picture.id, pk=self.mask.id)
        self.assertEqual(response.status_code, 204)

        with self.assertRaises(Mask.DoesNotExist):
            Mask.objects.get(id=self.mask.id)

    def test_create_labelme_endpoint(self) -> None:
        json_data = {
            'shapes': []
        }

        tmp_file = tempfile.NamedTemporaryFile(suffix='.json')
        tmp_file.write(json.dumps(json_data).encode('utf-8'))
        tmp_file.seek(0)

        data = {'json': tmp_file}

        request = self.client.post('masks/labelme/', data)
        force_authenticate(request, user=self.user)

        view = MaskViewSet.as_view({'post': 'create_labelme'})
        response = view(request, dataset_pk=self.dataset.id, image_pk=self.picture_no_mask.id)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(Mask.objects.filter(pk=self.picture_no_mask.id).count(), 1)

    def test_export_labelme_endpoint(self) -> None:
        request = self.client.get(f'masks/{self.mask.id}/labelme/')
        force_authenticate(request, user=self.user)

        view = MaskViewSet.as_view({'get': 'export_labelme'})
        response = view(request, dataset_pk=self.dataset.id, image_pk=self.picture.id, pk=self.mask.id)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/octet-stream')


# @override_settings(MEDIA_ROOT=MEDIA_ROOT)
# class TestModelViewSet(APITestCase):
#     def setUp(self) -> None:
#         self.user = User.objects.create_user(username='test', password='test')
#         # self.model = Model.objects.create(name="test_model", model_type=Model.UNET, model_weights=)

#     def test_list_endpoint(self) -> None:
#         request = self.client.get('datasets/')
#         force_authenticate(request, user=self.user)

#         view = DatasetViewSet.as_view({'get': 'list'})
#         response = view(request)
#         self.assertEqual(response.status_code, 200)
#         self.assertEqual(len(response.data), 0)
