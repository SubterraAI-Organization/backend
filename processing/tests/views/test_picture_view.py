import os
import tempfile
import shutil
import io
from PIL import Image as PILImage
from urllib.parse import urlparse
from rest_framework.test import APIRequestFactory, force_authenticate, APITestCase
from django.core.files import File
from django.test import override_settings
from django.utils.http import urlencode
from django.contrib.auth.models import User

from processing.models import Dataset, Picture, Mask
from processing.views import PictureViewSet


MEDIA_ROOT = tempfile.mkdtemp()


@override_settings(MEDIA_ROOT=MEDIA_ROOT)
class TestPictureListEndpoint(APITestCase):
    def setUp(self) -> None:
        self.client = APIRequestFactory()
        os.makedirs(MEDIA_ROOT, exist_ok=True)

        self.user = User.objects.create_user(username='test', password='test')
        self.public = Dataset.objects.create(name='private', description='test', owner=self.user, public=True)
        self.private = Dataset.objects.create(name='test', description='test', owner=self.user)

        image = PILImage.new('RGB', (100, 100), color='red')
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')

        Picture.objects.bulk_create([
            Picture(dataset=self.public, image=File(image_bytes, name='test.png')),
            Picture(dataset=self.private, image=File(image_bytes, name='test.png'))
        ])

    def tearDown(self) -> None:
        shutil.rmtree(MEDIA_ROOT)

    def test_returns_public_pictures_when_anonymous(self) -> None:
        request = self.client.get('images/')
        view = PictureViewSet.as_view({'get': 'list'})
        response = view(request, dataset_pk=self.public.id)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)

    def test_returns_private_pictures_when_owner(self) -> None:
        request = self.client.get('images/')
        force_authenticate(request, user=self.user)
        view = PictureViewSet.as_view({'get': 'list'})
        response = view(request, dataset_pk=self.private.id)

        self.assertEqual(len(response.data), 1)

    def test_does_not_return_private_pictures_when_anonymous(self) -> None:
        request = self.client.get('images/')
        view = PictureViewSet.as_view({'get': 'list'})
        response = view(request, dataset_pk=self.private.id)

        self.assertEqual(len(response.data), 0)

    def test_does_not_return_private_pictures_when_not_owner(self) -> None:
        request = self.client.get('images/')
        force_authenticate(request, user=User.objects.create_user(username='test2', password='test'))
        view = PictureViewSet.as_view({'get': 'list'})
        response = view(request, dataset_pk=self.private.id)

        self.assertEqual(len(response.data), 0)


@override_settings(MEDIA_ROOT=MEDIA_ROOT)
class TestPictureRetrieveEndpoint(APITestCase):
    def setUp(self) -> None:
        self.client = APIRequestFactory()
        os.makedirs(MEDIA_ROOT, exist_ok=True)

        self.user = User.objects.create_user(username='test', password='test')
        self.public_dataset = Dataset.objects.create(name='public', description='test', owner=self.user, public=True)
        self.private_dataset = Dataset.objects.create(name='private', description='test', owner=self.user)

        image = PILImage.new('RGB', (100, 100), color='red')
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')

        self.public_picture = Picture.objects.create(
            dataset=self.public_dataset, image=File(image_bytes, name='test.png'))
        self.private_picture = Picture.objects.create(
            dataset=self.private_dataset, image=File(image_bytes, name='test.png'))

    def tearDown(self) -> None:
        shutil.rmtree(MEDIA_ROOT)

    def test_all_fields_are_present(self) -> None:
        request = self.client.get(f'images/{self.public_picture.id}/')
        view = PictureViewSet.as_view({'get': 'retrieve'})
        response = view(request, dataset_pk=self.public_dataset.id, pk=self.public_picture.id)

        expected_data = {
            'id': self.public_picture.id,
            'dataset': self.public_dataset.id,
        }
        parsed_url = urlparse(response.data['image'])
        self.assertDictContainsSubset(expected_data, response.data)
        self.assertIn('created', response.data)
        self.assertIn('updated', response.data)
        self.assertTrue(parsed_url.scheme in ['http', 'https'])
        self.assertTrue(response.data['image'].startswith('http'))

    def test_returns_public_picture_when_anonymous(self) -> None:
        request = self.client.get(f'images/{self.public_picture.id}/')
        view = PictureViewSet.as_view({'get': 'retrieve'})
        response = view(request, dataset_pk=self.public_dataset.id, pk=self.public_picture.id)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['id'], self.public_picture.id)

    def test_returns_private_image_when_owner(self) -> None:
        request = self.client.get(f'images/{self.private_picture.id}/')
        force_authenticate(request, user=self.user)
        view = PictureViewSet.as_view({'get': 'retrieve'})
        response = view(request, dataset_pk=self.private_dataset.id, pk=self.private_picture.id)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['id'], self.private_picture.id)

    def test_does_not_return_private_pictures_when_anonymous(self) -> None:
        request = self.client.get(f'images/{self.private_picture.id}/')
        view = PictureViewSet.as_view({'get': 'retrieve'})
        response = view(request, dataset_pk=self.private_dataset.id, pk=self.private_picture.id)

        self.assertEqual(response.status_code, 404)

    def test_does_not_return_private_pictures_when_not_owner(self) -> None:
        request = self.client.get(f'images/{self.private_picture.id}/')
        force_authenticate(request, user=User.objects.create_user(username='test2', password='test'))
        view = PictureViewSet.as_view({'get': 'retrieve'})
        response = view(request, dataset_pk=self.private_dataset.id, pk=self.private_picture.id)

        self.assertEqual(response.status_code, 404)


@override_settings(MEDIA_ROOT=MEDIA_ROOT)
class TestPictureCreateEndpoint(APITestCase):
    def setUp(self) -> None:
        self.client = APIRequestFactory()
        os.makedirs(MEDIA_ROOT, exist_ok=True)

        self.user = User.objects.create_user(username='test', password='test')
        self.dataset = Dataset.objects.create(name='test', description='test', owner=self.user)

        image = PILImage.new('RGB', (100, 100), color='red')
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        self.file = File(image_bytes, name='test.png')

        self.picture = Picture.objects.create(dataset=self.dataset, image=self.file)

    def tearDown(self) -> None:
        shutil.rmtree(MEDIA_ROOT)

    def test_creates_picture(self) -> None:
        image = PILImage.new('RGB', (100, 100), color='red')
        tmp_file = tempfile.NamedTemporaryFile(suffix='.png')
        image.save(tmp_file)
        tmp_file.seek(0)
        data = {'image': tmp_file}

        request = self.client.post('images/', data)
        force_authenticate(request, user=self.user)
        view = PictureViewSet.as_view({'post': 'create'})
        response = view(request, dataset_pk=self.dataset.id)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(Picture.objects.filter(pk=response.data['id']).count(), 1)

    def test_throws_if_no_image(self) -> None:
        request = self.client.post('images/')
        force_authenticate(request, user=self.user)
        view = PictureViewSet.as_view({'post': 'create'})
        response = view(request, dataset_pk=self.dataset.id)

        self.assertEqual(response.status_code, 400)

    def test_cannot_create_picture_while_anonymous(self) -> None:
        request = self.client.post('images/')
        view = PictureViewSet.as_view({'post': 'create'})
        response = view(request, dataset_pk=self.dataset.id)

        self.assertEqual(response.status_code, 401)

    def test_cannot_create_picture_while_not_owner(self) -> None:
        request = self.client.post('images/')
        force_authenticate(request, user=User.objects.create_user(username='test2', password='test'))
        view = PictureViewSet.as_view({'post': 'create'})
        response = view(request, dataset_pk=self.dataset.id)

        self.assertEqual(response.status_code, 404)


@override_settings(MEDIA_ROOT=MEDIA_ROOT)
class TestPictureDeleteEndpoint(APITestCase):
    def setUp(self) -> None:
        self.client = APIRequestFactory()
        os.makedirs(MEDIA_ROOT, exist_ok=True)

        self.user = User.objects.create_user(username='test', password='test')
        self.dataset = Dataset.objects.create(name='test', description='test', owner=self.user)

        image = PILImage.new('RGB', (100, 100), color='red')
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        self.file = File(image_bytes, name='test.png')

        self.picture = Picture.objects.create(dataset=self.dataset, image=self.file)

    def tearDown(self) -> None:
        shutil.rmtree(MEDIA_ROOT)

    def test_deletes_picture(self) -> None:
        request = self.client.delete(f'images/{self.picture.id}/')
        force_authenticate(request, user=self.user)
        view = PictureViewSet.as_view({'delete': 'destroy'})
        response = view(request, dataset_pk=self.dataset.id, pk=self.picture.id)

        self.assertEqual(response.status_code, 204)
        with self.assertRaises(Picture.DoesNotExist):
            Picture.objects.get(id=self.picture.id)

    def test_cannot_delete_picture_while_anonymous(self) -> None:
        request = self.client.delete(f'images/{self.picture.id}/')
        view = PictureViewSet.as_view({'delete': 'destroy'})
        response = view(request, dataset_pk=self.dataset.id, pk=self.picture.id)

        self.assertEqual(response.status_code, 401)

    def test_cannot_delete_picture_while_not_owner(self) -> None:
        request = self.client.delete(f'images/{self.picture.id}/')
        force_authenticate(request, user=User.objects.create_user(username='test2', password='test'))
        view = PictureViewSet.as_view({'delete': 'destroy'})
        response = view(request, dataset_pk=self.dataset.id, pk=self.picture.id)

        self.assertEqual(response.status_code, 404)


@override_settings(MEDIA_ROOT=MEDIA_ROOT)
class TestPictureBulkDestroyEndpoint(APITestCase):
    def setUp(self) -> None:
        self.client = APIRequestFactory()
        os.makedirs(MEDIA_ROOT, exist_ok=True)

        self.user = User.objects.create_user(username='test', password='test')
        self.dataset = Dataset.objects.create(name='test', description='test', owner=self.user)

        image = PILImage.new('RGB', (100, 100), color='red')
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')

        self.pictures = Picture.objects.bulk_create([
            Picture(dataset=self.dataset, image=File(image_bytes, name='test.png')),
            Picture(dataset=self.dataset, image=File(image_bytes, name='test.png'))
        ])

    def tearDown(self) -> None:
        shutil.rmtree(MEDIA_ROOT)

    def test_deletes_pictures(self) -> None:
        request = self.client.delete('images/bulk_destroy/',
                                     QUERY_STRING=urlencode({'ids': f'{self.pictures[0].id},{self.pictures[1].id}'}))
        force_authenticate(request, user=self.user)
        view = PictureViewSet.as_view({'delete': 'bulk_destroy'})
        response = view(request, dataset_pk=self.dataset.id)

        self.assertEqual(response.status_code, 204)
        self.assertEqual(Picture.objects.filter(dataset=self.dataset).count(), 0)

    def test_cannot_delete_pictures_while_anonymous(self) -> None:
        request = self.client.delete('images/bulk_destroy/',
                                     QUERY_STRING=urlencode({'ids': f'{self.pictures[0].id},{self.pictures[1].id}'}))
        view = PictureViewSet.as_view({'delete': 'bulk_destroy'})
        response = view(request, dataset_pk=self.dataset.id)

        self.assertEqual(response.status_code, 401)
        self.assertEqual(Picture.objects.filter(dataset=self.dataset).count(), 2)

    def test_cannot_delete_pictures_while_not_owner(self) -> None:
        request = self.client.delete('images/bulk_destroy/',
                                     QUERY_STRING=urlencode({'ids': f'{self.pictures[0].id},{self.pictures[1].id}'}))
        force_authenticate(request, user=User.objects.create_user(username='test2', password='test'))
        view = PictureViewSet.as_view({'delete': 'bulk_destroy'})
        response = view(request, dataset_pk=self.dataset.id)

        self.assertEqual(response.status_code, 404)
        self.assertEqual(Picture.objects.filter(dataset=self.dataset).count(), 2)


@override_settings(MEDIA_ROOT=MEDIA_ROOT)
class TestPictureBulkPredictEndpoint(APITestCase):
    def setUp(self) -> None:
        self.client = APIRequestFactory()
        os.makedirs(MEDIA_ROOT, exist_ok=True)

        self.user = User.objects.create_user(username='test', password='test')
        self.dataset = Dataset.objects.create(name='test', description='test', owner=self.user)

        image = PILImage.new('RGB', (100, 100), color='red')
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')

        self.pictures = Picture.objects.bulk_create([
            Picture(dataset=self.dataset, image=File(image_bytes, name='test.png')),
            Picture(dataset=self.dataset, image=File(image_bytes, name='test.png'))
        ])

    def tearDown(self) -> None:
        shutil.rmtree(MEDIA_ROOT)

    def test_creates_predictions(self) -> None:
        request = self.client.post('images/bulk_predict/',
                                   QUERY_STRING=urlencode({'ids': f'{self.pictures[0].id},{self.pictures[1].id}'}))
        force_authenticate(request, user=self.user)

        view = PictureViewSet.as_view({'post': 'bulk_predict'})
        response = view(request, dataset_pk=self.dataset.id)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(Mask.objects.filter(picture__dataset=self.dataset).count(), 2)

    def test_cannot_create_predictions_while_anonymous(self) -> None:
        request = self.client.post('images/bulk_predict/',
                                   QUERY_STRING=urlencode({'ids': f'{self.pictures[0].id},{self.pictures[1].id}'}))
        view = PictureViewSet.as_view({'post': 'bulk_predict'})
        response = view(request, dataset_pk=self.dataset.id)

        self.assertEqual(response.status_code, 401)
        self.assertEqual(Mask.objects.filter(picture__dataset=self.dataset).count(), 0)

    def test_cannot_create_predictions_while_not_owner(self) -> None:
        request = self.client.post('images/bulk_predict/',
                                   QUERY_STRING=urlencode({'ids': f'{self.pictures[0].id},{self.pictures[1].id}'}))
        force_authenticate(request, user=User.objects.create_user(username='test2', password='test'))
        view = PictureViewSet.as_view({'post': 'bulk_predict'})
        response = view(request, dataset_pk=self.dataset.id)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(Mask.objects.filter(picture__dataset=self.dataset).count(), 0)


@override_settings(MEDIA_ROOT=MEDIA_ROOT)
class TestPictureBulkDestroyPredictionsEndpoint(APITestCase):
    def setUp(self) -> None:
        self.client = APIRequestFactory()
        os.makedirs(MEDIA_ROOT, exist_ok=True)

        self.user = User.objects.create_user(username='test', password='test')
        self.dataset = Dataset.objects.create(name='test', description='test', owner=self.user)

        image = PILImage.new('RGB', (100, 100), color='red')
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')

        self.pictures = Picture.objects.bulk_create([
            Picture(dataset=self.dataset, image=File(image_bytes, name='test.png')),
            Picture(dataset=self.dataset, image=File(image_bytes, name='test.png'))
        ])

        Mask.objects.bulk_create([
            Mask(picture=self.pictures[0], image=File(image_bytes, name='test_mask.png')),
            Mask(picture=self.pictures[1], image=File(image_bytes, name='test_mask.png'))
        ])

    def tearDown(self) -> None:
        shutil.rmtree(MEDIA_ROOT)

    def test_deletes_predictions(self) -> None:
        request = self.client.delete('images/bulk_destroy_predictions/',
                                     QUERY_STRING=urlencode({'ids': f'{self.pictures[0].id},{self.pictures[1].id}'}))
        force_authenticate(request, user=self.user)
        view = PictureViewSet.as_view({'delete': 'bulk_destroy_predictions'})
        response = view(request, dataset_pk=self.dataset.id)

        self.assertEqual(response.status_code, 204)
        self.assertEqual(Mask.objects.filter(picture__dataset=self.dataset).count(), 0)

    def test_cannot_delete_predictions_while_anonymous(self) -> None:
        request = self.client.delete('images/bulk_destroy_predictions/',
                                     QUERY_STRING=urlencode({'ids': f'{self.pictures[0].id},{self.pictures[1].id}'}))
        view = PictureViewSet.as_view({'delete': 'bulk_destroy_predictions'})
        response = view(request, dataset_pk=self.dataset.id)

        self.assertEqual(response.status_code, 401)
        self.assertEqual(Mask.objects.filter(picture__dataset=self.dataset).count(), 2)

    def test_cannot_delete_predictions_while_not_owner(self) -> None:
        request = self.client.delete('images/bulk_destroy_predictions/',
                                     QUERY_STRING=urlencode({'ids': f'{self.pictures[0].id},{self.pictures[1].id}'}))
        force_authenticate(request, user=User.objects.create_user(username='test2', password='test'))
        view = PictureViewSet.as_view({'delete': 'bulk_destroy_predictions'})
        response = view(request, dataset_pk=self.dataset.id)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(Mask.objects.filter(picture__dataset=self.dataset).count(), 2)
