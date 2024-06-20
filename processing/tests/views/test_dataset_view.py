from rest_framework.test import APIRequestFactory, force_authenticate, APITestCase
from django.contrib.auth.models import User

from processing.views import DatasetViewSet
from processing.models import Dataset


class TestDatasetListEndpoint(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='test', password='test')
        self.user2 = User.objects.create_user(username='test2', password='test2')
        self.datasets = Dataset.objects.bulk_create([
            Dataset(name='public', description='public', owner=self.user, public=True),
            Dataset(name='hidden', description='hidden', owner=self.user, public=False)
        ])
        self.client = APIRequestFactory()

    def test_route_exists(self) -> None:
        request = self.client.get('datasets/')
        view = DatasetViewSet.as_view({'get': 'list'})
        response = view(request)

        self.assertEqual(response.status_code, 200)

    def test_returns_json(self) -> None:
        request = self.client.get('datasets/')
        view = DatasetViewSet.as_view({'get': 'list'})
        response = view(request)

        self.assertEqual(response.accepted_media_type, 'application/json')

    def test_only_returns_public_datasets_without_authorization(self) -> None:
        request = self.client.get('datasets/')
        view = DatasetViewSet.as_view({'get': 'list'})
        response = view(request)

        self.assertEqual(len(response.data), 1)

    def test_does_not_show_unowned_datasets(self) -> None:
        request = self.client.get('datasets/')
        force_authenticate(request, user=self.user2)
        view = DatasetViewSet.as_view({'get': 'list'})
        response = view(request)

        self.assertEqual(len(response.data), 1)

    def test_shows_all_datasets_with_authorization(self) -> None:
        request = self.client.get('datasets/')
        force_authenticate(request, user=self.user)
        view = DatasetViewSet.as_view({'get': 'list'})
        response = view(request)

        self.assertEqual(len(response.data), 2)


class TestDatasetRetrieveEndpoint(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='test', password='test')
        self.dataset = Dataset.objects.create(name='test', description='test', owner=self.user, public=False)
        self.client = APIRequestFactory()

    def test_route_exists(self) -> None:
        request = self.client.get(f'datasets/{self.dataset.id}/')
        force_authenticate(request, user=self.user)
        view = DatasetViewSet.as_view({'get': 'retrieve'})
        response = view(request, pk=self.dataset.id)

        self.assertEqual(response.status_code, 200)

    def test_returns_json(self) -> None:
        request = self.client.get(f'datasets/{self.dataset.id}/')
        force_authenticate(request, user=self.user)
        view = DatasetViewSet.as_view({'get': 'retrieve'})
        response = view(request, pk=self.dataset.id)

        self.assertEqual(response.accepted_media_type, 'application/json')

    def test_returns_correct_dataset(self) -> None:
        request = self.client.get(f'datasets/{self.dataset.id}/')
        force_authenticate(request, user=self.user)
        view = DatasetViewSet.as_view({'get': 'retrieve'})
        response = view(request, pk=self.dataset.id)

        expected_data = {
            'id': self.dataset.id,
            'name': self.dataset.name,
            'description': self.dataset.description,
            'owner': self.user.id,
            'public': self.dataset.public
        }
        self.assertEqual(response.status_code, 200)
        self.assertDictContainsSubset(expected_data, response.data)

    def test_cannot_retrieve_private_when_anonymous(self) -> None:
        request = self.client.get(f'datasets/{self.dataset.id}/')
        view = DatasetViewSet.as_view({'get': 'retrieve'})
        response = view(request, pk=self.dataset.id)

        self.assertEqual(response.status_code, 404)

    def test_cannot_retrieve_private_when_not_owner(self) -> None:
        request = self.client.get(f'datasets/{self.dataset.id}/')
        force_authenticate(request, user=User.objects.create_user(username='test2', password='test2'))
        view = DatasetViewSet.as_view({'get': 'retrieve'})
        response = view(request, pk=self.dataset.id)

        self.assertEqual(response.status_code, 404)


class TestDatasetCreateEndpoint(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='test', password='test')
        self.client = APIRequestFactory()

    def test_route_exists(self) -> None:
        data = {'name': 'test', 'description': 'test'}
        request = self.client.post('datasets/', data)
        force_authenticate(request, user=self.user)
        view = DatasetViewSet.as_view({'post': 'create'})
        response = view(request)

        self.assertEqual(response.status_code, 201)

    def test_returns_json(self) -> None:
        request = self.client.post('datasets/')
        force_authenticate(request, user=self.user)
        view = DatasetViewSet.as_view({'post': 'create'})
        response = view(request)

        self.assertEqual(response.accepted_media_type, 'application/json')

    def test_creates_dataset(self) -> None:
        data = {'name': 'test', 'description': 'test'}
        request = self.client.post('datasets/', data)
        force_authenticate(request, user=self.user)
        view = DatasetViewSet.as_view({'post': 'create'})
        response = view(request)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(Dataset.objects.filter(pk=response.data['id']).count(), 1)

    def test_cannot_create_dataset_while_unauthorized(self) -> None:
        request = self.client.post('datasets/')
        view = DatasetViewSet.as_view({'post': 'create'})
        response = view(request)

        self.assertEqual(response.status_code, 401)


class TestDatasetUpdateEndpoint(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='test', password='test')
        self.dataset = Dataset.objects.create(name='test', description='test', owner=self.user)
        self.client = APIRequestFactory()
        self.data = {'name': 'updated', 'description': 'updated'}

    def test_route_exists(self) -> None:
        request = self.client.patch(f'datasets/{self.dataset.id}/', self.data)
        force_authenticate(request, user=self.user)
        view = DatasetViewSet.as_view({'patch': 'update'})
        response = view(request, pk=self.dataset.id)

        self.assertEqual(response.status_code, 200)

    def test_returns_json(self) -> None:
        request = self.client.patch(f'datasets/{self.dataset.id}/', self.data)
        force_authenticate(request, user=self.user)
        view = DatasetViewSet.as_view({'patch': 'update'})
        response = view(request, pk=self.dataset.id)

        self.assertEqual(response.accepted_media_type, 'application/json')

    def test_updates_dataset(self) -> None:
        request = self.client.patch(f'datasets/{self.dataset.id}/', self.data)
        force_authenticate(request, user=self.user)
        view = DatasetViewSet.as_view({'patch': 'update'})
        response = view(request, pk=self.dataset.id)

        self.dataset.refresh_from_db()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.dataset.name, 'updated')
        self.assertEqual(self.dataset.description, 'updated')

    def test_cannot_update_dataset_while_unauthorized(self) -> None:
        request = self.client.patch(f'datasets/{self.dataset.id}/', self.data)
        view = DatasetViewSet.as_view({'patch': 'update'})
        response = view(request, pk=self.dataset.id)

        self.assertEqual(response.status_code, 401)

    def test_cannot_update_dataset_while_not_owner(self) -> None:
        request = self.client.patch(f'datasets/{self.dataset.id}/', self.data)
        force_authenticate(request, user=User.objects.create_user(username='test2', password='test2'))
        view = DatasetViewSet.as_view({'patch': 'update'})
        response = view(request, pk=self.dataset.id)

        self.assertEqual(response.status_code, 404)


class TestDatasetDeleteEndpoint(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='test', password='test')
        self.dataset = Dataset.objects.create(name='test', description='test', owner=self.user)
        self.client = APIRequestFactory()

    def test_route_exists(self) -> None:
        request = self.client.delete(f'datasets/{self.dataset.id}/')
        force_authenticate(request, user=self.user)
        view = DatasetViewSet.as_view({'delete': 'destroy'})
        response = view(request, pk=self.dataset.id)

        self.assertEqual(response.status_code, 204)

    def test_deletes_dataset(self) -> None:
        request = self.client.delete(f'datasets/{self.dataset.id}/')
        force_authenticate(request, user=self.user)
        view = DatasetViewSet.as_view({'delete': 'destroy'})
        response = view(request, pk=self.dataset.id)

        self.assertEqual(response.status_code, 204)
        with self.assertRaises(Dataset.DoesNotExist):
            Dataset.objects.get(id=self.dataset.id)

    def test_cannot_delete_dataset_while_unauthorized(self) -> None:
        request = self.client.delete(f'datasets/{self.dataset.id}/')
        view = DatasetViewSet.as_view({'delete': 'destroy'})
        response = view(request, pk=self.dataset.id)

        self.assertEqual(response.status_code, 401)

    def test_cannot_delete_dataset_while_not_owner(self) -> None:
        request = self.client.delete(f'datasets/{self.dataset.id}/')
        force_authenticate(request, user=User.objects.create_user(username='test2', password='test2'))
        view = DatasetViewSet.as_view({'delete': 'destroy'})
        response = view(request, pk=self.dataset.id)

        self.assertEqual(response.status_code, 404)
