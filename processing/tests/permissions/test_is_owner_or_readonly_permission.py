from django.http import HttpRequest
from django.test import TestCase
from processing.permissions import IsOwnerOrReadOnly
from django.contrib.auth.models import User
from processing.models import Dataset, Picture, Mask


class TestIsOwnerOrReadOnlyPermission(TestCase):
    def setUp(self):
        self.permission = IsOwnerOrReadOnly()
        self.view = None
        self.user = User()
        self.dataset = Dataset(owner=self.user)
        self.picture = Picture(dataset=self.dataset)
        self.mask = Mask(picture=self.picture)
        self.request = HttpRequest()

    def test_unauthenticated_users_can_read_while_public(self) -> None:
        self.request.method = 'GET'
        self.request.user = None
        self.dataset.public = True

        self.assertTrue(self.permission.has_object_permission(self.request, self.view, self.dataset))
        self.assertTrue(self.permission.has_object_permission(self.request, self.view, self.picture))
        self.assertTrue(self.permission.has_object_permission(self.request, self.view, self.mask))

    def test_unauthenticated_users_cannot_write_while_public(self) -> None:
        self.request.method = 'POST'
        self.request.user = None
        self.dataset.public = True

        self.assertFalse(self.permission.has_object_permission(self.request, self.view, self.dataset))
        self.assertFalse(self.permission.has_object_permission(self.request, self.view, self.picture))
        self.assertFalse(self.permission.has_object_permission(self.request, self.view, self.mask))

    def test_owner_can_read_while_private(self) -> None:
        self.request.method = 'GET'
        self.request.user = self.user
        self.dataset.public = False

        self.assertTrue(self.permission.has_object_permission(self.request, self.view, self.dataset))
        self.assertTrue(self.permission.has_object_permission(self.request, self.view, self.picture))
        self.assertTrue(self.permission.has_object_permission(self.request, self.view, self.mask))

    def test_owner_can_write_while_private(self) -> None:
        self.request.method = 'POST'
        self.request.user = self.user
        self.dataset.public = False

        self.assertTrue(self.permission.has_object_permission(self.request, self.view, self.dataset))
        self.assertTrue(self.permission.has_object_permission(self.request, self.view, self.picture))
        self.assertTrue(self.permission.has_object_permission(self.request, self.view, self.mask))
