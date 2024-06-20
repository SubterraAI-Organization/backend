import tempfile
from django.test import override_settings, TestCase
from django.contrib.auth.models import User
from processing.models import Picture, Dataset

MEDIA_ROOT = tempfile.mkdtemp()


@override_settings(MEDIA_ROOT=MEDIA_ROOT)
class TestPictureModel(TestCase):
    def setUp(self):
        self.picture = Picture(dataset=Dataset(owner=User()), image='images/test.jpg')

    def test_filename_property(self) -> None:
        self.assertEqual(self.picture.filename, 'test.jpg')

    def test_filename_noext_property(self) -> None:
        self.assertEqual(self.picture.filename_noext, 'test')

    def test_picture_inherits_owner_property(self) -> None:
        self.assertEqual(self.picture.owner, self.picture.dataset.owner)

    def test_picture_inherits_public_property(self) -> None:
        self.assertFalse(self.picture.public)
