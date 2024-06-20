import tempfile
from django.test import override_settings, TestCase
from django.contrib.auth.models import User
from processing.models import Dataset, Picture, Mask

MEDIA_ROOT = tempfile.mkdtemp()


@override_settings(MEDIA_ROOT=MEDIA_ROOT)
class TestMaskModel(TestCase):
    def setUp(self):
        self.picture = Picture(dataset=Dataset(owner=User()), image='images/test.jpg')
        self.mask = Mask(picture=self.picture, image='images/test_mask.jpg')

    def test_filename_property(self) -> None:
        self.assertEqual(self.mask.filename, 'test_mask.jpg')

    def test_filename_noext_property(self) -> None:
        self.assertEqual(self.mask.filename_noext, 'test_mask')

    def test_picture_inherits_owner_property(self) -> None:
        self.assertEqual(self.mask.owner, self.picture.dataset.owner)

    def test_picture_inherits_public_property(self) -> None:
        self.assertFalse(self.mask.public)
