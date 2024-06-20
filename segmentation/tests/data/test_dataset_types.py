from unittest import TestCase

from segmentation.data.dataset_types import DatasetType
from segmentation.data.labelme import LabelmeDataset
from segmentation.data.prmi import PRMIDataset


class TestDatasetType(TestCase):
    def test_string_maps_to_correct_dataset(self):
        self.assertEqual(str(DatasetType.PRMI), "prmi")
        self.assertEqual(str(DatasetType.LABELME), "labelme")

    def test_get_dataset_returns_correct_dataset_type(self):
        labelme_dataset = DatasetType.LABELME.get_dataset('path')
        self.assertIsInstance(labelme_dataset, LabelmeDataset)

        prmi_dataset = DatasetType.PRMI.get_dataset('path', 'path')
        self.assertIsInstance(prmi_dataset, PRMIDataset)
