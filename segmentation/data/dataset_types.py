from enum import Enum

from torch.utils.data import Dataset

from segmentation.data.labelme import LabelmeDataset
from segmentation.data.prmi import PRMIDataset


class DatasetType(Enum):
    LABELME = 'labelme'
    PRMI = 'prmi'

    def __str__(self) -> str:
        return self.value

    def get_dataset(self, *args) -> Dataset:
        match self.value:
            case 'labelme':
                return LabelmeDataset(*args)
            case 'prmi':
                return PRMIDataset(*args)
            case _:
                raise ValueError(f'Invalid dataset type: {self.value}')
