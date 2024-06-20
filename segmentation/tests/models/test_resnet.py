from unittest import TestCase
import torch

from segmentation.models.resnet import ResNet


class ResNet18Test(TestCase):
    def test_outputs_correct_shape(self):
        model = ResNet(3, 1, 18)
        image = torch.rand(1, 3, 256, 256)
        output = model(image)
        self.assertEqual(output.shape, (1, 1, 256, 256))


class ResNet34Test(TestCase):
    def test_outputs_correct_shape(self):
        model = ResNet(3, 1, 34)
        image = torch.rand(1, 3, 256, 256)
        output = model(image)
        self.assertEqual(output.shape, (1, 1, 256, 256))


class ResNet50Test(TestCase):
    def test_outputs_correct_shape(self):
        model = ResNet(3, 1, 50)
        image = torch.rand(1, 3, 256, 256)
        output = model(image)
        self.assertEqual(output.shape, (1, 1, 256, 256))


class ResNet101Test(TestCase):
    def test_outputs_correct_shape(self):
        model = ResNet(3, 1, 101)
        image = torch.rand(1, 3, 256, 256)
        output = model(image)
        self.assertEqual(output.shape, (1, 1, 256, 256))


class ResNet152Test(TestCase):
    def test_outputs_correct_shape(self):
        model = ResNet(3, 1, 152)
        image = torch.rand(1, 3, 256, 256)
        output = model(image)
        self.assertEqual(output.shape, (1, 1, 256, 256))
