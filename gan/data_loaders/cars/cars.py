from functools import partial
from torchvision.datasets import StanfordCars
import torchvision
import torchvision.transforms as transforms

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import torch

from random import random


def exists(val):
    return val is not None


class identity(object):
    def __call__(self, tensor):
        return tensor


def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image


def resize_to_square(image, size):
    return torchvision.transforms.functional.resize(image, (size, size))


class expand_greyscale(object):
    def __init__(self, transparent):
        self.transparent = transparent

    def __call__(self, tensor):
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

        if channels == num_target_channels:
            return tensor

        alpha = None
        if channels == 1:
            color = tensor.expand(3, -1, -1)
        elif channels == 2:
            color = tensor[:1].expand(3, -1, -1)
            alpha = tensor[1:]
        else:
            raise Exception(f"image with invalid number of channels given {channels}")

        if not exists(alpha) and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))


class RandomApply(torch.nn.Module):
    def __init__(self, prob, fn, fn_else=lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob

    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)


class CarsLightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_size: int = 128,
        aug_prob: float = 0.5,
        in_channels: int = 3,
        data_dir: str = "./data",
        batch_size: int = 64,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = os.cpu_count()
        self.in_channels = in_channels

        if in_channels == 4:
            pillow_mode = "RGBA"
            expand_fn = expand_greyscale(True)
        elif in_channels == 1:
            pillow_mode = "L"
            expand_fn = identity()
        else:
            num_channels = 3
            pillow_mode = "RGB"
            expand_fn = expand_greyscale(False)

        convert_image_fn = partial(convert_image_to, pillow_mode)

        self.transform = transforms.Compose(
            [
                transforms.Lambda(convert_image_fn),
                transforms.Lambda(partial(resize_to_minimum_size, image_size)),
                transforms.Resize(image_size),
                RandomApply(
                    aug_prob,
                    transforms.RandomResizedCrop(
                        image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)
                    ),
                    transforms.CenterCrop(image_size),
                ),
                transforms.ToTensor(),
                transforms.Lambda(expand_fn),
            ]
        )

        self.dataset = StanfordCars(
            self.data_dir, transform=self.transform, download=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
