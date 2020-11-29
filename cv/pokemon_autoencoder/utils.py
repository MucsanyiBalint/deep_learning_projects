from PIL import Image
from torch.utils.data import Dataset
import math
import os
from random import randint
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


def print_bar(title, pct, width=50):
    full_num = math.floor(pct * width)
    empty_num = width - full_num
    print(
        f"\r{title} - [{'-' * full_num}{' ' * empty_num}] - {round(100 * pct)}%",
        end='')


def show_image(tensor):
    image = transforms.ToPILImage()(tensor).resize((240, 330))
    image.show()


class RandomCut:
    """Cut out cut_num parts of the image. The parts have width and height
    chosen uniformly at random from the range provided by the min_cut_size and
    max_cut_size sequences.

    Args:
        min_cut_size (sequence): Sequence of size 2 providing the minimum width
            and height of the parts needed to be cut out, in this order.
        max_cut_size (sequence): Sequence of size 2 providing the maximum width
            and height of the parts needed to be cut out, in this order.
        cut_num (int, optional): Integer providing the number of cut out parts.
    """
    def __init__(self, min_cut_size, max_cut_size, cut_num=1):
        if len(min_cut_size) != 2 or len(max_cut_size) != 2:
            raise ValueError('Invalid cut sizes provided.')
        self.min_cut_size = min_cut_size
        self.max_cut_size = max_cut_size
        self.cut_num = cut_num

    def __call__(self, tensor):
        tensor_copy = tensor.clone().detach()
        for _ in range(self.cut_num):
            width = randint(self.min_cut_size[0], self.max_cut_size[0])
            height = randint(self.min_cut_size[1], self.max_cut_size[1])
            width_start = randint(0, tensor.size()[-1] - width)
            height_start = randint(0, tensor.size()[-2] - height)
            tensor_copy[..., height_start:height_start + height,
                        width_start:width_start +
                        width, ] = 0  # blacks out window
        return tensor_copy

    def __repr__(self):
        return (
            f'{self.__class__}({self.min_cut_size=}, {self.max_cut_size=}, '
            f'{self.cut_num=})')


class PokemonDataset(Dataset):
    """Implement a dataset of (original_picture, patchy_picture) pairs
    of Pokemon cards.
    """
    def __init__(self, normal_transform, patchy_transform, directory):
        self.normal_transform = normal_transform
        self.patchy_transform = patchy_transform
        self.directory = directory
        self.files = os.listdir(directory)
        self.len_files = len(self.files)

    def __len__(self):
        return self.len_files

    def __getitem__(self, index):
        """Return item at index.

        Doesn't check the validity of index, as DataLoader takes chare of that.
        """
        file_name = self.files[index]
        image = Image.open(f'{self.directory}/{file_name}')
        normal_tensor = self.normal_transform(image)
        patchy_tensor = self.patchy_transform(image)

        return patchy_tensor, normal_tensor

    # Iterable dataset problem:
    # def __iter__(self):
    #     return self

    # def __next__(self):
    #     if self.index == self.len_files:
    #         # self.index = 0  # We want to iterate on it multiple times! (Common mistake)
    #         raise StopIteration()
    #     file_name = self.files[self.index]
    #     self.index += 1
    #     image = Image.open(f'{self.directory}/{file_name}')
    #     normal_tensor = self.normal_transform(image)
    #     patchy_tensor = self.patchy_transform(image)

    #     return patchy_tensor, normal_tensor


class Interpolate(nn.Module):
    """Wrapper class around F.interpolate to be used in a Sequential Module."""
    def __init__(self, size=None, scale_factor=None):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = 'nearest'

    def forward(self, x):
        x = self.interp(x,
                        size=self.size,
                        scale_factor=self.scale_factor,
                        mode=self.mode)
        return x
