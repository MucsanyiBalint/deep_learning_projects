import math
import os
import random
import requests
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose
from utils import print_bar, RandomCut, PokemonDataset


def _get_images():
    os.mkdir('images')
    data = requests.get('https://raw.githubusercontent.com/PokemonTCG/'
                        'pokemon-tcg-data/master/json/cards/Base.json')
    data_json = data.json()

    len_data_json = len(data_json)

    for i, pokemon in enumerate(data_json):
        print_bar('Downloading images', (i + 1) / len_data_json)
        image = requests.get(pokemon['imageUrl']).content
        with open(f"images/image_{pokemon['name'].lower()}.png", 'wb') as f:
            f.write(image)
    print(f'\r\nDownloaded {len_data_json} images.')


def _distribute_images(split=(0.7, 0.2, 0.1)):
    dir_names = ['train_dataset', 'val_dataset', 'test_dataset']

    for dir_name in dir_names:
        os.mkdir(dir_name)

    file_list = os.listdir('images')
    random.shuffle(file_list)
    len_file_list = len(file_list)

    print('Distributing images into train/val/test split...')

    low = 0
    high = 0
    for pct, dir_name in zip(split, dir_names):
        high = high + math.ceil(pct * len_file_list)
        for file_name in file_list[low:high]:
            os.rename(f'images/{file_name}', f'{dir_name}/{file_name}')
        low = high
        print(f'Created {dir_name}.')

    print('Done distributing images.')
    os.rmdir('images')
    os.mkdir('models')

def get_datasets():
    """Returns train, validation and test datasets.
    
    Usage:
        First run get_dataset.py, then import the get_datasets()
        function.
    """
    normal_transform = Compose([
        transforms.Resize((100, 75)),
        transforms.ToTensor(),
    ])

    patchy_transform = Compose([
        transforms.Resize((100, 75)),
        transforms.ToTensor(),
        RandomCut((5, 10), (10, 15), 5),  # h: 330, w: 240
    ])

    train_dataset = PokemonDataset(
        normal_transform,
        patchy_transform,
        'train_dataset',
    )
    train_dl = DataLoader(train_dataset, 32, shuffle=True)

    val_dataset = PokemonDataset(
        normal_transform,
        patchy_transform,
        'val_dataset',
    )
    val_dl = DataLoader(val_dataset, 16)

    test_dataset = PokemonDataset(
        normal_transform,
        patchy_transform,
        'test_dataset',
    )
    test_dl = DataLoader(test_dataset, 4)

    return train_dl, val_dl, test_dl


if __name__ == "__main__":
    _get_images()
    _distribute_images()