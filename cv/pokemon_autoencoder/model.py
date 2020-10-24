import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Interpolate, show_image

# Formula to calculate output dimensions of a convolutional or pooling layer:
#   out_width = floor((in_width - kernel_width + 2 * padding) / stride) + 1
#   out_height = floor((in_height - kernel_height + 2 * padding) / stride) + 1


class PokemonAutoencoder(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # (3, 240, 330)
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=32,
                kernel_size=7,
                padding=3,
            ),  # 32, 240, 330
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=4,
                stride=4,
            ),  # 32, 60, 82
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),  # 64, 60, 82
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=6,
                stride=6,
            ),  # 64, 10, 13
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),  # 128, 10, 13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=(13, 10),
            ),  # 128, 1, 1
        )

        self.decoder = nn.Sequential(
            Interpolate(scale_factor=(13, 10)),  # 128, 10, 13
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),  # 64, 10, 13
            nn.ReLU(inplace=True),
            Interpolate(size=(82, 60)),  # 64, 60, 82
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),  # 32, 60, 82
            nn.ReLU(inplace=True),
            Interpolate(size=(330, 240)),  # 32, 240, 330
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=channels,
                kernel_size=7,
                padding=3,
            ),  # 3, 240, 330
            nn.Sigmoid(),
        )

    def __call__(self, tensor):
        x = self.encoder(tensor)
        x = self.decoder(x)
        return x

    def fit(self, train_dl, val_dl, epochs, optimizer, loss_func):
        for epoch in range(epochs):
            self.train()
            for x_batch, y_batch in train_dl:
                pred = self(x_batch)
                loss = loss_func(pred, y_batch)

                print('\rEpoch:', epoch, '- Training loss:', loss.item(), end='')

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            avg_val_loss = self.test(val_dl, loss_func)
            print('\nEpoch:', epoch, '- Validation loss:', avg_val_loss.item())

    def test(self, test_dl, loss_func):
        """Test function to be used on foreign datasets."""
        self.eval()
        counter = 0
        loss = 0
        with torch.no_grad():
            for x_batch, y_batch in test_dl:
                counter += 1
                pred = self(x_batch)
                loss += loss_func(pred, y_batch)
        return loss / counter


class ResizedPokemonAutoencoder(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # (3, 75, 100)
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=32,
                kernel_size=7,
                padding=3,
            ),  # 32, 75, 100
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=4,
                stride=4,
            ),  # 32, 18, 25
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),  # 64, 18, 25
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),  # 32, 18, 25
            nn.ReLU(inplace=True),
            Interpolate(size=(100, 75)),  # 32, 75, 100
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=channels,
                kernel_size=3,
                padding=1,
            ),  # 3, 75, 100
            nn.Sigmoid(),
        )

    def __call__(self, tensor):
        x = self.encoder(tensor)
        x = self.decoder(x)
        return x

    def fit(self, train_dl, val_dl, epochs, optimizer, loss_func):
        for epoch in range(epochs):
            self.train()
            for x_batch, y_batch in train_dl:
                pred = self(x_batch)
                loss = loss_func(pred, y_batch)

                print('\rEpoch:', epoch, '- Training loss:', loss.item(), end='')

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            avg_val_loss = self.test(val_dl, loss_func)
            print('\nEpoch:', epoch, '- Validation loss:', avg_val_loss.item())

    def test(self, test_dl, loss_func):
        """Test function to be used on foreign datasets."""
        self.eval()
        counter = 0
        loss = 0
        with torch.no_grad():
            for x_batch, y_batch in test_dl:
                counter += 1
                pred = self(x_batch)
                loss += loss_func(pred, y_batch)
        return loss / counter


if __name__ == "__main__":
    from setup_datasets import get_datasets

    model = ResizedPokemonAutoencoder(3)
    train_dl, val_dl, test_dl = get_datasets()
    epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.MSELoss()

    model.fit(train_dl, val_dl, epochs, optimizer, loss_func)
    test_loss = model.test(test_dl, loss_func)

    print('Test loss:', test_loss.item())

    torch.save(model.state_dict(), 'models/model.pt')