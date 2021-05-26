import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, noise_size: int):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=noise_size,
                               out_channels=512,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=512,
                               out_channels=256,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=64,
                               out_channels=32,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=32,
                               out_channels=16,
                               kernel_size=5,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=16,
                               out_channels=3,
                               kernel_size=1,
                               stride=1,
                               padding=1,
                               bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, image_size: int):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=32,
                      kernel_size=4,
                      stride=2,
                      padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=512,
                      out_channels=1,
                      kernel_size=4,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.Sigmoid()
        )

        self.label_to_condition = nn.Linear(24, image_size * image_size, bias=True)
        self.image_size = image_size

    def forward(self, x, label):
        condition = self.label_to_condition(label).view(-1, 1, self.image_size, self.image_size)
        x = torch.cat([x, condition], 1)
        return self.net(x).view(-1, 1)
