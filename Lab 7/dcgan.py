import torch.nn as nn
import torch


# DCGAN is designed for image size 64

def weights_init(model: nn.Module) -> None:
    """
    Initialize weights in convolution and batch norm layers
    :param model: Model
    :return: None
    """
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


class DCGenerator(nn.Module):
    def __init__(self, noise_size: int, label_size: int):
        super(DCGenerator, self).__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=noise_size + label_size,
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
                               out_channels=3,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generator forwarding
        :param x: Batched data
        :return: Batched image
        """
        return self.net(x)


class DCDiscriminator(nn.Module):
    def __init__(self, num_classes: int, image_size: int):
        super(DCDiscriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=64,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
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
                      bias=False)
        )

        self.label_to_condition = nn.Sequential(
            nn.ConvTranspose2d(in_channels=num_classes,
                               out_channels=16,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=16,
                               out_channels=4,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=4),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=4,
                               out_channels=1,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(True),
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=16 * 16,
                      out_features=32 * 32,
                      bias=False),
            nn.ReLU(True),

            nn.Linear(in_features=32 * 32,
                      out_features=image_size * image_size,
                      bias=False),
            nn.Tanh()
        )
        self.image_size = image_size

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Discriminator forwarding
        :param x: Batched data
        :param label: Batched labels
        :return: Discrimination results
        """
        batch_size, num_classes = label.size()
        label = label.view(batch_size, num_classes, 1, 1)
        condition = self.label_to_condition(label).view(batch_size, 1, -1)
        condition = self.linear(condition).view(-1, 1, self.image_size, self.image_size)
        inputs = torch.cat([x, condition], 1)
        return self.net(inputs).view(-1, 1)
