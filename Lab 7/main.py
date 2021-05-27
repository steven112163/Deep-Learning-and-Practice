from model import Generator, Discriminator
from data_loader import ICLEVRLoader
from evaluator import EvaluationModel
from argument_parser import parse_arguments
from visualizer import plot_losses, plot_accuracies
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image, make_grid
from torch import device, cuda
from typing import Tuple
import torch.optim as optim
import torch.nn as nn
import sys
import os
import torch


def info_log(log: str) -> None:
    """
    Print information log
    :param log: log to be displayed
    :return: None
    """
    global verbosity
    if verbosity:
        print(f'[\033[96mINFO\033[00m] {log}')
        sys.stdout.flush()


def debug_log(log: str) -> None:
    """
    Print debug log
    :param log: log to be displayed
    :return: None
    """
    global verbosity
    if verbosity > 1:
        print(f'[\033[93mDEBUG\033[00m] {log}')
        sys.stdout.flush()


def train(data_loader: DataLoader,
          generator: Generator,
          discriminator: Discriminator,
          optimizer_g: optim,
          optimizer_d: optim,
          image_size: int,
          num_classes: int,
          epoch: int,
          num_of_epochs: int,
          training_device: device) -> Tuple[float, float]:
    """
    Train the model
    :param data_loader: train data loader
    :param generator: generator
    :param discriminator: discriminator
    :param optimizer_g: optimizer for generator
    :param optimizer_d: optimizer for discriminator
    :param image_size: image size (noise size)
    :param num_classes: number of classes (object IDs)
    :param epoch: current epoch
    :param num_of_epochs: number of total epochs
    :param training_device: training_device
    :return: total generator loss and total discriminator loss
    """
    generator.train()
    discriminator.train()
    total_g_loss = 0.0
    total_d_loss = 0.0
    for batch_idx, batch_data in enumerate(data_loader):
        images, real_labels = batch_data
        real_labels = real_labels.to(training_device).type(torch.float)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        discriminator.zero_grad()

        # Train with all-real batch
        # Format batch
        images = images.to(training_device)
        batch_size = images.size(0)
        labels = torch.full((batch_size, 1), 1.0, dtype=torch.float, device=training_device)

        # Forward pass real batch through discriminator
        outputs = discriminator.forward(images, real_labels)
        d_x = outputs.mean().item()

        # Calculate loss on all-real batch
        loss_d_real = nn.BCELoss()(outputs, labels)
        loss_d_real.backward()

        # Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.cat([
            torch.randn(batch_size, image_size - num_classes),
            real_labels.cpu()
        ], 1).view(-1, image_size, 1, 1).to(training_device)
        labels = torch.full((batch_size, 1), 0.0, dtype=torch.float, device=training_device)

        # Generate fake image batch with generator
        fake_outputs = generator.forward(noise)

        # Forward pass fake batch through discriminator
        outputs = discriminator.forward(fake_outputs.detach(), real_labels)
        d_g_z1 = outputs.mean().item()

        # Calculate loss on fake batch
        loss_d_fake = nn.BCELoss()(outputs, labels)
        loss_d_fake.backward()
        optimizer_d.step()

        # Compute loss of discriminator as sum over the fake and the real batches
        loss_d = loss_d_real + loss_d_fake
        total_d_loss += loss_d.item()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        generator.zero_grad()

        # Since we just updated discriminator, perform another forward pass of all-fake batch through discriminator
        outputs = discriminator.forward(fake_outputs, real_labels)
        d_g_z2 = outputs.mean().item()
        labels = torch.full((batch_size, 1), 1.0, dtype=torch.float, device=training_device)

        # Calculate generator's loss based on this output
        loss_g = nn.BCELoss()(outputs, labels)
        total_g_loss += loss_g.item()

        # Calculate gradients for generator and update
        loss_g.backward()
        optimizer_g.step()

        if batch_idx % 50 == 0:
            output_string = f'[{epoch}/{num_of_epochs}][{batch_idx}/{len(data_loader)}]\t' + \
                            f'Loss_D: {loss_d.item():.4f}\tLoss_G: {loss_g.item():.4f}\t' + \
                            f'D(x): {d_x:.4f}\tD(G(z)): {d_g_z1:.4f} / {d_g_z2:.4f}'
            debug_log(output_string)

    return total_g_loss, total_d_loss


def test(data_loader: DataLoader,
         generator: Generator,
         image_size: int,
         num_classes: int,
         epoch: int,
         num_of_epochs: int,
         evaluator: EvaluationModel,
         training_device: device) -> Tuple[torch.Tensor, float]:
    """
    Test the model
    :param data_loader: test data loader
    :param generator: generator
    :param image_size: image size (noise size)
    :param num_classes: number of classes (object IDs)
    :param epoch: current epoch
    :param num_of_epochs: number of total epochs
    :param training_device: training_device
    :return: generated images and total accuracy of all batches
    """
    generator.eval()
    total_accuracy = 0.0
    norm_image = torch.randn(0, 3, 64, 64)
    for batch_idx, batch_data in enumerate(data_loader):
        labels = batch_data
        batch_size = len(labels)

        # Generate batch of latent vectors
        noise = torch.cat([
            torch.randn(batch_size, image_size - num_classes, 1, 1, device=training_device),
            labels
        ], 1).view(-1, image_size, 1, 1).to(training_device)

        # Generate fake image batch with generator
        fake_outputs = generator.forward(noise)

        # Compute accuracy
        acc = evaluator.eval(fake_outputs, labels)
        total_accuracy += acc

        # Generate images from fake images
        transformation = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                                  std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
                                             transforms.Normalize(mean=[-0.5, -0.5, -0.5],
                                                                  std=[1., 1., 1.]),
                                             ])
        for fake_image in fake_outputs:
            n_image = transformation(fake_image.cpu().detach())
            norm_image = torch.cat([norm_image, n_image.view(1, 3, 64, 64)], 0)

        debug_log(f'[{epoch}/{num_of_epochs}][{batch_idx}/{len(data_loader)}]\tAccuracy: {acc}')

    return norm_image, total_accuracy


def main() -> None:
    """
    Main function
    :return: None
    """
    # Get training device
    training_device = device('cuda' if cuda.is_available() else 'cpu')

    # Parse arguments
    args = parse_arguments()
    batch_size = args.batch_size
    image_size = args.image_size
    learning_rate_d = args.learning_rate_discriminator
    learning_rate_g = args.learning_rate_generator
    epochs = args.epochs
    global verbosity
    verbosity = args.verbosity
    info_log(f'Batch size: {batch_size}')
    info_log(f'Image size: {image_size}')
    info_log(f'Learning rate of discriminator: {learning_rate_d}')
    info_log(f'Learning rate of generator: {learning_rate_g}')
    info_log(f'Number of epochs: {epochs}')
    info_log(f'Training device: {training_device}')

    # Read data
    info_log('Read data ...')
    transformation = transforms.Compose([transforms.RandomCrop(240),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.Resize((image_size, image_size)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = ICLEVRLoader(root_folder='data/task_1/', trans=transformation, mode='train')
    test_dataset = ICLEVRLoader(root_folder='data/task_1/', mode='test')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    num_classes = train_dataset.num_classes

    # Setup models
    info_log('Setup models ...')
    generator = Generator(noise_size=image_size).to(training_device)
    discriminator = Discriminator(image_size=image_size).to(training_device)
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate_g, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate_d, betas=(0.5, 0.999))

    # Setup average losses/accuracies container
    generator_losses = [0.0 for _ in range(epochs)]
    discriminator_losses = [0.0 for _ in range(epochs)]
    accuracies = [0.0 for _ in range(epochs)]

    # Setup evaluator
    evaluator = EvaluationModel(training_device=training_device)

    # Create directories
    if not os.path.exists('./model'):
        os.mkdir('./model')
    if not os.path.exists('./test_figure'):
        os.mkdir('./test_figure')
    if not os.path.exists('./figure'):
        os.mkdir('./figure')

    # Start training
    info_log('Start training')
    for epoch in range(epochs):
        # Train
        total_g_loss, total_d_loss = train(data_loader=train_loader,
                                           generator=generator,
                                           discriminator=discriminator,
                                           optimizer_g=optimizer_g,
                                           optimizer_d=optimizer_d,
                                           image_size=image_size,
                                           num_classes=num_classes,
                                           epoch=epoch,
                                           num_of_epochs=epochs,
                                           training_device=training_device)
        generator_losses[epoch] = total_g_loss / len(train_dataset)
        discriminator_losses[epoch] = total_d_loss / len(train_dataset)
        print(f'[{epoch}/{epochs}]\tAverage generator loss: {generator_losses[epoch]}')
        print(f'[{epoch}/{epochs}]\tAverage discriminator loss: {discriminator_losses[epoch]}')

        # Test
        generated_image, total_accuracy = test(data_loader=test_loader,
                                               generator=generator,
                                               image_size=image_size,
                                               num_classes=num_classes,
                                               epoch=epoch,
                                               num_of_epochs=epochs,
                                               evaluator=evaluator,
                                               training_device=training_device)
        accuracies[epoch] = total_accuracy / len(test_loader)
        save_image(make_grid(generated_image, nrow=8), f'test_figure/{epoch}.jpg')
        print(f'[{epoch}/{epochs}]\tAverage accuracy: {accuracies[epoch]:.2f}')

        if epoch % 10 == 0:
            torch.save(generator, f'model/{epoch}_{accuracies[epoch]:.4f}_g.pt')
            torch.save(discriminator, f'model/{epoch}_{accuracies[epoch]:.4f}_d.pt')

    # Plot losses and accuracies
    info_log('Plot losses and accuracies ...')
    plot_losses(generator_losses=generator_losses, discriminator_losses=discriminator_losses)
    plot_accuracies(accuracies=accuracies)


if __name__ == '__main__':
    verbosity = None
    main()
