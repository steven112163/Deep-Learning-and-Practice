from task_1_model import Generator, Discriminator
from task_1_dataset import ICLEVRLoader
# from task_2_model import Glow
from task_2_test_model import CondGlowModel
from task_2_test_2_model import Glow, NLLLoss
from evaluator import EvaluationModel
from argument_parser import parse_arguments
from visualizer import plot_losses, plot_accuracies
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image, make_grid
from torch import device, cuda
from typing import Tuple
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
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


def train_and_evaluate_dcgan(train_dataset: ICLEVRLoader,
                             train_loader: DataLoader,
                             test_loader: DataLoader,
                             evaluator: EvaluationModel,
                             learning_rate_g: float,
                             learning_rate_d: float,
                             image_size: int,
                             num_classes: int,
                             epochs: int,
                             training_device: device) -> None:
    """
    Train and test DCGAN
    :param train_dataset: training dataset
    :param train_loader: training data loader
    :param test_loader: testing data loader
    :param evaluator: evaluator
    :param learning_rate_g: learning rate of generator
    :param learning_rate_d: learning rate of discriminator
    :param image_size: image size (noise size)
    :param num_classes: number of classes (object IDs)
    :param epochs: number of epochs
    :param training_device: training device
    :return: None
    """
    # Setup models
    info_log('Setup models ...')
    generator = Generator(noise_size=image_size).to(training_device)
    discriminator = Discriminator(image_size=image_size).to(training_device)
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate_g)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate_d)

    # Setup average losses/accuracies container
    generator_losses = [0.0 for _ in range(epochs)]
    discriminator_losses = [0.0 for _ in range(epochs)]
    accuracies = [0.0 for _ in range(epochs)]

    # Start training
    info_log('Start training')
    for epoch in range(epochs):
        # Train
        total_g_loss, total_d_loss = train_dcgan(data_loader=train_loader,
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
        print(f'[{epoch + 1}/{epochs}] Average generator loss: {generator_losses[epoch]}')
        print(f'[{epoch + 1}/{epochs}] Average discriminator loss: {discriminator_losses[epoch]}')

        # Test
        generated_image, total_accuracy = test_dcgan(data_loader=test_loader,
                                                     generator=generator,
                                                     image_size=image_size,
                                                     num_classes=num_classes,
                                                     epoch=epoch,
                                                     num_of_epochs=epochs,
                                                     evaluator=evaluator,
                                                     training_device=training_device)
        accuracies[epoch] = total_accuracy / len(test_loader)
        save_image(make_grid(generated_image, nrow=8), f'test_figure/{epoch}.jpg')
        print(f'[{epoch + 1}/{epochs}] Average accuracy: {accuracies[epoch]:.2f}')

        if epoch % 10 == 0:
            torch.save(generator, f'model/{epoch}_{accuracies[epoch]:.4f}_g.pt')
            torch.save(discriminator, f'model/{epoch}_{accuracies[epoch]:.4f}_d.pt')

    # Plot losses and accuracies
    info_log('Plot losses and accuracies ...')
    plot_losses(losses=(generator_losses, discriminator_losses), labels=['Generator', 'Discriminator'])
    plot_accuracies(accuracies=accuracies)
    plt.close()


def train_dcgan(data_loader: DataLoader,
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
    Train the DCGAN
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
        optimizer_d.zero_grad()
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
        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        if batch_idx % 50 == 0:
            output_string = f'[{epoch + 1}/{num_of_epochs}][{batch_idx + 1}/{len(data_loader)}]   ' + \
                            f'Loss_D: {loss_d.item():.4f}   Loss_G: {loss_g.item():.4f}   ' + \
                            f'D(x): {d_x:.4f}   D(G(z)): {d_g_z1:.4f} / {d_g_z2:.4f}'
            debug_log(output_string)

    return total_g_loss, total_d_loss


def test_dcgan(data_loader: DataLoader,
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
        labels = labels.to(training_device).type(torch.float)
        batch_size = len(labels)

        # Generate batch of latent vectors
        noise = torch.cat([
            torch.randn(batch_size, image_size - num_classes),
            labels.cpu()
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

        debug_log(f'[{epoch + 1}/{num_of_epochs}][{batch_idx + 1}/{len(data_loader)}]   Accuracy: {acc}')

    return norm_image, total_accuracy


def train_and_evaluate_glow(train_dataset: ICLEVRLoader,
                            train_loader: DataLoader,
                            test_loader: DataLoader,
                            evaluator: EvaluationModel,
                            learning_rate_nf: float,
                            image_size: int,
                            width: int,
                            depth: int,
                            num_levels: int,
                            grad_norm_clip: float,
                            epochs: int,
                            training_device: device) -> None:
    """
    Train and test Glow
    :param train_dataset: training dataset
    :param train_loader: training data loader
    :param test_loader: testing data loader
    :param evaluator: evaluator
    :param learning_rate_nf: learning rate of normalizing flow
    :param image_size: image size (noise size)
    :param width: dimension of the hidden layers in normalizing flow
    :param depth: depth of the normalizing flow
    :param num_levels: number of levels in normalizing flow
    :param grad_norm_clip: clip gradients during training
    :param epochs: number of total epochs
    :param training_device: training device
    :return: None
    """
    # Setup average losses/accuracies container
    losses = [0.0 for _ in range(epochs)]
    accuracies = [0.0 for _ in range(epochs)]

    # Setup models
    info_log('Setup models ...')
    # glow = Glow(width=width, depth=depth, n_levels=num_levels).to(training_device)
    # glow = CondGlowModel(x_size=(3, image_size, image_size),
    #                      y_size=(3, image_size, image_size),
    #                      x_hidden_channels=128,
    #                      x_hidden_size=64,
    #                      y_hidden_channels=256,
    #                      flow_depth=depth,
    #                      num_levels=num_levels,
    #                      learn_top=False,
    #                      y_bins=2.0).to(training_device)
    glow = Glow(num_channels=width, num_levels=num_levels, num_steps=depth).to(training_device)
    optimizer = optim.Adam(glow.parameters(), lr=learning_rate_nf)
    loss_fn = NLLLoss().to(training_device)

    # Start training
    info_log('Start training')
    for epoch in range(epochs):
        # Train
        total_loss = train_glow(data_loader=train_loader,
                                glow=glow,
                                optimizer=optimizer,
                                loss_fn=loss_fn,
                                grad_norm_clip=grad_norm_clip,
                                epoch=epoch,
                                num_of_epochs=epochs,
                                training_device=training_device)
        losses[epoch] = total_loss / len(train_dataset)
        print(f'[{epoch + 1}/{epochs}] Average loss: {losses[epoch]}')

        # Test
        generated_image, total_accuracy = test_glow(data_loader=test_loader,
                                                    glow=glow,
                                                    epoch=epoch,
                                                    num_of_epochs=epochs,
                                                    evaluator=evaluator,
                                                    training_device=training_device)
        accuracies[epoch] = total_accuracy / len(test_loader)
        save_image(make_grid(generated_image, nrow=8), f'test_figure/{epoch}.jpg')
        print(f'[{epoch + 1}/{epochs}] Average accuracy: {accuracies[epoch]:.2f}')

    # Plot losses and accuracies
    info_log('Plot losses and accuracies ...')
    plot_losses(losses=(losses,), labels=['loss'])
    plot_accuracies(accuracies=accuracies)
    plt.close()


def train_glow(data_loader: DataLoader,
               glow: Glow,
               optimizer: optim,
               loss_fn: NLLLoss,
               grad_norm_clip: float,
               epoch: int,
               num_of_epochs: int,
               training_device: device) -> float:
    """
    Train Glow
    :param data_loader: training data loader
    :param glow: glow model
    :param optimizer: glow optimizer
    :param loss_fn: loss function
    :param grad_norm_clip: clipping gradient
    :param epoch: current epoch
    :param num_of_epochs: number of total epochs
    :param training_device: training device
    :return: total loss
    """
    glow.train()
    total_loss = 0.0
    for batch_idx, batch_data in enumerate(data_loader):
        images, labels = batch_data
        images = images.to(training_device)
        labels = labels.to(training_device).type(torch.float)

        # loss = -glow.log_prob(images, labels, bits_per_pixel=True).mean(0)
        z, nll = glow(images, labels)
        loss = loss_fn(z, nll)
        total_loss += loss.data.cpu().item()

        glow.zero_grad()
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(glow.parameters(), grad_norm_clip)

        optimizer.step()

        if batch_idx % 50 == 0:
            debug_log(f'[{epoch + 1}/{num_of_epochs}][{batch_idx + 1}/{len(data_loader)}]   Loss: {loss}')

    return total_loss


def test_glow(data_loader: DataLoader,
              glow: Glow,
              epoch: int,
              num_of_epochs: int,
              evaluator: EvaluationModel,
              training_device: device) -> Tuple[torch.Tensor, float]:
    """
    Test Glow
    :param data_loader: testing data loader
    :param glow: glow model
    :param epoch: current epoch
    :param num_of_epochs: number of total epochs
    :param evaluator: evaluator
    :param training_device: training device
    :return: generated images and total accuracy
    """
    glow.eval()
    total_accuracy = 0.0
    generated_image = torch.randn(0, 3, 64, 64)
    for batch_idx, batch_data in enumerate(data_loader):
        labels = batch_data
        labels = labels.to(training_device).type(torch.float)
        batch_size = len(labels)

        # sample, _ = glow.inverse(batch_size=batch_size)
        # sample = sample[:, :3, :, :]
        # log_prob = glow.log_prob(sample, labels, bits_per_pixel=True)
        # # sort by log_prob; flip high (left) to low (right)
        # fake_images = sample[log_prob.argsort().flip(0)]
        z = torch.randn((batch_size, 3, 64, 64), dtype=torch.float, device=training_device)
        fake_images, _ = glow(z, labels, reverse=True)
        fake_images = torch.sigmoid(fake_images)

        acc = evaluator.eval(fake_images, labels)
        total_accuracy += acc

        for fake_image in fake_images:
            n_image = fake_image.cpu().detach()
            generated_image = torch.cat([generated_image, n_image.view(1, 3, 64, 64)], 0)

        debug_log(f'[{epoch + 1}/{num_of_epochs}][{batch_idx + 1}/{len(data_loader)}]   Accuracy: {acc}')

    return generated_image, total_accuracy


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
    width = args.width
    depth = args.depth
    num_levels = args.num_levels
    grad_norm_clip = args.grad_norm_clip
    learning_rate_d = args.learning_rate_discriminator
    learning_rate_g = args.learning_rate_generator
    learning_rate_nf = args.learning_rate_normalizing_flow
    epochs = args.epochs
    task = args.task
    model = args.model
    global verbosity
    verbosity = args.verbosity
    info_log(f'Batch size: {batch_size}')
    info_log(f'Image size: {image_size}')
    info_log(f'Dimension of the hidden layers in normalizing flow: {width}')
    info_log(f'Depth of the normalizing flow: {depth}')
    info_log(f'Number of levels in normalizing flow: {num_levels}')
    info_log(f'Clip gradients during training: {grad_norm_clip}')
    info_log(f'Learning rate of discriminator: {learning_rate_d}')
    info_log(f'Learning rate of generator: {learning_rate_g}')
    info_log(f'Learning rate of normalizing flow: {learning_rate_nf}')
    info_log(f'Number of epochs: {epochs}')
    info_log(f'Perform task: {task}')
    info_log(f'Which model will be used: {model}')
    info_log(f'Training device: {training_device}')

    # Read data
    info_log('Read data ...')

    if model == 'gan':
        transformation = transforms.Compose([transforms.RandomCrop(240),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.Resize((image_size, image_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transformation = transforms.Compose([transforms.RandomCrop(240),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.Resize((image_size, image_size)),
                                             transforms.ToTensor()])
    # TODO: control data for different task
    train_dataset = ICLEVRLoader(root_folder='data/task_1/', trans=transformation, mode='train')
    test_dataset = ICLEVRLoader(root_folder='data/task_1/', mode='test')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    num_classes = train_dataset.num_classes

    # Setup evaluator
    evaluator = EvaluationModel(training_device=training_device)

    # Create directories
    if not os.path.exists('./model'):
        os.mkdir('./model')
    if not os.path.exists('./test_figure'):
        os.mkdir('./test_figure')
    if not os.path.exists('./figure'):
        os.mkdir('./figure')

    if task == 1:
        if model == 'gan':
            train_and_evaluate_dcgan(train_dataset=train_dataset,
                                     train_loader=train_loader,
                                     test_loader=test_loader,
                                     evaluator=evaluator,
                                     learning_rate_g=learning_rate_g,
                                     learning_rate_d=learning_rate_d,
                                     image_size=image_size,
                                     num_classes=num_classes,
                                     epochs=epochs,
                                     training_device=training_device)
        else:
            train_and_evaluate_glow(train_dataset=train_dataset,
                                    train_loader=train_loader,
                                    test_loader=test_loader,
                                    evaluator=evaluator,
                                    learning_rate_nf=learning_rate_nf,
                                    image_size=image_size,
                                    width=width,
                                    depth=depth,
                                    num_levels=num_levels,
                                    grad_norm_clip=grad_norm_clip,
                                    epochs=epochs,
                                    training_device=training_device)


if __name__ == '__main__':
    verbosity = None
    main()
