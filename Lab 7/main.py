from dcgan import DCGenerator, DCDiscriminator
from sagan import SAGenerator, SADiscriminator
from task_1_dataset import ICLEVRLoader
from normalizing_flow import CGlow, NLLLoss
from task_2_dataset import CelebALoader
from evaluator import EvaluationModel
from argument_parser import parse_arguments
from visualizer import plot_losses, plot_accuracies
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image, make_grid
from torch import device, cuda
from typing import Tuple, Optional
import matplotlib.pyplot as plt
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


def train_and_evaluate_cgan(train_dataset: ICLEVRLoader,
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
    Train and test cGAN
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
    # generator = DCGenerator(noise_size=image_size).to(training_device)
    # discriminator = DCDiscriminator(image_size=image_size).to(training_device)
    generator = SAGenerator(z_dim=image_size, g_conv_dim=image_size // 2, num_classes=num_classes).to(training_device)
    discriminator = SADiscriminator(d_conv_dim=image_size // 2, num_classes=num_classes).to(training_device)
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
        total_g_loss, total_d_loss = train_cgan(data_loader=train_loader,
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
        generated_image, total_accuracy = test_cgan(data_loader=test_loader,
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
            torch.save(generator, f'model/task_1/{epoch}_{accuracies[epoch]:.4f}_g.pt')
            torch.save(discriminator, f'model/task_2/{epoch}_{accuracies[epoch]:.4f}_d.pt')

    # Plot losses and accuracies
    info_log('Plot losses and accuracies ...')
    plot_losses(losses=(generator_losses, discriminator_losses), labels=['Generator', 'Discriminator'], task='task_1')
    plot_accuracies(accuracies=accuracies)
    plt.close()


def train_cgan(data_loader: DataLoader,
               generator: Optional[DCGenerator or SAGenerator],
               discriminator: Optional[DCDiscriminator or SADiscriminator],
               optimizer_g: optim,
               optimizer_d: optim,
               image_size: int,
               num_classes: int,
               epoch: int,
               num_of_epochs: int,
               training_device: device) -> Tuple[float, float]:
    """
    Train the cGAN
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
        # DCGAN
        # real_labels = real_labels.to(training_device).type(torch.float)
        # SAGAN
        real_labels = real_labels.to(training_device).type(torch.long)

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
        # DCGAN
        # loss_d_real = nn.BCELoss()(outputs, labels)
        # SAGAN with WGAN-hinge
        loss_d_real = nn.ReLU()(1.0 - outputs).mean()
        loss_d_real.backward()

        # Train with all-fake batch
        # Generate batch of latent vectors
        # DCGAN
        # noise = torch.cat([
        #     torch.randn(batch_size, image_size - num_classes),
        #     real_labels.cpu()
        # ], 1).view(-1, image_size, 1, 1).to(training_device)
        # SAGAN
        noise = torch.randn((batch_size, image_size)).to(training_device)
        labels = torch.full((batch_size, 1), 0.0, dtype=torch.float, device=training_device)

        # Generate fake image batch with generator
        # DCGAN
        # fake_outputs = generator.forward(noise)
        # SAGAN
        fake_outputs = generator.forward(noise, real_labels)

        # Forward pass fake batch through discriminator
        outputs = discriminator.forward(fake_outputs.detach(), real_labels)
        d_g_z1 = outputs.mean().item()

        # Calculate loss on fake batch
        # DCGAN
        # loss_d_fake = nn.BCELoss()(outputs, labels)
        # SAGAN with WGAN-hinge
        loss_d_fake = torch.nn.ReLU()(1.0 + outputs).mean()
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
        # DCGAN
        # loss_g = nn.BCELoss()(outputs, labels)
        # SAGAN with WGAN-hinge
        loss_g = -outputs.mean()
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


def test_cgan(data_loader: DataLoader,
              generator: Optional[DCGenerator or SAGenerator],
              image_size: int,
              num_classes: int,
              epoch: int,
              num_of_epochs: int,
              evaluator: EvaluationModel,
              training_device: device) -> Tuple[torch.Tensor, float]:
    """
    Test the cGAN
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
        # DCGAN
        # labels = labels.to(training_device).type(torch.float)
        # SAGAN
        labels = labels.to(training_device).type(torch.long)
        batch_size = len(labels)

        # Generate batch of latent vectors
        # DCGAN
        # noise = torch.cat([
        #     torch.randn(batch_size, image_size - num_classes),
        #     labels.cpu()
        # ], 1).view(-1, image_size, 1, 1).to(training_device)
        # SAGAN
        noise = torch.randn((batch_size, image_size)).to(training_device)

        # Generate fake image batch with generator
        # DCGAN
        # fake_outputs = generator.forward(noise)
        # SAGAN
        fake_outputs = generator.forward(noise, labels)

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


def train_and_evaluate_cglow(train_dataset: ICLEVRLoader,
                             train_loader: DataLoader,
                             test_loader: DataLoader,
                             evaluator: EvaluationModel,
                             learning_rate_nf: float,
                             image_size: int,
                             width: int,
                             depth: int,
                             num_levels: int,
                             num_classes: int,
                             grad_norm_clip: float,
                             epochs: int,
                             training_device: device) -> None:
    """
    Train and test cGlow
    :param train_dataset: training dataset
    :param train_loader: training data loader
    :param test_loader: testing data loader
    :param evaluator: evaluator
    :param learning_rate_nf: learning rate of normalizing flow
    :param image_size: image size (noise size)
    :param width: dimension of the hidden layers in normalizing flow
    :param depth: depth of the normalizing flow
    :param num_levels: number of levels in normalizing flow
    :param num_classes: number of different conditions
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
    glow = CGlow(num_channels=width, num_levels=num_levels, num_steps=depth, num_classes=num_classes,
                 image_size=image_size).to(training_device)
    optimizer = optim.Adam(glow.parameters(), lr=learning_rate_nf)
    loss_fn = NLLLoss().to(training_device)

    # Start training
    info_log('Start training')
    for epoch in range(epochs):
        # Train
        total_loss = train_cglow(data_loader=train_loader,
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
        generated_image, total_accuracy = test_cglow(data_loader=test_loader,
                                                     glow=glow,
                                                     epoch=epoch,
                                                     num_of_epochs=epochs,
                                                     evaluator=evaluator,
                                                     training_device=training_device)
        accuracies[epoch] = total_accuracy / len(test_loader)
        save_image(make_grid(generated_image, nrow=8), f'test_figure/{epoch}.jpg')
        print(f'[{epoch + 1}/{epochs}] Average accuracy: {accuracies[epoch]:.2f}')

        if epoch % 10 == 0:
            torch.save(glow, f'model/task_1/{epoch}_{accuracies[epoch]:.4f}.pt')

    # Plot losses and accuracies
    info_log('Plot losses and accuracies ...')
    plot_losses(losses=(losses,), labels=['loss'], task='task_1')
    plot_accuracies(accuracies=accuracies)
    plt.close()


def train_cglow(data_loader: DataLoader,
                glow: CGlow,
                optimizer: optim,
                loss_fn: NLLLoss,
                grad_norm_clip: float,
                epoch: int,
                num_of_epochs: int,
                training_device: device) -> float:
    """
    Train cGlow
    :param data_loader: training data loader
    :param glow: conditional glow model
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


def test_cglow(data_loader: DataLoader,
               glow: CGlow,
               epoch: int,
               num_of_epochs: int,
               evaluator: EvaluationModel,
               training_device: device) -> Tuple[torch.Tensor, float]:
    """
    Test cGlow
    :param data_loader: testing data loader
    :param glow: conditional glow model
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

        z = torch.randn((batch_size, 3, 64, 64), dtype=torch.float, device=training_device)
        fake_images, _ = glow(z, labels, reverse=True)
        fake_images = torch.sigmoid(fake_images)

        transformed_images = torch.randn(0, 3, 64, 64)
        transformation = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        for fake_image in fake_images:
            n_image = fake_image.cpu().detach()
            transformed_images = torch.cat([transformed_images, transformation(n_image).view(1, 3, 64, 64)], 0)
        transformed_images = transformed_images.to(training_device)

        acc = evaluator.eval(transformed_images, labels)
        total_accuracy += acc

        for fake_image in fake_images:
            n_image = fake_image.cpu().detach()
            generated_image = torch.cat([generated_image, n_image.view(1, 3, 64, 64)], 0)

        debug_log(f'[{epoch + 1}/{num_of_epochs}][{batch_idx + 1}/{len(data_loader)}]   Accuracy: {acc}')

    return generated_image, total_accuracy


def train_and_inference_celeb(train_dataset: CelebALoader,
                              train_loader: DataLoader,
                              learning_rate_nf: float,
                              image_size: int,
                              width: int,
                              depth: int,
                              num_levels: int,
                              num_classes: int,
                              grad_norm_clip: float,
                              epochs: int,
                              training_device: device) -> None:
    """
    Train and inference cGlow
    :param train_dataset: training dataset
    :param train_loader: training data loader
    :param learning_rate_nf: learning rate of normalizing flow
    :param image_size: image size (noise size)
    :param width: dimension of the hidden layers in normalizing flow
    :param depth: depth of the normalizing flow
    :param num_levels: number of levels in normalizing flow
    :param num_classes: number of different conditions
    :param grad_norm_clip: clip gradients during training
    :param epochs: number of total epochs
    :param training_device: training device
    :return: None
    """
    # Setup average losses container
    losses = [0.0 for _ in range(epochs)]

    # Setup models
    info_log('Setup models ...')
    glow = CGlow(num_channels=width, num_levels=num_levels, num_steps=depth, num_classes=num_classes,
                 image_size=image_size).to(training_device)
    optimizer = optim.Adam(glow.parameters(), lr=learning_rate_nf)
    loss_fn = NLLLoss().to(training_device)

    # Start training
    info_log('Start training')
    for epoch in range(epochs):
        # Train
        total_loss = train_cglow(data_loader=train_loader,
                                 glow=glow,
                                 optimizer=optimizer,
                                 loss_fn=loss_fn,
                                 grad_norm_clip=grad_norm_clip,
                                 epoch=epoch,
                                 num_of_epochs=epochs,
                                 training_device=training_device)
        losses[epoch] = total_loss / len(train_dataset)
        print(f'[{epoch + 1}/{epochs}] Average loss: {losses[epoch]}')

        inference_celeb(train_dataset=train_dataset,
                        glow=glow,
                        num_classes=num_classes,
                        training_device=training_device)

    # Plot losses
    info_log('Plot losses ...')
    plot_losses(losses=(losses,), labels=['loss'], task='task_2')
    plt.close()


def inference_celeb(train_dataset: CelebALoader,
                    glow: CGlow,
                    num_classes: int,
                    training_device: device) -> None:
    """
    Use cGlow to inference celebrity data with 3 applications
    :param train_dataset: training dataset
    :param glow: conditional glow model
    :param num_classes: number of classes (attributes)
    :param training_device: training device
    :return: None
    """
    glow.eval()

    # Application 1
    # Get labels for inference
    debug_log(f'Perform app 1')
    labels = torch.rand(0, num_classes)
    for idx in range(32):
        _, label = train_dataset[idx]
        labels = torch.cat([labels, torch.from_numpy(label).view(1, 40)], 0)
    labels = labels.to(training_device).type(torch.float)

    # Generate random latent code
    z = torch.randn((32, 3, 64, 64), dtype=torch.float, device=training_device)

    # Produce fake images
    fake_images, _ = glow(z, labels, reverse=True)
    fake_images = torch.sigmoid(fake_images)

    # Save fake images for application 1
    generated_images = torch.randn(0, 3, 64, 64)
    for fake_image in fake_images:
        n_image = fake_image.cpu().detach()
        generated_images = torch.cat([generated_images, n_image.view(1, 3, 64, 64)], 0)
    save_image(make_grid(generated_images, nrow=8), f'figure/task_2/app_1.jpg')

    # Application 2
    # Get 2 images to perform linear interpolation
    debug_log(f'Perform app 2')
    linear_images = torch.randn(0, 3, 64, 64)
    for idx in range(5):
        # Get first image and label
        first_image, first_label = train_dataset[idx]
        first_image = first_image.to(training_device).type(torch.float).view(1, 3, 64, 64)
        first_label = torch.from_numpy(first_label).to(training_device).type(torch.float).view(1, 40)

        # Get second image and label
        second_image, second_label = train_dataset[idx + 5]
        second_image = second_image.to(training_device).type(torch.float).view(1, 3, 64, 64)
        second_label = torch.from_numpy(second_label).to(training_device).type(torch.float).view(1, 40)

        # Generate latent code
        first_z, _ = glow(first_image, first_label)
        second_z, _ = glow(second_image, second_label)

        # Compute interval
        interval_z = (second_z - first_z) / 8.0
        interval_label = (second_label - first_label) / 8.0

        # Generate linear images
        for num_of_intervals in range(9):
            image, _ = glow(first_z + num_of_intervals * interval_z,
                            first_label + num_of_intervals * interval_label,
                            reverse=True)
            image = torch.sigmoid(image)
            linear_images = torch.cat([linear_images, image.cpu().detach().view(1, 3, 64, 64)], 0)
    save_image(make_grid(linear_images, nrow=9), f'figure/task_2/app_2.jpg')

    # Application 3
    # Get a image and labels with negative/positive smiling
    debug_log(f'Perform app 3')
    image, label = train_dataset[1]
    image = image.to(training_device).type(torch.float).view(1, 3, 64, 64)
    label = torch.from_numpy(label).to(training_device).type(torch.float).view(1, 40)
    neg_smile_label = torch.clone(label)
    neg_smile_label[0, 31] = -1.
    pos_smile_label = torch.clone(label)
    pos_smile_label[0, 31] = 1.

    # Generate latent code
    neg_z, _ = glow(image, neg_smile_label)
    pos_z, _ = glow(image, pos_smile_label)

    # Compute interval
    interval_z = (pos_z - neg_z) / 4.0
    interval_label = (pos_smile_label - neg_smile_label) / 4.0

    # Generate manipulated images
    manipulated_images = torch.randn(0, 3, 64, 64)
    for num_of_intervals in range(5):
        image, _ = glow(neg_z + num_of_intervals * interval_z,
                        neg_smile_label + num_of_intervals * interval_label,
                        reverse=True)
        image = torch.sigmoid(image)
        manipulated_images = torch.cat([manipulated_images, image.cpu().detach().view(1, 3, 64, 64)], 0)
    save_image(make_grid(manipulated_images, nrow=5), f'figure/task_2/app_3.jpg')


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
        if task == 1:
            transformation = transforms.Compose([transforms.RandomCrop(240),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.Resize((image_size, image_size)),
                                                 transforms.ToTensor()])
        else:
            transformation = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                 transforms.Resize((image_size, image_size)),
                                                 transforms.ToTensor()])

    if task == 1:
        train_dataset = ICLEVRLoader(root_folder='data/task_1/', trans=transformation, mode='train')
        test_dataset = ICLEVRLoader(root_folder='data/task_1/', mode='test')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        num_classes = train_dataset.num_classes
    else:
        train_dataset = CelebALoader(root_folder='data/task_2/', trans=transformation)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        num_classes = train_dataset.num_classes

    # Setup evaluator
    evaluator = EvaluationModel(training_device=training_device)

    # Create directories
    if not os.path.exists('./model/task_1'):
        os.makedirs('./model/task_1')
    if not os.path.exists('./model/task_2'):
        os.makedirs('./model/task_2')
    if not os.path.exists('./test_figure'):
        os.mkdir('./test_figure')
    if not os.path.exists('./figure/task_1'):
        os.makedirs('./figure/task_1')
    if not os.path.exists('./figure/task_2'):
        os.makedirs('./figure/task_2')

    if task == 1:
        if model == 'gan':
            train_and_evaluate_cgan(train_dataset=train_dataset,
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
            train_and_evaluate_cglow(train_dataset=train_dataset,
                                     train_loader=train_loader,
                                     test_loader=test_loader,
                                     evaluator=evaluator,
                                     learning_rate_nf=learning_rate_nf,
                                     image_size=image_size,
                                     width=width,
                                     depth=depth,
                                     num_levels=num_levels,
                                     num_classes=num_classes,
                                     grad_norm_clip=grad_norm_clip,
                                     epochs=epochs,
                                     training_device=training_device)
    else:
        train_and_inference_celeb(train_dataset=train_dataset,
                                  train_loader=train_loader,
                                  learning_rate_nf=learning_rate_nf,
                                  image_size=image_size,
                                  width=width,
                                  depth=depth,
                                  num_levels=num_levels,
                                  num_classes=num_classes,
                                  grad_norm_clip=grad_norm_clip,
                                  epochs=epochs,
                                  training_device=training_device)


if __name__ == '__main__':
    verbosity = None
    main()
