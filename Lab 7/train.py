from dcgan import DCGenerator, DCDiscriminator
from sagan import SAGenerator, SADiscriminator
from srgan import SRGenerator, SRDiscriminator, GeneratorLoss
from glow import CGlow, NLLLoss
from util import debug_log
from torch.utils.data import DataLoader
from torch import device
from typing import Tuple, Optional
from argparse import Namespace
import torch.optim as optim
import torch.nn as nn
import torch


def train_cgan(data_loader: DataLoader,
               generator: Optional[DCGenerator or SAGenerator or SRGenerator],
               discriminator: Optional[DCDiscriminator or SADiscriminator or SRDiscriminator],
               optimizer_g: optim,
               optimizer_d: optim,
               scheduler_g: optim.lr_scheduler,
               scheduler_d: optim.lr_scheduler,
               num_classes: int,
               epoch: int,
               args: Namespace,
               training_device: device) -> Tuple[float, float]:
    """
    Train cGAN
    :param data_loader: Training data loader
    :param generator: Generator
    :param discriminator: Discriminator
    :param optimizer_g: Optimizer for generator
    :param optimizer_d: Optimizer for discriminator
    :param scheduler_g: Learning rate scheduler for generator
    :param scheduler_d: Learning rate scheduler for discriminator
    :param num_classes: Number of classes (object IDs)
    :param epoch: Current epoch
    :param args: All arguments
    :param training_device: Training device
    :return: Total generator loss and total discriminator loss
    """
    generator.train()
    discriminator.train()
    total_g_loss = 0.0
    total_d_loss = 0.0

    if args.model == 'DCGAN':
        # DCGAN
        criterion = nn.BCELoss().to(training_device)
    elif args.model == 'SAGAN':
        # SAGAN
        criterion = nn.ReLU().to(training_device)
    else:
        # SRGAN
        criterion = GeneratorLoss().to(training_device)

    for batch_idx, batch_data in enumerate(data_loader):
        images, real_labels = batch_data
        if args.model == 'DCGAN' or args.model == 'SRGAN':
            # DCGAN or SRGAN
            real_labels = real_labels.to(training_device).type(torch.float)
        else:
            # SAGAN
            real_labels = real_labels.to(training_device).type(torch.long)

        ############################
        # (1) Update D network
        ###########################
        # Train with all-real batch
        # Format batch
        images = images.to(training_device)
        batch_size = images.size(0)
        labels = torch.full((batch_size, 1), 1.0, dtype=torch.float, device=training_device)

        # Forward pass real batch through discriminator
        outputs = discriminator.forward(images, real_labels)

        # Calculate loss on all-real batch
        optimizer_d.zero_grad()
        if args.model == 'DCGAN':
            # DCGAN
            loss_d_real = criterion(outputs, labels)
        elif args.model == 'SAGAN':
            # SAGAN with WGAN-hinge
            loss_d_real = criterion(1.0 - outputs).mean()
        else:
            # SRGAN
            loss_d_real = (1.0 - outputs).mean()
        loss_d_real.backward()

        # Train with all-fake batch
        # Generate batch of latent vectors
        if args.model == 'DCGAN':
            # DCGAN
            noise = torch.cat([
                torch.randn((batch_size, args.image_size)),
                real_labels.cpu()
            ], 1).view(-1, args.image_size + num_classes, 1, 1).to(training_device)
        elif args.model == 'SAGAN':
            # SAGAN
            noise = torch.randn((batch_size, args.image_size)).to(training_device)
        else:
            # SRGAN
            noise = torch.randn((batch_size, 3, args.image_size, args.image_size)).to(training_device)
        labels = torch.full((batch_size, 1), 0.0, dtype=torch.float, device=training_device)

        # Generate fake image batch with generator
        if args.model == 'DCGAN':
            # DCGAN
            fake_outputs = generator.forward(noise)
        else:
            # SAGAN or SRGAN
            fake_outputs = generator.forward(noise, real_labels)

        # Forward pass fake batch through discriminator
        outputs = discriminator.forward(fake_outputs.detach(), real_labels)

        # Calculate loss on fake batch
        if args.model == 'DCGAN':
            # DCGAN
            loss_d_fake = criterion(outputs, labels)
        elif args.model == 'SAGAN':
            # SAGAN with WGAN-hinge
            loss_d_fake = criterion(1.0 + outputs).mean()
        else:
            # SRGAN
            loss_d_fake = outputs.mean()
        loss_d_fake.backward()
        optimizer_d.step()
        scheduler_d.step()

        # Compute loss of discriminator as sum over the fake and the real batches
        loss_d = loss_d_real + loss_d_fake
        total_d_loss += loss_d.item()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        # Since we just updated discriminator, perform another forward pass of all-fake batch through discriminator
        outputs = discriminator.forward(fake_outputs, real_labels)
        labels = torch.full((batch_size, 1), 1.0, dtype=torch.float, device=training_device)

        # Calculate generator's loss based on this output
        if args.model == 'DCGAN':
            # DCGAN
            loss_g = criterion(outputs, labels)
        elif args.model == 'SAGAN':
            # SAGAN with WGAN-hinge
            loss_g = -outputs.mean()
        else:
            # SRGAN
            loss_g = criterion(outputs, fake_outputs, images)
        total_g_loss += loss_g.item()

        # Calculate gradients for generator and update
        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()
        scheduler_g.step()

        if batch_idx % 50 == 0:
            output_string = f'[{epoch + 1}/{args.epochs}][{batch_idx + 1}/{len(data_loader)}]   ' + \
                            f'Loss_D: {loss_d.item():.4f}   Loss_G: {loss_g.item():.4f}'
            debug_log(output_string, args.verbosity)

    return total_g_loss, total_d_loss


def train_cnf(data_loader: DataLoader,
              normalizing_flow: CGlow,
              optimizer: optim,
              scheduler: optim.lr_scheduler,
              loss_fn: NLLLoss,
              epoch: int,
              args: Namespace,
              training_device: device) -> float:
    """
    Train cNF
    :param data_loader: Training data loader
    :param normalizing_flow: Conditional normalizing flow model
    :param optimizer: Glow optimizer
    :param scheduler: Learning rate scheduler
    :param loss_fn: Loss function
    :param epoch: Current epoch
    :param args: All arguments
    :param training_device: Training device
    :return: Total loss
    """
    normalizing_flow.train()
    total_loss = 0.0
    for batch_idx, batch_data in enumerate(data_loader):
        images, labels = batch_data
        images = images.to(training_device)
        labels = labels.to(training_device).type(torch.float)

        if epoch == 0 and batch_idx == 0:
            # Initialize the network
            normalizing_flow.forward(x=images, x_label=labels)

        z, nll, label_logits = normalizing_flow.forward(x=images, x_label=labels)
        loss = loss_fn.forward(nll=nll, label_logits=label_logits, labels=labels)
        total_loss += loss.data.cpu().item()

        optimizer.zero_grad()
        loss.backward()

        if args.grad_value_clip > 0:
            nn.utils.clip_grad_value_(normalizing_flow.parameters(), args.grad_value_clip)
        if args.grad_norm_clip > 0:
            nn.utils.clip_grad_norm_(normalizing_flow.parameters(), args.grad_norm_clip)

        optimizer.step()
        scheduler.step()

        if batch_idx % 50 == 0:
            debug_log(f'[{epoch + 1}/{args.epochs}][{batch_idx + 1}/{len(data_loader)}]   Loss: {loss}',
                      args.verbosity)

    return total_loss
