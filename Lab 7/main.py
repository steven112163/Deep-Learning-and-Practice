from dcgan import DCGenerator, DCDiscriminator, weights_init
from sagan import SAGenerator, SADiscriminator
from srgan import SRGenerator, SRDiscriminator
from glow import CGlow, NLLLoss
from task_1_dataset import ICLEVRLoader
from task_2_dataset import CelebALoader
from train import train_cgan, train_cnf
from test import test_cgan, test_cnf, inference_celeb
from evaluator import EvaluationModel
from argument_parser import parse_arguments
from visualizer import plot_losses, plot_accuracies
from util import info_log, create_directories
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image, make_grid
from torch import device, cuda
from argparse import Namespace
from math import inf
import torch.optim as optim
import os
import torch


def train_and_evaluate_cgan(train_dataset: ICLEVRLoader,
                            train_loader: DataLoader,
                            test_loader: DataLoader,
                            evaluator: EvaluationModel,
                            num_classes: int,
                            args: Namespace,
                            training_device: device) -> None:
    """
    Train and test cGAN
    :param train_dataset: training dataset
    :param train_loader: training data loader
    :param test_loader: testing data loader
    :param evaluator: evaluator
    :param num_classes: number of classes (object IDs)
    :param args: all arguments
    :param training_device: training device
    :return: None
    """
    # Setup models
    info_log('Setup models ...', args.verbosity)

    if args.model == 'DCGAN':
        # DCGAN
        generator = DCGenerator(noise_size=args.image_size,
                                label_size=num_classes).to(training_device)
        discriminator = DCDiscriminator(num_classes=num_classes,
                                        image_size=args.image_size).to(training_device)
        generator.apply(weights_init)
        discriminator.apply(weights_init)
    elif args.model == 'SAGAN':
        # Self Attention GAN
        generator = SAGenerator(z_dim=args.image_size,
                                g_conv_dim=args.image_size // 2,
                                num_classes=num_classes).to(training_device)
        discriminator = SADiscriminator(d_conv_dim=args.image_size // 2,
                                        num_classes=num_classes).to(training_device)
    else:
        # Super Resolution GAN
        generator = SRGenerator(scale_factor=1,
                                num_classes=num_classes,
                                image_size=args.image_size).to(training_device)
        discriminator = SRDiscriminator(num_classes=num_classes,
                                        image_size=args.image_size).to(training_device)

    if os.path.exists(f'model/task_1/{args.model}.pt'):
        checkpoint = torch.load(f'model/task_1/{args.model}.pt')
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])

    optimizer_g = optim.Adam(generator.parameters(), lr=args.learning_rate_generator, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.learning_rate_discriminator, betas=(0.5, 0.999))

    # Setup average losses/accuracies container
    generator_losses = [0.0 for _ in range(args.epochs)]
    discriminator_losses = [0.0 for _ in range(args.epochs)]
    accuracies = [0.0 for _ in range(args.epochs)]

    if not args.inference:
        # Start training
        info_log('Start training', args.verbosity)
        max_acc = 0.0
        for epoch in range(args.epochs):
            # Train
            total_g_loss, total_d_loss = train_cgan(data_loader=train_loader,
                                                    generator=generator,
                                                    discriminator=discriminator,
                                                    optimizer_g=optimizer_g,
                                                    optimizer_d=optimizer_d,
                                                    num_classes=num_classes,
                                                    epoch=epoch,
                                                    args=args,
                                                    training_device=training_device)
            generator_losses[epoch] = total_g_loss / len(train_dataset)
            discriminator_losses[epoch] = total_d_loss / len(train_dataset)
            print(f'[{epoch + 1}/{args.epochs}]   Average generator loss: {generator_losses[epoch]}')
            print(f'[{epoch + 1}/{args.epochs}]   Average discriminator loss: {discriminator_losses[epoch]}')

            # Test
            generated_image, total_accuracy = test_cgan(data_loader=test_loader,
                                                        generator=generator,
                                                        num_classes=num_classes,
                                                        epoch=epoch,
                                                        evaluator=evaluator,
                                                        args=args,
                                                        training_device=training_device)
            accuracies[epoch] = total_accuracy / len(test_loader)
            print(f'[{epoch + 1}/{args.epochs}]   Average accuracy: {accuracies[epoch]:.2f}')

            # Save generator and discriminator, and plot test image
            if accuracies[epoch] > max_acc:
                max_acc = accuracies[epoch]
                save_image(make_grid(generated_image, nrow=8),
                           f'test_figure/{args.model}_{epoch}_{accuracies[epoch]:.2f}.jpg')
                checkpoint = {
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict()
                }
                torch.save(checkpoint, f'model/task_1/{args.model}_{epoch}_{accuracies[epoch]:.4f}.pt')

        # Plot losses and accuracies
        info_log('Plot losses and accuracies ...', args.verbosity)
        plot_losses(losses=(generator_losses, discriminator_losses), labels=['Generator', 'Discriminator'],
                    task='task_1', model=args.model)
        plot_accuracies(accuracies=accuracies, model=args.model)
    else:
        # Start inferring
        info_log('Start inferring', args.verbosity)
        generated_image, total_accuracy = test_cgan(data_loader=test_loader,
                                                    generator=generator,
                                                    num_classes=num_classes,
                                                    epoch=0,
                                                    evaluator=evaluator,
                                                    args=args,
                                                    training_device=training_device)
        total_accuracy /= len(test_loader)
        save_image(make_grid(generated_image, nrow=8), f'figure/task_1/{args.model}_{total_accuracy:.2f}.png')
        print(f'Average accuracy: {total_accuracy:.2f}')


def train_and_evaluate_cnf(train_dataset: ICLEVRLoader,
                           train_loader: DataLoader,
                           test_loader: DataLoader,
                           evaluator: EvaluationModel,
                           num_classes: int,
                           args: Namespace,
                           training_device: device) -> None:
    """
    Train and test cNF
    :param train_dataset: training dataset
    :param train_loader: training data loader
    :param test_loader: testing data loader
    :param evaluator: evaluator
    :param num_classes: number of different conditions
    :param args: all arguments
    :param training_device: training device
    :return: None
    """
    # Setup models
    info_log('Setup models ...', args.verbosity)

    normalizing_flow = CGlow(num_channels=args.width,
                             num_levels=args.num_levels,
                             num_steps=args.depth,
                             num_classes=num_classes,
                             image_size=args.image_size).to(training_device)
    if os.path.exists(f'model/task_1/{args.model}.pt'):
        checkpoint = torch.load(f'/model/task_1/{args.model}.pt')
        normalizing_flow.load_state_dict(checkpoint['normalizing_flow'])

    optimizer = optim.Adam(normalizing_flow.parameters(), lr=args.learning_rate_normalizing_flow)
    loss_fn = NLLLoss().to(training_device)

    # Setup average losses/accuracies container
    losses = [0.0 for _ in range(args.epochs)]
    accuracies = [0.0 for _ in range(args.epochs)]

    if not args.inference:
        # Start training
        info_log('Start training', args.verbosity)
        max_acc = 0.0
        for epoch in range(args.epochs):
            # Train
            total_loss = train_cnf(data_loader=train_loader,
                                   normalizing_flow=normalizing_flow,
                                   optimizer=optimizer,
                                   loss_fn=loss_fn,
                                   epoch=epoch,
                                   args=args,
                                   training_device=training_device)
            losses[epoch] = total_loss / len(train_dataset)
            print(f'[{epoch + 1}/{args.epochs}]   Average loss: {losses[epoch]}')

            # Test
            generated_image, total_accuracy = test_cnf(data_loader=test_loader,
                                                       normalizing_flow=normalizing_flow,
                                                       epoch=epoch,
                                                       evaluator=evaluator,
                                                       args=args,
                                                       training_device=training_device)
            accuracies[epoch] = total_accuracy / len(test_loader)
            print(f'[{epoch + 1}/{args.epochs}]   Average accuracy: {accuracies[epoch]:.2f}')

            # Save normalizing flow, and plot test image
            if accuracies[epoch] > max_acc:
                max_acc = accuracies[epoch]
                save_image(make_grid(generated_image, nrow=8),
                           f'test_figure/{args.model}_{epoch}_{accuracies[epoch]:.2f}.jpg')
                checkpoint = {'normalizing_flow': normalizing_flow.state_dict()}
                torch.save(checkpoint, f'model/task_1/{args.model}_{epoch}_{accuracies[epoch]:.4f}.pt')

        # Plot losses and accuracies
        info_log('Plot losses and accuracies ...', args.verbosity)
        plot_losses(losses=(losses,), labels=['loss'], task='task_1', model=args.model)
        plot_accuracies(accuracies=accuracies, model=args.model)
    else:
        # Start inferring
        info_log('Start inferring', args.verbosity)
        generated_image, total_accuracy = test_cnf(data_loader=test_loader,
                                                   normalizing_flow=normalizing_flow,
                                                   epoch=0,
                                                   evaluator=evaluator,
                                                   args=args,
                                                   training_device=training_device)
        total_accuracy /= len(test_loader)
        save_image(make_grid(generated_image, nrow=8), f'figure/task_1/{args.model}_{total_accuracy:.2f}.png')
        print(f'Average accuracy: {total_accuracy:.2f}')


def train_and_inference_celeb(train_dataset: CelebALoader,
                              train_loader: DataLoader,
                              num_classes: int,
                              args: Namespace,
                              training_device: device) -> None:
    """
    Train and inference cGlow
    :param train_dataset: training dataset
    :param train_loader: training data loader
    :param num_classes: number of different conditions
    :param args: all arguments
    :param training_device: training device
    :return: None
    """
    # Setup models
    info_log('Setup models ...', args.verbosity)

    normalizing_flow = CGlow(num_channels=args.width,
                             num_levels=args.num_levels,
                             num_steps=args.depth,
                             num_classes=num_classes,
                             image_size=args.image_size).to(training_device)
    if os.path.exists(f'model/task_2/{args.model}.pt'):
        checkpoint = torch.load(f'/model/task_2/{args.model}.pt')
        normalizing_flow.load_state_dict(checkpoint['normalizing_flow'])

    optimizer = optim.Adam(normalizing_flow.parameters(), lr=args.learning_rate_normalizing_flow)
    loss_fn = NLLLoss().to(training_device)

    # Setup average losses container
    losses = [0.0 for _ in range(args.epochs)]

    if not args.inference:
        # Start training
        info_log('Start training', args.verbosity)
        min_loss = inf
        for epoch in range(args.epochs):
            # Train
            total_loss = train_cnf(data_loader=train_loader,
                                   normalizing_flow=normalizing_flow,
                                   optimizer=optimizer,
                                   loss_fn=loss_fn,
                                   epoch=epoch,
                                   args=args,
                                   training_device=training_device)
            losses[epoch] = total_loss / len(train_dataset)
            print(f'[{epoch + 1}/{args.epochs}]   Average loss: {losses[epoch]}')

            # 3 applications
            inference_celeb(train_dataset=train_dataset,
                            normalizing_flow=normalizing_flow,
                            num_classes=num_classes,
                            args=args,
                            training_device=training_device)

            # Save the model
            if losses[epoch] < min_loss:
                min_loss = losses[epoch]
                checkpoint = {'normalizing_flow': normalizing_flow.state_dict()}
                torch.save(checkpoint, f'model/task_2/{args.model}_{epoch}_{losses[epoch]:.4f}.pt')

        # Plot losses
        info_log('Plot losses ...', args.verbosity)
        plot_losses(losses=(losses,), labels=['loss'], task='task_2', model=args.model)
    else:
        # Start inferring
        info_log('Start inferring', args.verbosity)
        inference_celeb(train_dataset=train_dataset,
                        normalizing_flow=normalizing_flow,
                        num_classes=num_classes,
                        args=args,
                        training_device=training_device)


def main() -> None:
    """
    Main function
    :return: None
    """
    # Get training device
    training_device = device('cuda' if cuda.is_available() else 'cpu')

    # Parse arguments
    args = parse_arguments()
    info_log(f'Batch size: {args.batch_size}', args.verbosity)
    info_log(f'Image size: {args.image_size}', args.verbosity)
    info_log(f'Dimension of the hidden layers in normalizing flow: {args.width}', args.verbosity)
    info_log(f'Depth of the normalizing flow: {args.depth}', args.verbosity)
    info_log(f'Number of levels in normalizing flow: {args.num_levels}', args.verbosity)
    info_log(f'Clip gradients at specific value: {args.grad_value_clip}', args.verbosity)
    info_log(f"Clip gradients' norm at specific value: {args.grad_norm_clip}", args.verbosity)
    info_log(f'Learning rate of discriminator: {args.learning_rate_discriminator}', args.verbosity)
    info_log(f'Learning rate of generator: {args.learning_rate_generator}', args.verbosity)
    info_log(f'Learning rate of normalizing flow: {args.learning_rate_normalizing_flow}', args.verbosity)
    info_log(f'Number of epochs: {args.epochs}', args.verbosity)
    info_log(f'Perform task: {args.task}', args.verbosity)
    info_log(f'Which model will be used: {args.model}', args.verbosity)
    info_log(f'Only inference or not: {True if args.inference else False}', args.verbosity)
    info_log(f'Training device: {training_device}', args.verbosity)

    # Read data
    info_log('Read data ...', args.verbosity)

    if args.model == 'GLOW':
        if args.task == 1:
            transformation = transforms.Compose([transforms.RandomCrop(240),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.Resize(args.image_size),
                                                 transforms.ToTensor()])
        else:
            transformation = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                 transforms.Resize(args.image_size),
                                                 transforms.ToTensor()])
    else:
        transformation = transforms.Compose([transforms.RandomCrop(240),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.Resize(args.image_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if args.task == 1:
        train_dataset = ICLEVRLoader(root_folder='data/task_1/', trans=transformation, mode='train')
        test_dataset = ICLEVRLoader(root_folder='data/task_1/', mode='test')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        num_classes = train_dataset.num_classes
    else:
        train_dataset = CelebALoader(root_folder='data/task_2/', trans=transformation)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        num_classes = train_dataset.num_classes

    # Setup evaluator
    evaluator = EvaluationModel(training_device=training_device)

    # Create directories
    create_directories()

    if args.task == 1:
        if args.model == 'GLOW':
            train_and_evaluate_cnf(train_dataset=train_dataset,
                                   train_loader=train_loader,
                                   test_loader=test_loader,
                                   evaluator=evaluator,
                                   num_classes=num_classes,
                                   args=args,
                                   training_device=training_device)
        else:
            train_and_evaluate_cgan(train_dataset=train_dataset,
                                    train_loader=train_loader,
                                    test_loader=test_loader,
                                    evaluator=evaluator,
                                    num_classes=num_classes,
                                    args=args,
                                    training_device=training_device)
    else:
        train_and_inference_celeb(train_dataset=train_dataset,
                                  train_loader=train_loader,
                                  num_classes=num_classes,
                                  args=args,
                                  training_device=training_device)


if __name__ == '__main__':
    main()
