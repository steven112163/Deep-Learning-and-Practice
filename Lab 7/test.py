from dcgan import DCGenerator
from sagan import SAGenerator
from srgan import SRGenerator
from normalizing_flow import CGlow
from task_2_dataset import CelebALoader
from evaluator import EvaluationModel
from util import debug_log
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image, make_grid
from torch import device
from typing import Tuple, Optional
from argparse import Namespace
import torch


def test_cgan(data_loader: DataLoader,
              generator: Optional[DCGenerator or SAGenerator or SRGenerator],
              num_classes: int,
              epoch: int,
              evaluator: EvaluationModel,
              args: Namespace,
              training_device: device) -> Tuple[torch.Tensor, float]:
    """
    Test cGAN
    :param data_loader: test data loader
    :param generator: generator
    :param num_classes: number of classes (object IDs)
    :param epoch: current epoch
    :param args: all arguments
    :param training_device: training_device
    :return: generated images and total accuracy of all batches
    """
    generator.eval()
    total_accuracy = 0.0
    norm_image = torch.randn(0, 3, 64, 64)
    for batch_idx, batch_data in enumerate(data_loader):
        labels = batch_data
        batch_size = len(labels)
        if args.model == 'DCGAN' or args.model == 'SRGAN':
            # DCGAN or SRGAN
            labels = labels.to(training_device).type(torch.float)
        else:
            # SAGAN
            labels = labels.to(training_device).type(torch.long)

        # Generate batch of latent vectors
        if args.model == 'DCGAN':
            # DCGAN
            noise = torch.cat([
                torch.randn((batch_size, args.image_size - num_classes)),
                labels.cpu()
            ], 1).view(-1, args.image_size, 1, 1).to(training_device)
        elif args.model == 'SAGAN':
            # SAGAN
            noise = torch.randn((batch_size, args.image_size)).to(training_device)
        else:
            # SRGAN
            noise = torch.randn((batch_size, 3, args.image_size, args.image_size)).to(training_device)

        # Generate fake image batch with generator
        if args.model == 'DCGAN':
            # DCGAN
            fake_outputs = generator.forward(noise)
        else:
            # SAGAN or SRGAN
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

        debug_log(f'[{epoch + 1}/{args.epochs}][{batch_idx + 1}/{len(data_loader)}]   Accuracy: {acc}', args.verbosity)

    return norm_image, total_accuracy


def test_cnf(data_loader: DataLoader,
             normalizing_flow: CGlow,
             epoch: int,
             evaluator: EvaluationModel,
             args: Namespace,
             training_device: device) -> Tuple[torch.Tensor, float]:
    """
    Test cNF
    :param data_loader: testing data loader
    :param normalizing_flow: conditional normalizing flow model
    :param epoch: current epoch
    :param evaluator: evaluator
    :param args: all arguments
    :param training_device: training device
    :return: generated images and total accuracy
    """
    normalizing_flow.eval()
    total_accuracy = 0.0
    generated_image = torch.randn(0, 3, 64, 64)
    for batch_idx, batch_data in enumerate(data_loader):
        labels = batch_data
        labels = labels.to(training_device).type(torch.float)
        batch_size = len(labels)

        z = torch.randn((batch_size, 3, 64, 64), dtype=torch.float, device=training_device)
        fake_images, _ = normalizing_flow(z, labels, reverse=True)
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

        debug_log(f'[{epoch + 1}/{args.epochs}][{batch_idx + 1}/{len(data_loader)}]   Accuracy: {acc}', args.verbosity)

    return generated_image, total_accuracy


def inference_celeb(train_dataset: CelebALoader,
                    normalizing_flow: CGlow,
                    num_classes: int,
                    args: Namespace,
                    training_device: device) -> None:
    """
    Use cNF to inference celebrity data with 3 applications
    :param train_dataset: training dataset
    :param normalizing_flow: conditional normalizing flow model
    :param num_classes: number of classes (attributes)
    :param args: all arguments
    :param training_device: training device
    :return: None
    """
    normalizing_flow.eval()

    # Application 1
    # Get labels for inference
    debug_log(f'Perform app 1', args.verbosity)
    labels = torch.rand(0, num_classes)
    for idx in range(32):
        _, label = train_dataset[idx]
        labels = torch.cat([labels, torch.from_numpy(label).view(1, 40)], 0)
    labels = labels.to(training_device).type(torch.float)

    # Generate random latent code
    z = torch.randn((32, 3, 64, 64), dtype=torch.float, device=training_device)

    # Produce fake images
    fake_images, _ = normalizing_flow(z, labels, reverse=True)
    fake_images = torch.sigmoid(fake_images)

    # Save fake images for application 1
    generated_images = torch.randn(0, 3, 64, 64)
    for fake_image in fake_images:
        n_image = fake_image.cpu().detach()
        generated_images = torch.cat([generated_images, n_image.view(1, 3, 64, 64)], 0)
    save_image(make_grid(generated_images, nrow=8), f'figure/task_2/{args.model}_app_1.jpg')

    # Application 2
    # Get 2 images to perform linear interpolation
    debug_log(f'Perform app 2', args.verbosity)
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
        first_z, _ = normalizing_flow(first_image, first_label)
        second_z, _ = normalizing_flow(second_image, second_label)

        # Compute interval
        interval_z = (second_z - first_z) / 10.0
        interval_label = (second_label - first_label) / 10.0

        # Generate linear images
        for num_of_intervals in range(11):
            image, _ = normalizing_flow(first_z + num_of_intervals * interval_z,
                                        first_label + num_of_intervals * interval_label,
                                        reverse=True)
            image = torch.sigmoid(image)
            linear_images = torch.cat([linear_images, image.cpu().detach().view(1, 3, 64, 64)], 0)
    save_image(make_grid(linear_images, nrow=9), f'figure/task_2/{args.model}_app_2.jpg')

    # Application 3
    # Get a image and labels with negative/positive smiling
    debug_log(f'Perform app 3', args.verbosity)
    image, label = train_dataset[1]
    image = image.to(training_device).type(torch.float).view(1, 3, 64, 64)
    label = torch.from_numpy(label).to(training_device).type(torch.float).view(1, 40)
    neg_label = torch.clone(label)
    neg_label[0, 4] = -1.
    neg_label[0, 31] = -1.
    pos_label = torch.clone(label)
    pos_label[0, 4] = 1.
    pos_label[0, 31] = 1.

    # Generate latent code
    neg_z, _ = normalizing_flow(image, neg_label)
    pos_z, _ = normalizing_flow(image, pos_label)

    # Compute interval
    interval_z = (pos_z - neg_z) / 10.0
    interval_label = (pos_label - neg_label) / 10.0

    # Generate manipulated images
    manipulated_images = torch.randn(0, 3, 64, 64)
    for num_of_intervals in range(11):
        image, _ = normalizing_flow(neg_z + num_of_intervals * interval_z,
                                    neg_label + num_of_intervals * interval_label,
                                    reverse=True)
        image = torch.sigmoid(image)
        manipulated_images = torch.cat([manipulated_images, image.cpu().detach().view(1, 3, 64, 64)], 0)
    save_image(make_grid(manipulated_images, nrow=10), f'figure/task_2/{args.model}_app_3.jpg')
