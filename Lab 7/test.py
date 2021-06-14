from dcgan import DCGenerator
from sagan import SAGenerator
from glow import CGlow
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
              generator: Optional[DCGenerator or SAGenerator],
              num_classes: int,
              epoch: int,
              evaluator: EvaluationModel,
              args: Namespace,
              training_device: device) -> Tuple[torch.Tensor, float]:
    """
    Test cGAN
    :param data_loader: Testing data loader
    :param generator: Generator
    :param num_classes: Number of classes (object IDs)
    :param epoch: Current epoch
    :param args: All arguments
    :param training_device: Training device
    :return: Generated images and total accuracy of all batches
    """
    generator.eval()
    total_accuracy = 0.0
    norm_image = torch.randn(0, 3, args.image_size, args.image_size)
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
                torch.randn((batch_size, args.image_size)),
                labels.cpu()
            ], 1).view(-1, args.image_size + num_classes, 1, 1).to(training_device)
        elif args.model == 'SAGAN':
            # SAGAN
            noise = torch.randn((batch_size, args.image_size)).to(training_device)
        else:
            # SRGAN
            noise = torch.randn((batch_size, 3, args.image_size, args.image_size)).to(training_device)

        # Generate fake image batch with generator
        if args.model == 'DCGAN':
            # DCGAN
            with torch.no_grad():
                fake_outputs = generator.forward(noise)
        else:
            # SAGAN or SRGAN
            with torch.no_grad():
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
            norm_image = torch.cat([norm_image, n_image.view(1, 3, args.image_size, args.image_size)], 0)

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
    :param data_loader: Testing data loader
    :param normalizing_flow: Conditional normalizing flow model
    :param epoch: Current epoch
    :param evaluator: Evaluator
    :param args: All arguments
    :param training_device: Training device
    :return: Generated images and total accuracy
    """
    normalizing_flow.eval()
    total_accuracy = 0.0
    generated_image = torch.randn(0, 3, args.image_size, args.image_size)
    for batch_idx, batch_data in enumerate(data_loader):
        labels = batch_data
        labels = labels.to(training_device).type(torch.float)

        with torch.no_grad():
            fake_images, _, _ = normalizing_flow.forward(x=None, x_label=labels, reverse=True)

        transformed_images = torch.randn(0, 3, args.image_size, args.image_size)
        transformation = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        for fake_image in fake_images:
            n_image = fake_image.cpu().detach()
            transformed_images = torch.cat([transformed_images,
                                            transformation(n_image).view(1, 3, args.image_size, args.image_size)], 0)
        transformed_images = transformed_images.to(training_device)

        acc = evaluator.eval(transformed_images, labels)
        total_accuracy += acc

        for fake_image in fake_images:
            n_image = fake_image.cpu().detach()
            generated_image = torch.cat([generated_image, n_image.view(1, 3, args.image_size, args.image_size)], 0)

        debug_log(f'[{epoch + 1}/{args.epochs}][{batch_idx + 1}/{len(data_loader)}]   Accuracy: {acc}', args.verbosity)

    return generated_image, total_accuracy


def inference_celeb(data_loader: DataLoader,
                    train_dataset: CelebALoader,
                    normalizing_flow: CGlow,
                    num_classes: int,
                    args: Namespace,
                    training_device: device) -> None:
    """
    Use cNF to inference celebrity data with 3 applications
    :param data_loader: Training data loader
    :param train_dataset: Training dataset
    :param normalizing_flow: Conditional normalizing flow model
    :param num_classes: Number of classes (attributes)
    :param args: All arguments
    :param training_device: Training device
    :return: None
    """
    normalizing_flow.eval()

    # Application 1
    debug_log(f'Perform app 1', args.verbosity)
    application_one(train_dataset=train_dataset,
                    normalizing_flow=normalizing_flow,
                    num_classes=num_classes,
                    args=args,
                    training_device=training_device)

    # Application 2
    debug_log(f'Perform app 2', args.verbosity)
    application_two(train_dataset=train_dataset,
                    normalizing_flow=normalizing_flow,
                    args=args,
                    training_device=training_device)

    # Application 3
    debug_log(f'Perform app 3', args.verbosity)
    application_three(data_loader=data_loader,
                      train_dataset=train_dataset,
                      normalizing_flow=normalizing_flow,
                      args=args,
                      training_device=training_device)


def application_one(train_dataset: CelebALoader,
                    normalizing_flow: CGlow,
                    num_classes: int,
                    args: Namespace,
                    training_device: device) -> None:
    """
    Application 1
    :param train_dataset: Training dataset
    :param normalizing_flow: Conditional normalizing flow model
    :param num_classes: Number of classes (attributes)
    :param args: All arguments
    :param training_device: Training device
    :return: None
    """
    # Get labels for inference
    labels = torch.rand(0, num_classes)
    for idx in range(32):
        _, label = train_dataset[idx]
        labels = torch.cat([labels, torch.from_numpy(label).view(1, 40)], 0)
    labels = labels.to(training_device).type(torch.float)

    # Produce fake images
    with torch.no_grad():
        fake_images, _, _ = normalizing_flow.forward(x=None, x_label=labels, reverse=True)

    # Save fake images for application 1
    generated_images = torch.randn(0, 3, args.image_size, args.image_size)
    for fake_image in fake_images:
        n_image = fake_image.cpu().detach()
        generated_images = torch.cat([generated_images, n_image.view(1, 3, args.image_size, args.image_size)], 0)
    save_image(make_grid(generated_images, nrow=8), f'figure/task_2/{args.model}_app_1.jpg')


def application_two(train_dataset: CelebALoader,
                    normalizing_flow: CGlow,
                    args: Namespace,
                    training_device: device) -> None:
    """
    Application 2
    :param train_dataset: Training dataset
    :param normalizing_flow: Conditional normalizing flow model
    :param args: All arguments
    :param training_device: Training device
    :return: None
    """
    # Get 2 images to perform linear interpolation
    linear_images = torch.randn(0, 3, args.image_size, args.image_size)
    for idx in range(5):
        # Get first image and label
        first_image, first_label = train_dataset[idx]
        first_image = first_image.to(training_device).type(torch.float).view(1, 3, args.image_size, args.image_size)
        first_label = torch.from_numpy(first_label).to(training_device).type(torch.float).view(1, 40)

        # Get second image and label
        second_image, second_label = train_dataset[idx + 5]
        second_image = second_image.to(training_device).type(torch.float).view(1, 3, args.image_size, args.image_size)
        second_label = torch.from_numpy(second_label).to(training_device).type(torch.float).view(1, 40)

        # Generate latent code
        with torch.no_grad():
            first_z, _, _ = normalizing_flow.forward(x=first_image, x_label=first_label)
            second_z, _, _ = normalizing_flow.forward(x=second_image, x_label=second_label)

        # Compute interval
        interval_z = (second_z - first_z) / 8.0
        interval_label = (second_label - first_label) / 8.0

        # Generate linear images
        for num_of_intervals in range(9):
            with torch.no_grad():
                image, _, _ = normalizing_flow.forward(x=first_z + num_of_intervals * interval_z,
                                                       x_label=first_label + num_of_intervals * interval_label,
                                                       reverse=True)
            linear_images = torch.cat([linear_images,
                                       image.cpu().detach().view(1, 3, args.image_size, args.image_size)], 0)
    save_image(make_grid(linear_images, nrow=9), f'figure/task_2/{args.model}_app_2.jpg')


def application_three(data_loader: DataLoader,
                      train_dataset: CelebALoader,
                      normalizing_flow: CGlow,
                      args: Namespace,
                      training_device: device) -> None:
    """
    Application three
    :param data_loader: Training data loader
    :param train_dataset: Training dataset
    :param normalizing_flow: Conditional normalizing flow model
    :param args: All arguments
    :param training_device: Training device
    :return: None
    """
    # Get a image and labels with negative/positive smiling/bald
    image, label = train_dataset[1]
    image = image.to(training_device).type(torch.float).view(1, 3, args.image_size, args.image_size)
    label = torch.from_numpy(label).to(training_device).type(torch.float).view(1, 40)
    with torch.no_grad():
        latent, _, _ = normalizing_flow.forward(x=image, x_label=label)

    # Get negative labels
    neg_smiling_label = torch.clone(label)
    neg_bald_label = torch.clone(label)
    neg_smiling_label[0, 31] = -1.
    neg_bald_label[0, 4] = -1.

    # Get positive labels
    pos_smiling_label = torch.clone(label)
    pos_bald_label = torch.clone(label)
    pos_smiling_label[0, 31] = 1.
    pos_bald_label[0, 4] = 1.

    # Compute conditional interval
    interval_smiling_label = (pos_smiling_label - neg_smiling_label) / 4.0
    interval_bald_label = (pos_bald_label - neg_bald_label) / 4.0

    # Generate manipulated images
    manipulated_images = torch.randn(0, 3, args.image_size, args.image_size)

    # Generate smiling
    manipulated_images = generate_manipulated_images(data_loader=data_loader,
                                                     normalizing_flow=normalizing_flow,
                                                     latent=latent,
                                                     neg_label=neg_smiling_label,
                                                     interval_label=interval_smiling_label,
                                                     manipulated_images=manipulated_images,
                                                     idx=31,
                                                     args=args,
                                                     training_device=training_device)

    # Generate bald
    manipulated_images = generate_manipulated_images(data_loader=data_loader,
                                                     normalizing_flow=normalizing_flow,
                                                     latent=latent,
                                                     neg_label=neg_bald_label,
                                                     interval_label=interval_bald_label,
                                                     manipulated_images=manipulated_images,
                                                     idx=4,
                                                     args=args,
                                                     training_device=training_device)

    save_image(make_grid(manipulated_images, nrow=5), f'figure/task_2/{args.model}_app_3.jpg')


def generate_manipulated_images(data_loader: DataLoader,
                                normalizing_flow: CGlow,
                                latent: torch.Tensor,
                                neg_label: torch.Tensor,
                                interval_label: torch.Tensor,
                                manipulated_images: torch.Tensor,
                                idx: int,
                                args: Namespace,
                                training_device: device) -> torch.Tensor:
    """
    Generate images with manipulated attribute
    :param data_loader: Training data loader
    :param normalizing_flow: Conditional normalizing flow model
    :param latent: Latent code of the target image
    :param neg_label: Label with negative target attribute
    :param interval_label: Interval from negative label to positive label
    :param manipulated_images: Tensor of manipulated images
    :param idx: Index of the target attribute
    :param args: All arguments
    :param training_device: Training device
    :return: Manipulated images
    """
    pos_z_mean = torch.zeros(*(latent.size()), dtype=torch.float)
    neg_z_mean = torch.zeros(*(latent.size()), dtype=torch.float)
    num_pos, num_neg = 0, 0
    for images, labels in data_loader:
        images = images.to(training_device)
        labels = labels.to(training_device).type(torch.float)
        pos_indices = (labels[:, idx] == 1).nonzero(as_tuple=True)[0]
        neg_indices = (labels[:, idx] == -1).nonzero(as_tuple=True)[0]

        with torch.no_grad():
            z, _, _ = normalizing_flow.forward(x=images, x_label=labels)
        z = z.cpu().detach()

        if len(pos_indices) > 0:
            num_pos += len(pos_indices)
            pos_z_mean = (num_pos - len(pos_indices)) / num_pos * pos_z_mean + z[pos_indices].sum(dim=0) / num_pos
        if len(neg_indices) > 0:
            num_neg += len(neg_indices)
            neg_z_mean = (num_neg - len(neg_indices)) / num_neg * neg_z_mean + z[neg_indices].sum(dim=0) / num_neg
    interval_z = 1.6 * (pos_z_mean - neg_z_mean)
    interval_z = interval_z.to(training_device)

    alphas = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for num_of_intervals, alpha in enumerate(alphas):
        with torch.no_grad():
            image, _, _ = normalizing_flow.forward(x=latent + alpha * interval_z,
                                                   x_label=neg_label + num_of_intervals * interval_label,
                                                   reverse=True)
        manipulated_images = torch.cat([manipulated_images,
                                        image.cpu().detach().view(1, 3, args.image_size, args.image_size)], 0)

    return manipulated_images
