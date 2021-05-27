from torch import device
import torch
import torch.nn as nn
import torchvision.models as models

'''===============================================================
1. Title:     

DLP spring 2021 Lab7 classifier

2. Purpose:

For computing the classification accuracy.

3. Details:

The model is based on ResNet18 with only changing the
last linear layer. The model is trained on iclevr dataset
with 1 to 5 objects and the resolution is the up-sampled 
64x64 images from 32x32 images.

It will capture the top k highest accuracy indexes on generated
images and compare them with ground truth labels.

4. How to use

You should call eval(images, labels) and to get total accuracy.
images shape: (batch_size, 3, 64, 64)
labels shape: (batch_size, 24) where labels are one-hot vectors
e.g. [[1,1,0,...,0],[0,1,1,0,...],...]

==============================================================='''


class EvaluationModel:
    def __init__(self, training_device: device):
        checkpoint = torch.load('data/task_1/classifier_weight.pth', map_location=training_device)
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(512, 24),
            nn.Sigmoid()
        )
        self.resnet18.load_state_dict(checkpoint['model'])
        self.resnet18 = self.resnet18.to(training_device)
        self.resnet18.eval()
        self.class_num = 24

    @staticmethod
    def compute_acc(out: torch.Tensor, one_hot_labels: torch.Tensor):
        """
        Compute accuracy for one_hot_labels based on out
        :param out: output from ResNet18
        :param one_hot_labels: one_hot_labels from generator
        :return: accuracy
        """
        batch_size = out.size(0)
        acc = 0
        total = 0
        for i in range(batch_size):
            k = int(one_hot_labels[i].sum().item())
            total += k
            out_v, out_i = out[i].topk(k)
            lv, li = one_hot_labels[i].topk(k)
            for j in out_i:
                if j in li:
                    acc += 1
        return acc / total

    def eval(self, images: torch.Tensor, labels: torch.Tensor):
        """
        Evaluate labels generated from the generator
        :param images: images from generator
        :param labels: labels from generator
        :return: accuracy
        """
        with torch.no_grad():
            # Your image shape should be (batch, 3, 64, 64)
            out = self.resnet18(images)
            acc = self.compute_acc(out.cpu(), labels.cpu())
            return acc
