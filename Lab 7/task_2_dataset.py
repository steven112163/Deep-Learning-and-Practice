import json
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os
import numpy as np


def get_celebrity_data(root_folder):
    img_list = os.listdir(os.path.join(root_folder, 'CelebA-HQ-img'))
    label_list = []
    f = open(os.path.join(root_folder, 'CelebA-HQ-attribute-anno.txt'), 'r')
    num_images = int(f.readline()[:-1])
    attrs = f.readline()[:-1].split(' ')
    for idx in range(num_images):
        line = f.readline()[:-1].split(' ')
        label = line[2:]
        label = list(map(int, label))
        label_list.append(label)
    f.close()
    return img_list, label_list


class CelebALoader(data.Dataset):
    def __init__(self, root_folder, trans=None, cond=False):
        self.root_folder = root_folder
        assert os.path.isdir(self.root_folder), '{} is not a valid directory'.format(self.root_folder)

        self.cond = cond
        self.tranform = trans
        self.img_list, self.label_list = get_celebrity_data(self.root_folder)
        self.num_classes = 40
        print("> Found %d images..." % (len(self.img_list)))

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
