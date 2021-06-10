from torch.utils import data
from typing import Dict, List
from PIL import Image
import torchvision.transforms as transforms
import os
import json
import numpy as np


def change_labels_to_one_hot(obj: Dict[str, int], ori_label: List[List[str]]):
    """
    Change labels in each image to one hot vectors
    :param obj: ID for each label
    :param ori_label: original labels
    :return: converted labels
    """
    converted_labels = np.zeros((len(ori_label), len(obj)))
    for img_idx, labels in enumerate(ori_label):
        for label_idx, label in enumerate(labels):
            ori_label[img_idx][label_idx] = obj[label]
        tmp = np.zeros(len(obj))
        tmp[ori_label[img_idx]] = 1
        converted_labels[img_idx] = tmp

    return converted_labels


def get_iclevr_data(root_folder: str, mode: str):
    """
    Read training/testing/new_testing data from the file in root_folder
    :param root_folder: root folder containing training/testing data
    :param mode: train or test
    :return: image & labels for train, otherwise none & labels
    """
    if mode == 'train':
        training_data = json.load(open(os.path.join(root_folder, 'train.json')))
        obj = json.load(open(os.path.join(root_folder, 'objects.json')))
        img = list(training_data.keys())
        label = list(training_data.values())
        label = change_labels_to_one_hot(obj=obj, ori_label=label)
        return np.squeeze(img), np.squeeze(label)
    else:
        testing_data = json.load(open(os.path.join(root_folder, f'{mode}.json')))
        obj = json.load(open(os.path.join(root_folder, 'objects.json')))
        label = testing_data
        label = change_labels_to_one_hot(obj=obj, ori_label=label)
        return None, label


class ICLEVRLoader(data.Dataset):
    def __init__(self, root_folder: str, trans: transforms.transforms = None, cond: bool = False, mode: str = 'train'):
        self.root_folder = root_folder
        self.mode = mode
        self.img_list, self.label_list = get_iclevr_data(root_folder, mode)
        if self.mode == 'train':
            print(f'> Found {len(self.img_list)} images...')

        self.transform = trans
        self.cond = cond
        self.num_classes = 24

    def __len__(self):
        """
        Return the size of dataset
        :return: size of dataset
        """
        return len(self.label_list)

    def __getitem__(self, index: int):
        """
        Get current data
        :param index: index of training/testing data
        :return: data
        """
        if self.mode == 'train':
            img_path = self.root_folder + 'images/' + self.img_list[index]
            label = self.label_list[index]
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image, label
        else:
            return self.label_list[index]
