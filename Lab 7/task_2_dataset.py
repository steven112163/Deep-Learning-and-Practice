from torch.utils import data
from PIL import Image
import numpy as np
import os


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
    return img_list, np.array(label_list)


class CelebALoader(data.Dataset):
    def __init__(self, root_folder, trans=None, cond=False):
        self.root_folder = root_folder
        assert os.path.isdir(self.root_folder), '{} is not a valid directory'.format(self.root_folder)

        self.img_list, self.label_list = get_celebrity_data(self.root_folder)

        print("> Found %d images..." % (len(self.img_list)))

        self.cond = cond
        self.transform = trans
        self.num_classes = 40

    def __len__(self):
        """
        Return the size of dataset
        :return: size of dataset
        """
        return len(self.label_list)

    def __getitem__(self, index):
        """
        Get current data
        :param index: index of training data
        :return: data
        """
        img_path = self.root_folder + 'CelebA-HQ-img/' + self.img_list[index]
        label = self.label_list[index]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label
