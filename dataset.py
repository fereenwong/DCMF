from PIL import Image
import torch
from torch.utils import data
import transforms as trans
from torchvision import transforms
import random
from parameter import *


def get_list(path, file_list=None):
    if file_list is None:
        file_list = []
    for file in os.listdir(path):
        file_list.append(os.path.join(path, file))
    return file_list


def load_list(file):

    with open(file) as f:
        lines = f.read().splitlines()

    files = []
    depths = []
    labels = []

    for line in lines:
        files.append(line.split(' ')[0])
        depths.append(line.split(' ')[1])
        labels.append(line.split(' ')[2])

    return files, depths, labels


def load_test_list(file):

    with open(file) as f:
        lines = f.read().splitlines()

    files = []
    depths = []
    for line in lines:
        files.append(line.split(' ')[0])
        depths.append(line.split(' ')[1])

    return files, depths


class ImageData(data.Dataset):
    def __init__(self, data_root, dataset_list, transform, depth_transform, t_transform,
                 label_32_transform, label_64_transform, label_128_transform, mode):

        self.image_list, self.depth_list, self.label_list = [], [], []
        self.load_data(data_root, dataset_list)

        self.transform = transform
        self.depth_transform = depth_transform
        self.t_transform = t_transform
        self.label_32_transform = label_32_transform
        self.label_64_transform = label_64_transform
        self.label_128_transform = label_128_transform
        self.mode = mode

    def load_data(self, data_root, dataset_list):
        for dataset in dataset_list:
            dataset_path = os.path.join(data_root, dataset)
            for image in os.listdir(os.path.join(dataset_path, 'RGB')):
                self.image_list.append(os.path.join(dataset_path, 'RGB', image))
                self.depth_list.append(os.path.join(dataset_path, 'depth', image.replace('.jpg', '.png')))
                self.label_list.append(os.path.join(dataset_path, 'GT', image.replace('.jpg', '.png')))

    def __getitem__(self, item):
        fn = self.image_list[item].split('/')

        filename = fn[-1]
        image = Image.open(self.image_list[item]).convert('RGB')
        image_w, image_h = int(image.size[0]), int(image.size[1])
        depth = Image.open(self.depth_list[item]).convert('L')

        # data augmentation
        if self.mode == 'train':

            label = Image.open(self.label_list[item]).convert('L')
            random_size = scale_size

            new_img = trans.Scale((random_size, random_size))(image)
            new_depth = trans.Scale((random_size, random_size))(depth)
            new_label = trans.Scale((random_size, random_size), interpolation=Image.NEAREST)(label)

            # random crop
            w, h = new_img.size
            if w != img_size and h != img_size:
                x1 = random.randint(0, w - img_size)
                y1 = random.randint(0, h - img_size)
                new_img = new_img.crop((x1, y1, x1 + img_size, y1 + img_size))
                new_depth = new_depth.crop((x1, y1, x1 + img_size, y1 + img_size))
                new_label = new_label.crop((x1, y1, x1 + img_size, y1 + img_size))

            # random flip
            if random.random() < 0.5:
                new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
                new_depth = new_depth.transpose(Image.FLIP_LEFT_RIGHT)
                new_label = new_label.transpose(Image.FLIP_LEFT_RIGHT)

            new_img = self.transform(new_img)
            new_depth = self.depth_transform(new_depth)

            label_256 = self.t_transform(new_label)
            if self.label_32_transform is not None and self.label_64_transform is not None and self.label_128_transform is\
                    not None:
                label_32 = self.label_32_transform(new_label)
                label_64 = self.label_64_transform(new_label)
                label_128 = self.label_128_transform(new_label)
                return new_img, new_depth, label_256, label_32, label_64, label_128, filename
        else:

            image = self.transform(image)
            depth = self.depth_transform(depth)
            label = Image.open(self.label_list[item]).convert('L')
            label = self.t_transform(label)

            return image, depth, label, image_w, image_h, self.image_list[item]

    def __len__(self):
        return len(self.image_list)


def get_loader(data_root, datasets, img_size, batch_size, mode='train', num_thread=1):
    shuffle = False

    mean_bgr = torch.Tensor(3, img_size, img_size)
    mean_bgr[0, :, :] = 104.008  # B
    mean_bgr[1, :, :] = 116.669  # G
    mean_bgr[2, :, :] = 122.675  # R

    depth_mean_bgr = torch.Tensor(1, img_size, img_size)
    depth_mean_bgr[0, :, :] = 115.8695

    if mode == 'train':
        transform = trans.Compose([
            # trans.ToTensor  image -> [0,255]
            trans.ToTensor_BGR(),
            trans.Lambda(lambda x: x - mean_bgr)
        ])

        depth_transform = trans.Compose([
            # trans.ToTensor  image -> [0,255]
            trans.ToTensor(),
            trans.Lambda(lambda x: x - depth_mean_bgr)
        ])

        t_transform = trans.Compose([
            # transform.ToTensor  label -> [0,1]
            transforms.ToTensor(),
        ])
        label_32_transform = trans.Compose([
            trans.Scale((img_size // 8, img_size // 8), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_64_transform = trans.Compose([
            trans.Scale((img_size // 4, img_size // 4), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_128_transform = trans.Compose([
            trans.Scale((img_size // 2, img_size // 2), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        shuffle = True
    else:
        transform = trans.Compose([
            trans.Scale((img_size, img_size)),
            trans.ToTensor_BGR(),
            trans.Lambda(lambda x: x - mean_bgr)
        ])

        depth_transform = trans.Compose([
            trans.Scale((img_size, img_size)),
            trans.ToTensor(),
            trans.Lambda(lambda x: x - depth_mean_bgr)
        ])

        t_transform = trans.Compose([
            # trans.Scale((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
    if mode == 'train':
        dataset = ImageData(data_root, datasets, transform, depth_transform, t_transform,
                            label_32_transform, label_64_transform, label_128_transform, mode)
    else:
        dataset = ImageData(data_root, datasets, transform, depth_transform, t_transform,
                            label_32_transform=None, label_64_transform=None, label_128_transform=None, mode=mode)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread)
    return data_loader

