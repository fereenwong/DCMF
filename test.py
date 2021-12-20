# -*-coding:utf-8-*-
import os
import torch
import torch.nn.functional as F
from dataset import get_loader
import transforms as trans
from torchvision import transforms
import argparse
from parameter import *
from Network import DCMFNet
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('checkpoint', help='the checkpoint to be tested')
    parser.add_argument('data_root', help='dataset root path')
    parser.add_argument('--save_results_dir', help='where to save tested results', default='./output')
    args = parser.parse_args()
    return args


def test_net(net, args):
    test_loader = get_loader(args.data_root, test_datasets, img_size, 1, mode='test', num_thread=1)

    for data_batch in tqdm(test_loader):
        images, depths, labels, image_w, image_h, image_path = data_batch
        images, depths = images.cuda(), depths.cuda()

        outputs_image = net(images, depths)

        _, _, _, _, _, imageBran_output = outputs_image

        image_w, image_h = int(image_w[0]), int(image_h[0])

        output_imageBran = F.sigmoid(imageBran_output)
        output_imageBran = output_imageBran.data.cpu().squeeze(0)

        transform = trans.Compose([
            transforms.ToPILImage(),
            trans.Scale((image_w, image_h))
        ])
        outputImageBranch = transform(output_imageBran)

        dataset = image_path[0].split('RGBD_Sal')[1].split('/')[1]
        filename = image_path[0].split('/')[-1].split('.')[0]

        # save image branch output
        save_test_path = os.path.join(args.save_results_dir,
                                      args.checkpoint.split('/')[-2],
                                      dataset)
        if not os.path.exists(save_test_path):
            os.makedirs(save_test_path)
        outputImageBranch.save(os.path.join(save_test_path, filename + '.png'))


if __name__ == '__main__':
    args = parse_args()

    net = DCMFNet()
    net.cuda()
    net.eval()

    net.load_state_dict(torch.load(args.checkpoint))
    print('Model loaded from {}'.format(args.checkpoint))

    test_net(net, args)