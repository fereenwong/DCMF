import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import datetime
import argparse
from dataset import get_loader
import math
from parameter import *
from Network import DCMFNet
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('data_root', help='dataset root path')
    parser.add_argument('--save_model_dir', help='where to save trained models', default='./models')
    args = parser.parse_args()
    return args


def model_info(model, report='summary'):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if report is 'full':
        print('%5s %80s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %80s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients' % (len(list(model.parameters())), n_p, n_g))


def train_net(net, args):
    model_info(net, 'full')
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)
    train_loader = get_loader(args.data_root, train_datasets, img_size, batch_size, mode='train',
                              num_thread=4)

    print('''
        Starting training:
            Train steps: {}
            Batch size: {}
            Learning rate: {}
            Training size: {}
        '''.format(train_steps, batch_size, lr, len(train_loader.dataset)))

    N_train = len(train_loader) * batch_size

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, adjust_lr_steps, gamma=lr_decay_gamma)

    criterion = nn.BCEWithLogitsLoss()
    whole_iter_num = 0
    iter_num = math.ceil(len(train_loader.dataset) / batch_size)

    for epoch in range(epochs):

        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        print('epoch:{0}-------lr:{1}'.format(epoch + 1, lr))

        epoch_total_loss = 0
        epoch_loss = 0
        start = datetime.datetime.now()
        for i, data_batch in enumerate(train_loader):
            if (i + 1) > iter_num: break
            images, depths, label_256, label_32, label_64, label_128, filename = data_batch
            images, depths, label_256 = images.cuda(), depths.cuda(), label_256.cuda()
            label_32, label_64, label_128 = label_32.cuda(), label_64.cuda(), label_128.cuda()

            outputs_image = net(images, depths)
            for_loss6, for_loss5, for_loss4, for_loss3, for_loss2, for_loss1 = outputs_image

            # loss
            loss6 = criterion(for_loss6, label_32)
            loss5 = criterion(for_loss5, label_32)
            loss4 = criterion(for_loss4, label_32)
            loss3 = criterion(for_loss3, label_64)
            loss2 = criterion(for_loss2, label_128)
            loss1 = criterion(for_loss1, label_256)

            total_loss = loss_weights[0] * loss1 + loss_weights[1] * loss2 + loss_weights[2] * loss3 \
                             + loss_weights[3] * loss4 + loss_weights[4] * loss5 + loss_weights[5] * loss6

            epoch_total_loss += total_loss.cpu().data.item()
            epoch_loss += loss1.cpu().data.item()
            end = datetime.datetime.now()
            eta = (end - start) * (train_steps - whole_iter_num)
            start = end
            if (whole_iter_num + 1) % 10 == 0:
                print('whole_iter_num: {0} --- {1:.4f} --- lr: {2:.4f} '.format((whole_iter_num + 1),
                                                                                (i + 1) * batch_size / N_train,
                                                                                optimizer.param_groups[0]['lr'])
                      + '--- total_loss: {0:.6f} --- loss: {1:.6f} --- eta: {2}'
                      .format(total_loss.item(), loss1.item(), eta)
                      )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=max_norm, norm_type=2)
            optimizer.step()
            schedule.step()
            whole_iter_num += 1

            if whole_iter_num == train_steps:
                torch.save(net.state_dict(),
                           args.save_model_dir + 'iterations{}.pth'.format(train_steps))
                return

        print('Epoch finished ! Loss: {}'.format(epoch_total_loss / iter_num))

        torch.save(net.state_dict(),
                   args.save_model_dir + 'MODEL_EPOCH{}.pth'.format(epoch + 1))
        print('Saved')


if __name__ == '__main__':
    args = parse_args()

    net = DCMFNet()

    # load pretrain model for image and depth encoder
    vgg_model = torch.load(load_vgg_model)
    net = net.init_parameters(vgg_model)

    net.train()
    net.cuda()
    train_net(net, args)
