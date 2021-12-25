import os
max_norm = 5
with_pac = True
with_corr = True
corr_size = 32

# train.py
img_size = 256
scale_size = 288
batch_size = 4
lr = 0.005
epochs = 200
train_steps = 20000
lr_decay_gamma = 0.1
adjust_lr_steps = [15000, ]
loss_weights = [1, 0.8, 0.8, 0.5, 0.5, 0.5]
bn_momentum = 0.001


# dataset
train_datasets = ['NJU2K/trainset', 'DUT-RGBD/trainset', 'NLPR/trainset']
test_datasets = [
    'NJU2K/testset',
    # 'DUT-RGBD/testset',
    # 'NLPR/trainset',
    # 'STERE',
    # ...
]

load_vgg_model = './pretrained_model/vgg16_20M.caffemodel.pth'
