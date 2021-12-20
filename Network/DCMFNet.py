from EncDec import ImageEncoder
from EncDec import ImageDecoder
from EncDec import DepthEncoder
import torch.nn as nn


class DCMFNet(nn.Module):
    def __init__(self):
        super(DCMFNet, self).__init__()

        # encoder part
        self.ImageEncoder = ImageEncoder()
        self.DepthEnc = DepthEncoder(strides=[1, 2, 4, 8, 8, 8],
                                     out_channels=[64, 128, 256, 512, 512, 512])

        # decoder part
        self.ImageDecoder = ImageDecoder()

    def forward(self, image_Input, depth_Input):

        image_feat = self.ImageEncoder(image_Input)
        depth_feat = self.DepthEnc(depth_Input)

        outputs_image = self.ImageDecoder(image_feat, depth_feat)
        return outputs_image

    def init_parameters(self, pretrain_vgg16_1024):
        # init all layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # load rgb encoder parameters
        rgb_conv_blocks = [self.ImageEncoder.conv1,
                           self.ImageEncoder.conv2,
                           self.ImageEncoder.conv3,
                           self.ImageEncoder.conv4,
                           self.ImageEncoder.conv5,
                           self.ImageEncoder.fc6]

        listkey = [['conv1_1', 'conv1_2'], ['conv2_1', 'conv2_2'], ['conv3_1', 'conv3_2', 'conv3_3'],
                   ['conv4_1', 'conv4_2', 'conv4_3'], ['conv5_1', 'conv5_2', 'conv5_3'], ['fc6']]

        for idx, conv_block in enumerate(rgb_conv_blocks):
            num_conv = 0
            for l2 in conv_block:
                if isinstance(l2, nn.Conv2d):
                    if 'fc' in listkey[idx][num_conv]:
                        l2.weight.data = pretrain_vgg16_1024[str(listkey[idx][num_conv]) + '.weight'][:512, :512]
                        l2.bias.data = pretrain_vgg16_1024[str(listkey[idx][num_conv])
                                                           + '.bias'][:, :, :, :512].squeeze()
                    else:
                        l2.weight.data = pretrain_vgg16_1024[str(listkey[idx][num_conv]) + '.weight']
                        l2.bias.data = pretrain_vgg16_1024[str(listkey[idx][num_conv]) + '.bias'].squeeze(
                            0).squeeze(
                            0).squeeze(0).squeeze(0)
                    num_conv += 1
        return self
