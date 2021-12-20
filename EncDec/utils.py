import torch.nn as nn
import torch
import torch.nn.functional as F
from parameter import *
from .pacconv import PacConv2d


def corr_fun(Kernel_tmp, Feature):
    size = Kernel_tmp.size()
    CORR = []
    Kernel = []
    for i in range(len(Feature)):
        ker = Kernel_tmp[i:i + 1]
        fea = Feature[i:i + 1]
        ker = ker.view(size[1], size[2] * size[3]).transpose(0, 1)
        ker = ker.unsqueeze(2).unsqueeze(3)

        co = F.conv2d(fea, ker.contiguous())
        CORR.append(co)
        ker = ker.unsqueeze(0)
        Kernel.append(ker)
    corr = torch.cat(CORR, 0)
    Kernel = torch.cat(Kernel, 0)
    return corr, Kernel


class CorrelationLayer(nn.Module):
    def __init__(self, feat_channel):
        super(CorrelationLayer, self).__init__()

        self.pool_layer = nn.AdaptiveAvgPool2d((corr_size, corr_size))

        self.corr_reduce = nn.Sequential(
            nn.Conv2d(corr_size * corr_size, feat_channel, kernel_size=1),
            nn.InstanceNorm2d(feat_channel),
            nn.ReLU(),
            nn.Conv2d(feat_channel, feat_channel, 3, padding=1),
        )
        self.Dnorm = nn.InstanceNorm2d(feat_channel)

        self.feat_adapt = nn.Sequential(
            nn.Conv2d(feat_channel * 2, feat_channel, 1),
            nn.InstanceNorm2d(feat_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # calculate correlation map
        RGB_feat_downsize = F.normalize(self.pool_layer(x[0]))
        RGB_feat_norm = F.normalize(x[0])
        RGB_corr, _ = corr_fun(RGB_feat_downsize, RGB_feat_norm)

        Depth_feat_downsize = F.normalize(self.pool_layer(x[1]))
        Depth_feat_norm = F.normalize(x[1])
        Depth_corr, _ = corr_fun(Depth_feat_downsize, Depth_feat_norm)

        corr = (RGB_corr + Depth_corr) / 2
        Red_corr = self.corr_reduce(corr)

        # beta cond
        new_feat = torch.cat([x[0], Red_corr], 1)
        new_feat = self.feat_adapt(new_feat)

        Depth_feat = self.Dnorm(x[1])
        return new_feat, Depth_feat


class EncDecFusing(nn.Module):
    def __init__(self, in_channels):
        super(EncDecFusing, self).__init__()
        self.enc_fea_proc = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.fusing_layer = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, enc_fea, dec_fea):
        enc_fea = self.enc_fea_proc(enc_fea)

        if dec_fea.size(2) != enc_fea.size(2):
            dec_fea = F.upsample(dec_fea, size=[enc_fea.size(2), enc_fea.size(3)], mode='bilinear',
                                 align_corners=True)

        enc_fea = torch.cat([enc_fea, dec_fea], dim=1)
        output = self.fusing_layer(enc_fea)
        return output


class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()

        if with_pac:
            self.pac = PacConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            self.norm = nn.InstanceNorm2d(in_channels)
        self.decoding = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, feat, guide):
        if with_pac:
            feat = self.norm(self.pac(feat, guide)).relu()
        output = self.decoding(feat)
        return output