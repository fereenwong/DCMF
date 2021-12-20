from .utils import *


class ImageDecoder(nn.Module):
    def __init__(self):

        super(ImageDecoder, self).__init__()
        channels = [64, 128, 256, 512, 512, 512]
        self.with_corr = with_corr

        # feature fusing: encoder feature and decoder feature
        self.enc_dec_fusing5 = EncDecFusing(channels[4])
        self.enc_dec_fusing4 = EncDecFusing(channels[3])
        self.enc_dec_fusing3 = EncDecFusing(channels[2])
        self.enc_dec_fusing2 = EncDecFusing(channels[1])
        self.enc_dec_fusing1 = EncDecFusing(channels[0])

        # correlation calculate
        self.corr_layer6 = CorrelationLayer(feat_channel=channels[5])
        self.corr_layer5 = CorrelationLayer(feat_channel=channels[4])
        self.corr_layer4 = CorrelationLayer(feat_channel=channels[3])
        self.corr_layer3 = CorrelationLayer(feat_channel=channels[2])
        self.corr_layer2 = CorrelationLayer(feat_channel=channels[1])
        self.corr_layer1 = CorrelationLayer(feat_channel=channels[0])

        # decoder modules
        self.decoder6 = decoder(channels[5], channels[4])
        self.decoder5 = decoder(channels[4], channels[3])
        self.decoder4 = decoder(channels[3], channels[2])
        self.decoder3 = decoder(channels[2], channels[1])
        self.decoder2 = decoder(channels[1], channels[0])
        self.decoder1 = decoder(channels[0], channels[0])

        # predict layers
        self.conv_loss6 = nn.Conv2d(in_channels=channels[4], out_channels=1, kernel_size=3, padding=1)
        self.conv_loss5 = nn.Conv2d(in_channels=channels[3], out_channels=1, kernel_size=3, padding=1)
        self.conv_loss4 = nn.Conv2d(in_channels=channels[2], out_channels=1, kernel_size=3, padding=1)
        self.conv_loss3 = nn.Conv2d(in_channels=channels[1], out_channels=1, kernel_size=3, padding=1)
        self.conv_loss2 = nn.Conv2d(in_channels=channels[0], out_channels=1, kernel_size=3, padding=1)
        self.conv_loss1 = nn.Conv2d(in_channels=channels[0], out_channels=1, kernel_size=3, padding=1)

    def forward(self, image_feats, depth_feats):

        encoder_conv1, encoder_conv2, encoder_conv3, encoder_conv4, encoder_conv5, x6 = image_feats
        depth_feat1, depth_feat2, depth_feat3, depth_feat4, depth_feat5, depth_feat6 = depth_feats

        lamda = 0.01
        corr_fea_6, depth_feat6 = self.corr_layer6((x6, depth_feat6))
        dec_fea_6 = self.decoder6(corr_fea_6, depth_feat6 * lamda)
        mask6 = self.conv_loss6(dec_fea_6)

        fus_fea_5 = self.enc_dec_fusing5(encoder_conv5, dec_fea_6)
        corr_fea_5, depth_feat5 = self.corr_layer5((fus_fea_5, depth_feat5))
        dec_fea_5 = self.decoder5(corr_fea_5, depth_feat5 * lamda)
        mask5 = self.conv_loss5(dec_fea_5)

        fus_fea_4 = self.enc_dec_fusing4(encoder_conv4, dec_fea_5)
        corr_fea_4, depth_feat4 = self.corr_layer4((fus_fea_4, depth_feat4))
        dec_fea_4 = self.decoder4(corr_fea_4, depth_feat4 * lamda)
        mask4 = self.conv_loss4(dec_fea_4)

        # image (decoder3)
        fus_fea_3 = self.enc_dec_fusing3(encoder_conv3, dec_fea_4)
        corr_fea_3, depth_feat3 = self.corr_layer3((fus_fea_3, depth_feat3))
        dec_fea_3 = self.decoder3(corr_fea_3, depth_feat3 * lamda)
        mask3 = self.conv_loss3(dec_fea_3)

        # image (decoder2)
        fus_fea_2 = self.enc_dec_fusing2(encoder_conv2, dec_fea_3)
        corr_fea_2, depth_feat2 = self.corr_layer2((fus_fea_2, depth_feat2))
        dec_fea_2 = self.decoder2(corr_fea_2, depth_feat2 * lamda)
        mask2 = self.conv_loss2(dec_fea_2)

        # image (decoder1)
        fus_fea_1 = self.enc_dec_fusing1(encoder_conv1, dec_fea_2)
        corr_fea_1, depth_feat1 = self.corr_layer1((fus_fea_1, depth_feat1))
        dec_fea_1 = self.decoder1(corr_fea_1, depth_feat1 * lamda)
        mask1 = self.conv_loss1(dec_fea_1)

        return mask6, mask5, mask4, mask3, mask2, mask1
