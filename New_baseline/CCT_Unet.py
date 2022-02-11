import torch
import torch.nn as nn
import torch.nn.functional as F

from New_baseline.decoders import *


def double_conv(in_channels, out_channels, step):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True)
    )


def single_conv(in_channels, out_channels, step):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True)
    )


class CCT(nn.Module):

    def __init__(self, in_ch, width, class_no):

        super(CCT, self).__init__()
        # self.final_in = class_no
        if class_no == 2:
            self.final_in = 1
        else:

            self.final_in = class_no

        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8
        #
        self.econv0 = single_conv(in_channels=in_ch, out_channels=self.w1, step=1)
        self.econv1 = double_conv(in_channels=self.w1, out_channels=self.w2, step=2)
        self.econv2 = double_conv(in_channels=self.w2, out_channels=self.w3, step=2)
        self.econv3 = double_conv(in_channels=self.w3, out_channels=self.w4, step=2)
        self.bridge = double_conv(in_channels=self.w4, out_channels=self.w4, step=1)

        self.dconv3 = double_conv(in_channels=self.w4+self.w4, out_channels=self.w3, step=1)
        self.dconv2 = double_conv(in_channels=self.w3+self.w3, out_channels=self.w2, step=1)
        self.dconv1 = double_conv(in_channels=self.w2+self.w2, out_channels=self.w1, step=1)
        self.dconv0 = double_conv(in_channels=self.w1+self.w1, out_channels=self.w1, step=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_last = nn.Conv2d(self.w1, self.final_in, 1, bias=True)

        # feature augmentation at z, which is the output of the encoder:
        drop_decoder = DropOutDecoder(upscale=2, conv_in_ch=self.w4, num_classes=class_no)
        # context_m_decoder = ContextMaskingDecoder(upscale=2, conv_in_ch=self.w4, num_classes=class_no)
        # object_masking = ObjectMaskingDecoder(upscale=2, conv_in_ch=self.w4, num_classes=class_no)
        feature_drop = FeatureDropDecoder(upscale=2, conv_in_ch=self.w4, num_classes=class_no)
        feature_noise = FeatureNoiseDecoder(upscale=2, conv_in_ch=self.w4, num_classes=class_no)
        # vat_decoder = VATDecoder(upscale=2, conv_in_ch=self.w4, num_classes=class_no)
        self.main_decoder = MainDecoder(upscale=2, conv_in_ch=self.w4, num_classes=class_no)

        self.side_outputs = nn.ModuleList([drop_decoder,
                                           # object_masking,
                                           # context_m_decoder,
                                           feature_noise,
                                           feature_drop])

    def forward(self, x):

        side_outputs = []

        x0 = self.econv0(x)
        x1 = self.econv1(x0)
        x2 = self.econv2(x1)
        x3 = self.econv3(x2)
        x4 = self.bridge(x3)

        y = self.upsample(x4)
        if y.size()[2] != x3.size()[2]:
            diffY = torch.tensor([x3.size()[2] - y.size()[2]])
            diffX = torch.tensor([x3.size()[3] - y.size()[3]])
            y = F.pad(y, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Apply feature augmentation on z, output of the encoder:
        y_main = self.main_decoder(y)

        for decoder_side in self.side_outputs:
            # print(decoder_side)
            side_ = decoder_side(y.detach())
            side_outputs.append(side_)

        y3 = torch.cat([y, x3], dim=1)
        y3 = self.dconv3(y3)
        y2 = self.upsample(y3)
        y2 = torch.cat([y2, x2], dim=1)
        y2 = self.dconv2(y2)
        y1 = self.upsample(y2)
        y1 = torch.cat([y1, x1], dim=1)
        y1 = self.dconv1(y1)
        y0 = self.upsample(y1)
        y0 = torch.cat([y0, x0], dim=1)
        y0 = self.dconv0(y0)
        y = self.dconv_last(y0)

        return y, y_main, side_outputs