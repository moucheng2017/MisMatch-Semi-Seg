import torch
import torch.nn as nn
import torch.nn.functional as F
# =============================
# Networks
# =============================


class DMNet_seg(nn.Module):
    ''' Pytorch implementation of DMNet: difference minimization network
    for semi-supervised segmentation in medical images
    '''
    def __init__(self, in_ch, width, class_no):

        super(DMNet_seg, self).__init__()

        if class_no == 2:

            self.final_in = 1

        else:

            self.final_in = class_no

        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8

        self.econv0 = single_conv(in_channels=in_ch, out_channels=self.w1, step=1)
        self.econv1 = double_conv(in_channels=self.w1, out_channels=self.w2, step=2)
        self.econv2 = double_conv(in_channels=self.w2, out_channels=self.w3, step=2)
        self.econv3 = double_conv(in_channels=self.w3, out_channels=self.w4, step=2)
        self.bridge = double_conv(in_channels=self.w4, out_channels=self.w4, step=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.dconv_last = nn.Conv2d(self.w1, self.final_in, 1, bias=True)
        self.dconv_lastd = nn.Conv2d(self.w1, self.final_in, 1, bias=True)

        # =================
        # decoder for unet
        # =================
        self.dconv3 = double_conv(in_channels=self.w4+self.w4, out_channels=self.w3, step=1)
        self.dconv2 = double_conv(in_channels=self.w3+self.w3, out_channels=self.w2, step=1)
        self.dconv1 = double_conv(in_channels=self.w2+self.w2, out_channels=self.w1, step=1)
        self.dconv0 = double_conv(in_channels=self.w1+self.w1, out_channels=self.w1, step=1)

        # ====================
        # decoder for deeplab
        # ====================
        self.dconv3d = double_conv(in_channels=self.w4+self.w4, out_channels=self.w2, step=1)
        self.dconv1d = double_conv(in_channels=self.w2+self.w2, out_channels=self.w1, step=1)
        self.dconv0d = double_conv(in_channels=self.w1+self.w1, out_channels=self.w1, step=1)

    def forward(self, x):

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

        # ======================================
        # Decoder of DeepLab-v3: upsampling x 4
        # ======================================
        yd = torch.cat([y, x3], dim=1)
        yd = self.dconv3d(yd)
        yd = self.upsample4(yd)
        yd = torch.cat([yd, x1], dim=1)
        yd = self.dconv1(yd)
        yd = self.upsample(yd)
        yd = torch.cat([yd, x0], dim=1)
        yd = self.dconv0d(yd)
        yd = self.dconv_lastd(yd)

        # ===============================
        # Decoder of UNet: upsampling x 2
        # ===============================
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

        return y, yd


class DMNet_dis(nn.Module):
    ''' Pytorch implementation of DMNet: difference minimization network
    for semi-supervised segmentation in medical images
    '''
    def __init__(self, width, class_no):

        super(DMNet_dis, self).__init__()

        if class_no == 2:

            self.final_in = 1

        else:

            self.final_in = class_no

        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8

        # ====================
        # discrminator
        # ====================
        self.disconv1 = double_conv(in_channels=self.final_in, out_channels=self.w1, step=2)
        self.disconv2 = double_conv(in_channels=self.w1, out_channels=self.w2, step=2)
        self.disconv3 = double_conv(in_channels=self.w2, out_channels=self.w1, step=1)
        self.disconv4 = double_conv(in_channels=self.w1, out_channels=self.w1, step=1)
        self.disconv5 = nn.Conv2d(self.w1, self.final_in, 1, bias=True)

    def forward(self, y1, y2, label):

        # ================================
        # Discriminator
        # ================================
        yd_dis = self.disconv1(y1)
        yd_dis = self.disconv2(yd_dis)
        yd_dis = self.disconv3(yd_dis)
        yd_dis = self.disconv4(yd_dis)
        y1_dis = self.disconv5(yd_dis)

        y_dis = self.disconv1(y2)
        y_dis = self.disconv2(y_dis)
        y_dis = self.disconv3(y_dis)
        y_dis = self.disconv4(y_dis)
        y2_dis = self.disconv5(y_dis)

        l_dis = self.disconv1(label)
        l_dis = self.disconv2(l_dis)
        l_dis = self.disconv3(l_dis)
        l_dis = self.disconv4(l_dis)
        l_dis = self.disconv5(l_dis)

        return y1_dis, y2_dis, l_dis


class MTASSUnet(nn.Module):
    ''' MICCAI 2019 baseline

    '''
    def __init__(self, in_ch, width, class_no):
        #
        super(MTASSUnet, self).__init__()
        #
        if class_no == 2:
            #
            self.final_in = 1
            #
        else:
            #
            self.final_in = class_no
        #
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
        #
        self.dconv3 = double_conv(in_channels=self.w4+self.w4, out_channels=self.w3, step=1)
        self.dconv2 = double_conv(in_channels=self.w3+self.w3, out_channels=self.w2, step=1)
        self.dconv1 = double_conv(in_channels=self.w2+self.w2, out_channels=self.w1, step=1)
        self.dconv0 = double_conv(in_channels=self.w1+self.w1, out_channels=self.w1, step=1)
        #
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_last = nn.Conv2d(self.w1, self.final_in, 1, bias=True)
        # ============
        # Decoder R:
        # Auto-decoder
        # =============
        self.dconv3r = double_conv(in_channels=self.w4, out_channels=self.w3, step=1)
        self.dconv2r = double_conv(in_channels=self.w3, out_channels=self.w2, step=1)
        self.dconv1r = double_conv(in_channels=self.w2, out_channels=self.w1, step=1)
        self.dconv0r = double_conv(in_channels=self.w1, out_channels=2*self.final_in, step=1)

    def forward(self, x):

        x0 = self.econv0(x)
        x1 = self.econv1(x0)
        x2 = self.econv2(x1)
        x3 = self.econv3(x2)
        x4 = self.bridge(x3)

        y = self.upsample(x4)

        if y.size()[2] != x3.size()[2]:

            diffY = torch.tensor([x3.size()[2] - y.size()[2]])
            diffX = torch.tensor([x3.size()[3] - y.size()[3]])
            #
            y = F.pad(y, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # =============================
        # Reconstruct Decoder r
        # =============================
        reconstruction = self.dconv3r(y)
        reconstruction = self.upsample(reconstruction)

        reconstruction = self.dconv2r(reconstruction)
        reconstruction = self.upsample(reconstruction)

        reconstruction = self.dconv1r(reconstruction)
        reconstruction = self.upsample(reconstruction)

        reconstruction = self.dconv0r(reconstruction)

        # =============================
        # Segmentation Decoder s
        # =============================
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
        return y, reconstruction


class UNet(nn.Module):
    #
    def __init__(self, in_ch, width, class_no):
        #
        super(UNet, self).__init__()
        #
        # self.final_in = class_no
        if class_no == 2:
            #
            self.final_in = 1
            #
        else:
            #
            self.final_in = class_no
        #
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
        #
        self.dconv3 = double_conv(in_channels=self.w4+self.w4, out_channels=self.w3, step=1)
        self.dconv2 = double_conv(in_channels=self.w3+self.w3, out_channels=self.w2, step=1)
        self.dconv1 = double_conv(in_channels=self.w2+self.w2, out_channels=self.w1, step=1)
        self.dconv0 = double_conv(in_channels=self.w1+self.w1, out_channels=self.w1, step=1)
        #
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_last = nn.Conv2d(self.w1, self.final_in, 1, bias=True)

    def forward(self, x):

        x0 = self.econv0(x)
        x1 = self.econv1(x0)
        x2 = self.econv2(x1)
        x3 = self.econv3(x2)
        x4 = self.bridge(x3)

        y = self.upsample(x4)

        if y.size()[2] != x3.size()[2]:

            diffY = torch.tensor([x3.size()[2] - y.size()[2]])
            diffX = torch.tensor([x3.size()[3] - y.size()[3]])
            #
            y = F.pad(y, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

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
        return y


class TwoDecoderUnet(nn.Module):
    #
    def __init__(self, in_ch, width, class_no, decoder_type):
        #
        super(TwoDecoderUnet, self).__init__()
        #
        self.decoder_type = decoder_type
        #
        if class_no == 2:
            #
            self.final_in = 1
            #
        else:
            #
            self.final_in = class_no
        #
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
        #
        if self.decoder_type == 'FPA':

            self.d3l = FPA(in_channels=self.w3, out_channels=self.w3, step=1, addition=True)
            self.d3l_smooth = single_conv(in_channels=self.w4+self.w4, out_channels=self.w3, step=1)

            self.d2l = FPA(in_channels=self.w2, out_channels=self.w2, step=1, addition=True)
            self.d2l_smooth = single_conv(in_channels=self.w3+self.w3, out_channels=self.w2, step=1)

            self.d1l = FPA(in_channels=self.w1, out_channels=self.w1, step=1, addition=True)
            self.d1l_smooth = single_conv(in_channels=self.w2+self.w2, out_channels=self.w1, step=1)

            self.d0l = FPA(in_channels=self.w1, out_channels=self.w1, step=1, addition=True)
            self.d0l_smooth = single_conv(in_channels=self.w1+self.w1, out_channels=self.w1, step=1)

            self.d3r = single_conv(in_channels=self.w3, out_channels=self.w3, step=1)
            self.d3r_smooth = single_conv(in_channels=self.w4+self.w4, out_channels=self.w3, step=1)

            self.d2r = single_conv(in_channels=self.w2, out_channels=self.w2, step=1)
            self.d2r_smooth = single_conv(in_channels=self.w3+self.w3, out_channels=self.w2, step=1)

            self.d1r = single_conv(in_channels=self.w1, out_channels=self.w1, step=1)
            self.d1r_smooth = single_conv(in_channels=self.w2+self.w2, out_channels=self.w1, step=1)

            self.d0r = single_conv(in_channels=self.w1, out_channels=self.w1, step=1)
            self.d0r_smooth = single_conv(in_channels=self.w1+self.w1, out_channels=self.w1, step=1)

            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

            self.dl_last = nn.Conv2d(self.w1, self.final_in, 1, bias=True)
            self.dr_last = nn.Conv2d(self.w1, self.final_in, 1, bias=True)

        elif self.decoder_type == 'FNA':

            self.d3l = FNA(in_channels=self.w3, out_channels=self.w3, step=1, addition=True)
            self.d3l_smooth = single_conv(in_channels=self.w4+self.w4, out_channels=self.w3, step=1)

            self.d2l = FNA(in_channels=self.w2, out_channels=self.w2, step=1, addition=True)
            self.d2l_smooth = single_conv(in_channels=self.w3+self.w3, out_channels=self.w2, step=1)

            self.d1l = FNA(in_channels=self.w1, out_channels=self.w1, step=1, addition=True)
            self.d1l_smooth = single_conv(in_channels=self.w2+self.w2, out_channels=self.w1, step=1)

            self.d0l = FNA(in_channels=self.w1, out_channels=self.w1, step=1, addition=True)
            self.d0l_smooth = single_conv(in_channels=self.w1+self.w1, out_channels=self.w1, step=1)

            self.d3r = single_conv(in_channels=self.w3, out_channels=self.w3, step=1)
            self.d3r_smooth = single_conv(in_channels=self.w4+self.w4, out_channels=self.w3, step=1)

            self.d2r = single_conv(in_channels=self.w2, out_channels=self.w2, step=1)
            self.d2r_smooth = single_conv(in_channels=self.w3+self.w3, out_channels=self.w2, step=1)

            self.d1r = single_conv(in_channels=self.w1, out_channels=self.w1, step=1)
            self.d1r_smooth = single_conv(in_channels=self.w2+self.w2, out_channels=self.w1, step=1)

            self.d0r = single_conv(in_channels=self.w1, out_channels=self.w1, step=1)
            self.d0r_smooth = single_conv(in_channels=self.w1+self.w1, out_channels=self.w1, step=1)

            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

            self.dl_last = nn.Conv2d(self.w1, self.final_in, 1, bias=True)
            self.dr_last = nn.Conv2d(self.w1, self.final_in, 1, bias=True)

        elif self.decoder_type == 'Conv':

            self.d3l = single_conv(in_channels=self.w3, out_channels=self.w3, step=1)
            self.d3l_smooth = single_conv(in_channels=self.w4+self.w4, out_channels=self.w3, step=1)

            self.d2l = single_conv(in_channels=self.w2, out_channels=self.w2, step=1)
            self.d2l_smooth = single_conv(in_channels=self.w3+self.w3, out_channels=self.w2, step=1)

            self.d1l = single_conv(in_channels=self.w1, out_channels=self.w1, step=1)
            self.d1l_smooth = single_conv(in_channels=self.w2+self.w2, out_channels=self.w1, step=1)

            self.d0l = single_conv(in_channels=self.w1, out_channels=self.w1, step=1)
            self.d0l_smooth = single_conv(in_channels=self.w1+self.w1, out_channels=self.w1, step=1)

            self.d3r = single_conv(in_channels=self.w3, out_channels=self.w3, step=1)
            self.d3r_smooth = single_conv(in_channels=self.w4+self.w4, out_channels=self.w3, step=1)

            self.d2r = single_conv(in_channels=self.w2, out_channels=self.w2, step=1)
            self.d2r_smooth = single_conv(in_channels=self.w3+self.w3, out_channels=self.w2, step=1)

            self.d1r = single_conv(in_channels=self.w1, out_channels=self.w1, step=1)
            self.d1r_smooth = single_conv(in_channels=self.w2+self.w2, out_channels=self.w1, step=1)

            self.d0r = single_conv(in_channels=self.w1, out_channels=self.w1, step=1)
            self.d0r_smooth = single_conv(in_channels=self.w1+self.w1, out_channels=self.w1, step=1)

            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

            self.dl_last = nn.Conv2d(self.w1, self.final_in, 1, bias=True)
            self.dr_last = nn.Conv2d(self.w1, self.final_in, 1, bias=True)

    def forward(self, x):

        x0 = self.econv0(x)
        x1 = self.econv1(x0)
        x2 = self.econv2(x1)
        x3 = self.econv3(x2)
        x4 = self.bridge(x3)

        y = self.upsample(x4)

        if y.size()[2] != x3.size()[2]:

            diffY = torch.tensor([x3.size()[2] - y.size()[2]])
            diffX = torch.tensor([x3.size()[3] - y.size()[3]])
            #
            y = F.pad(y, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        y_l = torch.cat([y, x3], dim=1)
        y_l = self.upsample(y_l)
        y_l = self.d3l_smooth(y_l)
        y_l = self.d3l(y_l)

        y_l = torch.cat([y_l, x2], dim=1)
        y_l = self.upsample(y_l)
        y_l = self.d2l_smooth(y_l)
        y_l = self.d2l(y_l)

        y_l = torch.cat([y_l, x1], dim=1)
        y_l = self.upsample(y_l)
        y_l = self.d1l_smooth(y_l)
        y_l = self.d1l(y_l)

        y_l = torch.cat([y_l, x0], dim=1)
        y_l = self.d0l_smooth(y_l)
        y_l = self.d0l(y_l)

        y_l = self.dl_last(y_l)

        y_r = torch.cat([y, x3], dim=1)
        y_r = self.upsample(y_r)
        y_r = self.d3r_smooth(y_r)
        y_r = self.d3r(y_r)

        y_r = torch.cat([y_r, x2], dim=1)
        y_r = self.upsample(y_r)
        y_r = self.d2r_smooth(y_r)
        y_r = self.d2r(y_r)

        y_r = torch.cat([y_r, x1], dim=1)
        y_r = self.upsample(y_r)
        y_r = self.d1r_smooth(y_r)
        y_r = self.d1r(y_r)

        y_r = torch.cat([y_r, x0], dim=1)
        y_r = self.d0r_smooth(y_r)
        y_r = self.d0r(y_r)

        y_r = self.dr_last(y_r)

        return y_l, y_r


# =============================
# Blocks
# =============================


def double_conv(in_channels, out_channels, step):
    #
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True)
    )


def single_conv(in_channels, out_channels, step):
    #
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True)
    )


class FPA(nn.Module):
    #
    def __init__(self, in_channels, out_channels, step, addition):
        #
        super(FPA, self).__init__()
        self.addition = addition
        #
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=step, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
            nn.ReLU(inplace=True)
        )
        #
        self.attention_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=step, dilation=1, padding=0, bias=False),
            nn.Conv2d(in_channels=in_channels//8, out_channels=in_channels//8, kernel_size=3, stride=1, dilation=5, padding=5, bias=False),
            nn.InstanceNorm2d(num_features=in_channels//8, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels//8, out_channels=out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        attention = self.attention_branch(x)
        features = self.main_branch(x)

        if self.addition is True:
            output = attention*features + features
        else:
            output = attention * features

        return output


class FNA(nn.Module):
    #
    def __init__(self, in_channels, out_channels, step, addition):
        #
        super(FNA, self).__init__()
        #
        self.addition = addition
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=step, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
            nn.ReLU(inplace=True)
        )
        #
        self.attention_branch_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1, stride=step, dilation=1, padding=0, bias=False)
        self.attention_branch_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels // 8, out_channels=in_channels // 8, kernel_size=3, stride=1, dilation=1, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=in_channels // 8, affine=True),
            nn.ReLU(inplace=True)
        )
        #
        self.attention_branch_3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels // 8, out_channels=in_channels // 8, kernel_size=3, stride=1, dilation=1, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=in_channels // 8, affine=True),
            nn.ReLU(inplace=True)
        )
        #
        self.attention_branch_4 = nn.Conv2d(in_channels=in_channels//8, out_channels=out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False)

    def forward(self, x):

        f1 = self.attention_branch_1(x)
        f2 = self.attention_branch_2(f1) + f1
        attention = self.attention_branch_3(f2) + f2
        attention = self.attention_branch_4(attention)
        features = self.main_branch(x)
        #
        if self.addition is True:
            output = attention*features + features
        else:
            output = attention*features

        return output


