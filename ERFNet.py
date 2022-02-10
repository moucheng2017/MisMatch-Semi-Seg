import torch
import torch.nn as nn
import torch.nn.functional as F
# =============================
# Networks
# =============================


class ERFAnet(nn.Module):

    def __init__(self, in_ch, width, class_no, identity_add=True):

        super(ERFAnet, self).__init__()

        self.identity = identity_add

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

        # FPA branch:
        self.fpa3 = FPA(in_channels=self.w3, out_channels=self.w3, step=1, addition=self.identity)
        self.fpa3_smooth = single_conv(in_channels=self.w4+self.w4, out_channels=self.w3, step=1)

        self.fpa2 = FPA(in_channels=self.w2, out_channels=self.w2, step=1, addition=self.identity)
        self.fpa2_smooth = single_conv(in_channels=self.w3+self.w3, out_channels=self.w2, step=1)

        self.fpa1 = FPA(in_channels=self.w1, out_channels=self.w1, step=1, addition=self.identity)
        self.fpa1_smooth = single_conv(in_channels=self.w2+self.w2, out_channels=self.w1, step=1)

        self.fpa0 = FPA(in_channels=self.w1, out_channels=self.w1, step=1, addition=self.identity)
        self.fpa0_smooth = single_conv(in_channels=self.w1+self.w1, out_channels=self.w1, step=1)

        self.fna3 = FNA(in_channels=self.w3, out_channels=self.w3, step=1, addition=self.identity)
        self.fna3_smooth = single_conv(in_channels=self.w4+self.w4, out_channels=self.w3, step=1)

        self.fna2 = FNA(in_channels=self.w2, out_channels=self.w2, step=1, addition=self.identity)
        self.fna2_smooth = single_conv(in_channels=self.w3+self.w3, out_channels=self.w2, step=1)

        self.fna1 = FNA(in_channels=self.w1, out_channels=self.w1, step=1, addition=self.identity)
        self.fna1_smooth = single_conv(in_channels=self.w2+self.w2, out_channels=self.w1, step=1)

        self.fna0 = FNA(in_channels=self.w1, out_channels=self.w1, step=1, addition=self.identity)
        self.fna0_smooth = single_conv(in_channels=self.w1+self.w1, out_channels=self.w1, step=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_fp_last = nn.Conv2d(self.w1, self.final_in, 1, bias=True)
        self.dconv_fn_last = nn.Conv2d(self.w1, self.final_in, 1, bias=True)

        self.reduce_dim = nn.MaxPool2d(2)

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

        # FPA branch:
        y_fp = torch.cat([y, x3], dim=1)
        y_fp = self.upsample(y_fp)
        y_fp = self.fpa3_smooth(y_fp)
        y_fp = self.fpa3(y_fp)

        y_fp = torch.cat([y_fp, x2], dim=1)
        y_fp = self.upsample(y_fp)
        y_fp = self.fpa2_smooth(y_fp)
        y_fp = self.fpa2(y_fp)

        y_fp = torch.cat([y_fp, x1], dim=1)
        y_fp = self.upsample(y_fp)
        y_fp = self.fpa1_smooth(y_fp)
        y_fp = self.fpa1(y_fp)

        y_fp = torch.cat([y_fp, x0], dim=1)
        y_fp = self.fpa0_smooth(y_fp)
        y_fp = self.fpa0(y_fp)

        y_fp = self.dconv_fp_last(y_fp)

        # FNA branch:
        y_fn = torch.cat([y, x3], dim=1)
        y_fn = self.upsample(y_fn)
        y_fn = self.fna3_smooth(y_fn)
        y_fn = self.fna3(y_fn)

        y_fn = torch.cat([y_fn, x2], dim=1)
        y_fn = self.upsample(y_fn)
        y_fn = self.fna2_smooth(y_fn)
        y_fn = self.fna2(y_fn)

        y_fn = torch.cat([y_fn, x1], dim=1)
        y_fn = self.upsample(y_fn)
        y_fn = self.fna1_smooth(y_fn)
        y_fn = self.fna1(y_fn)

        y_fn = torch.cat([y_fn, x0], dim=1)
        y_fn = self.fna0_smooth(y_fn)
        y_fn = self.fna0(y_fn)

        y_fn = self.dconv_fn_last(y_fn)

        return y_fp, y_fn

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

        # self.renormalisation = nn.InstanceNorm2d(out_channels)

    def forward(self, x):

        attention = self.attention_branch(x)
        features = self.main_branch(x)

        # if self.addition is True:
        #     output = self.renormalisation(attention*features + features)
        # else:
        #     output = self.renormalisation(attention * features)
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
        # self.attention_branch_4 = nn.Sequential(nn.Conv2d(in_channels=in_channels//8, out_channels=out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False),
        #                                         nn.Sigmoid())
        self.attention_branch_4 = nn.Conv2d(in_channels=in_channels//8, out_channels=out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False),

        # self.renormalisation = nn.InstanceNorm2d(out_channels)

    def forward(self, x):

        f1 = self.attention_branch_1(x)
        f2 = self.attention_branch_2(f1) + f1
        attention = self.attention_branch_3(f2) + f2
        attention = self.attention_branch_4(attention)
        features = self.main_branch(x)
        #
        # if self.addition is True:
        #     output = self.renormalisation(attention*features + features)
        # else:
        #     output = self.renormalisation(attention * features)

        if self.addition is True:
            output = attention*features + features
        else:
            output = attention * features

        return output


