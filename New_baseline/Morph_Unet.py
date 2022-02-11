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


class Morphology(nn.Module):
    '''
    Base class for morpholigical operators
    For now, only supports stride=1, dilation=1, kernel_size H==W, and padding='same'.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=15, type=None):
        '''
        in_channels: scalar
        out_channels: scalar, the number of the morphological neure.
        kernel_size: scalar, the spatial size of the morphological neure.
        soft_max: bool, using the soft max rather the torch.max(), ref: Dense Morphological Networks: An Universal Function Approximator (Mondal et al. (2019)).
        beta: scalar, used by soft_max.
        type: str, dilation2d or erosion2d.
        '''
        super(Morphology, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.soft_max = soft_max
        self.beta = beta
        self.type = type

        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)

    def forward(self, x):
        '''
        x: tensor of shape (B,C,H,W)
        '''
        # padding
        x = fixed_padding(x, self.kernel_size, dilation=1)

        # unfold
        x = self.unfold(x)  # (B, Cin*kH*kW, L), where L is the numbers of patches
        x = x.unsqueeze(1)  # (B, 1, Cin*kH*kW, L)
        L = x.size(-1)
        L_sqrt = int(math.sqrt(L))

        # erosion
        weight = self.weight.view(self.out_channels, -1)  # (Cout, Cin*kH*kW)
        weight = weight.unsqueeze(0).unsqueeze(-1)  # (1, Cout, Cin*kH*kW, 1)

        if self.type == 'erosion2d':
            x = weight - x  # (B, Cout, Cin*kH*kW, L)
        elif self.type == 'dilation2d':
            x = weight + x  # (B, Cout, Cin*kH*kW, L)
        else:
            raise ValueError

        if not self.soft_max:
            x, _ = torch.max(x, dim=2, keepdim=False)  # (B, Cout, L)
        else:
            x = torch.logsumexp(x * self.beta, dim=2, keepdim=False) / self.beta  # (B, Cout, L)

        if self.type == 'erosion2d':
            x = -1 * x

        # instead of fold, we use view to avoid copy
        x = x.view(-1, self.out_channels, L_sqrt, L_sqrt)  # (B, Cout, L/2, L/2)

        return x


class Dilation2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Dilation2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'dilation2d')


class Erosion2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Erosion2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'erosion2d')


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class Morph(nn.Module):

    def __init__(self, in_ch, width, class_no):

        super(Morph, self).__init__()
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

        self.erosion = Erosion2d(self.w4, self.w4, 3)
        self.dilation = Dilation2d(self.w4, self.w4, 3)

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

        y_erosion = self.erosion(y)
        y_dilation = self.dilation(y)

        y3 = torch.cat([y_erosion, x3], dim=1)
        y3 = self.dconv3(y3)
        y2 = self.upsample(y3)
        # y2 = self.erosion(y2)
        y2 = torch.cat([y2, x2], dim=1)
        y2 = self.dconv2(y2)
        y1 = self.upsample(y2)
        # y1 = self.erosion(y1)
        y1 = torch.cat([y1, x1], dim=1)
        y1 = self.dconv1(y1)
        y0 = self.upsample(y1)
        y0 = torch.cat([y0, x0], dim=1)
        y0 = self.dconv0(y0)
        y_erosion = self.dconv_last(y0)

        y3 = torch.cat([y_dilation, x3], dim=1)
        y3 = self.dconv3(y3)
        y2 = self.upsample(y3)
        # y2 = self.dilation(y2)
        y2 = torch.cat([y2, x2], dim=1)
        y2 = self.dconv2(y2)
        y1 = self.upsample(y2)
        # y1 = self.dilation(y1)
        y1 = torch.cat([y1, x1], dim=1)
        y1 = self.dconv1(y1)
        y0 = self.upsample(y1)
        y0 = torch.cat([y0, x0], dim=1)
        y0 = self.dconv0(y0)
        y_dilation = self.dconv_last(y0)

        return y_erosion, y_dilation