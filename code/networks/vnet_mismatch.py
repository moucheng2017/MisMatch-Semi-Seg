import torch
from torch import nn
import torch.nn.functional as F


class PASBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, normalization='none', upsampling=False, residual=True, dilation_rate=6):
        super(PASBlock, self).__init__()
        #
        ops = []
        attention_branch = []
        if normalization != 'none':
            if upsampling is True:
                ops.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
                ops.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1))

                attention_branch.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
                attention_branch.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=dilation_rate, stride=1, dilation=dilation_rate))

            else:
                ops.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1))
                attention_branch.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=dilation_rate, stride=1, dilation=dilation_rate))

            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
                attention_branch.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
                attention_branch.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
                attention_branch.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            if upsampling is True:
                ops.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
                ops.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1))

                attention_branch.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
                attention_branch.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=dilation_rate, stride=1, dilation=dilation_rate))

            else:
                ops.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1))
                attention_branch.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=dilation_rate, stride=1, dilation=dilation_rate))

        ops.append(nn.ReLU(inplace=True))
        attention_branch.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.side_conv = nn.Sequential(*attention_branch)
        self.residual_mode = residual

    def forward(self, x):
        y = self.conv(x)
        a = self.side_conv(x)
        # print(a.size())
        if self.residual_mode is True:
            y = torch.sigmoid(a) * y + y
        else:
            y = torch.sigmoid(a) * y
        return y


class NASBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, normalization='none', upsampling=False, residual=True):
        super(NASBlock, self).__init__()
        #
        ops = []
        attention_branch = []
        attention_branch_stage2 = []
        if normalization != 'none':
            if upsampling is True:
                ops.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
                ops.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1))

                attention_branch.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
                attention_branch.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1, groups=n_filters_out))

                attention_branch_stage2.append(nn.Conv3d(in_channels=n_filters_out, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1, groups=n_filters_out))

            else:
                ops.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1))
                attention_branch.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1, groups=n_filters_out))

                attention_branch_stage2.append(nn.Conv3d(in_channels=n_filters_out, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1, groups=n_filters_out))

            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
                attention_branch.append(nn.BatchNorm3d(n_filters_out))
                attention_branch_stage2.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
                attention_branch.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
                attention_branch_stage2.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
                attention_branch.append(nn.InstanceNorm3d(n_filters_out))
                attention_branch_stage2.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            if upsampling is True:
                ops.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
                ops.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1))

                attention_branch.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
                attention_branch.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1, groups=n_filters_out))

                attention_branch_stage2.append(nn.Conv3d(in_channels=n_filters_out, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1, groups=n_filters_out))

            else:
                ops.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1))
                attention_branch.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1, groups=n_filters_out))

                attention_branch_stage2 .append(nn.Conv3d(in_channels=n_filters_out, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1, groups=n_filters_out))

        ops.append(nn.ReLU(inplace=True))
        attention_branch.append(nn.ReLU(inplace=True))
        attention_branch_stage2.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.side_conv = nn.Sequential(*attention_branch)
        self.side_conv_s2 = nn.Sequential(*attention_branch_stage2)
        self.residual_mode = residual

    def forward(self, x):
        y = self.conv(x)
        a = self.side_conv(x)
        a = self.side_conv_s2(a) + a
        if self.residual_mode is True:
            y = torch.sigmoid(a) * y + y
        else:
            y = torch.sigmoid(a) * y
        return y


class PNASBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, normalization='none', upsampling=False, residual=True):
        super(PNASBlock, self).__init__()
        # Shared conv
        ops = []
        negative_attention_branch_stage1 = []
        negative_attention_branch_stage2 = []
        positive_attention_branch = []
        if normalization != 'none':
            if upsampling is True:
                ops.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
                ops.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1))

                negative_attention_branch_stage1.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
                negative_attention_branch_stage1.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1))
                negative_attention_branch_stage2.append(nn.Conv3d(in_channels=n_filters_out, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1))

                positive_attention_branch.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
                positive_attention_branch.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=6, stride=1, dilation=6))

            else:
                ops.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1))

                negative_attention_branch_stage1.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1))
                negative_attention_branch_stage2.append(nn.Conv3d(in_channels=n_filters_out, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1))

                positive_attention_branch.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=6, stride=1, dilation=6))

            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
                negative_attention_branch_stage1.append(nn.BatchNorm3d(n_filters_out))
                negative_attention_branch_stage2.append(nn.BatchNorm3d(n_filters_out))
                positive_attention_branch.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
                negative_attention_branch_stage1.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
                negative_attention_branch_stage2.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
                positive_attention_branch.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
                negative_attention_branch_stage1.append(nn.InstanceNorm3d(n_filters_out))
                negative_attention_branch_stage2.append(nn.InstanceNorm3d(n_filters_out))
                positive_attention_branch.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            if upsampling is True:
                ops.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
                ops.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1))

                negative_attention_branch_stage1.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
                negative_attention_branch_stage1.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1))
                negative_attention_branch_stage2.append(nn.Conv3d(in_channels=n_filters_out, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1))

                positive_attention_branch.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
                positive_attention_branch.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1, dilation=6))

            else:
                ops.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1))

                negative_attention_branch_stage1.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1))
                negative_attention_branch_stage2.append(nn.Conv3d(in_channels=n_filters_out, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1))

                positive_attention_branch.append(nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1, dilation=6))

        ops.append(nn.ReLU(inplace=True))
        negative_attention_branch_stage1.append(nn.ReLU(inplace=True))
        negative_attention_branch_stage2.append(nn.ReLU(inplace=True))

        positive_attention_branch.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.n_side_conv_s1 = nn.Sequential(*negative_attention_branch_stage1)
        self.n_side_conv_s2 = nn.Sequential(*negative_attention_branch_stage2)
        self.p_side_conv = nn.Sequential(*positive_attention_branch)

        self.residual_mode = residual

    def forward(self, xp, xn):

        xpo = self.conv(xp)
        ap = self.p_side_conv(xp)
        if self.residual_mode is True:
            xp = torch.sigmoid(ap) * xpo + xpo
        else:
            xp = torch.sigmoid(ap) * xpo

        xno = self.conv(xn)
        an = self.n_side_conv_s1(xn)
        an = self.n_side_conv_s2(an) + an
        if self.residual_mode is True:
            xn = torch.sigmoid(an) * xno + xno
        else:
            xn = torch.sigmoid(an) * xno

        return xp, xn


# ============================
# Original blocks:
# ============================
class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class VNetMisMatchEfficient(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(VNetMisMatchEfficient, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = PNASBlock(n_filters * 16, n_filters * 8, normalization=normalization, upsampling=True)

        self.block_six = PNASBlock(n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = PNASBlock(n_filters * 8, n_filters * 4, normalization=normalization, upsampling=True)

        self.block_seven = PNASBlock(n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = PNASBlock(n_filters * 4, n_filters * 2, normalization=normalization, upsampling=True)

        self.block_eight = PNASBlock(n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = PNASBlock(n_filters * 2, n_filters, normalization=normalization, upsampling=True)

        self.block_nine = PNASBlock(n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up_p, x5_up_n = self.block_five_up(x5, x5)
        x5_up_p = x5_up_p + x4
        x5_up_n = x5_up_n + x4

        x6p, x6n = self.block_six(x5_up_p, x5_up_n)
        x6_up_p, x6_up_n = self.block_six_up(x6p, x6n)
        x6_up_p = x6_up_p + x3
        x6_up_n = x6_up_n + x3

        x7p, x7n = self.block_seven(x6_up_p, x6_up_n)
        x7_up_p, x7_up_n = self.block_seven_up(x7p, x7n)
        x7_up_p = x7_up_p + x2
        x7_up_n = x7_up_n + x2

        x8p, x8n = self.block_eight(x7_up_p, x7_up_n)
        x8_up_p, x8_up_n = self.block_eight_up(x8p, x8n)
        x8_up_p = x8_up_p + x1
        x9p, x9n = self.block_nine(x8_up_p, x8_up_n)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9p = self.dropout(x9p)
        if self.has_dropout:
            x9n = self.dropout(x9n)
        out_p = self.out_conv(x9p)
        out_n = self.out_conv(x9n)
        return out_p, out_n

    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out_p, out_n = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out_p, out_n

    # def __init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv3d):
    #             torch.nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, nn.BatchNorm3d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()


class VNetMisMatch(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=8, normalization='none', dilation=6, has_dropout=False):
        super(VNetMisMatch, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up_p = PASBlock(n_filters * 16, n_filters * 8, normalization=normalization, upsampling=True, dilation_rate=dilation)
        self.block_five_up_n = NASBlock(n_filters * 16, n_filters * 8, normalization=normalization, upsampling=True)

        self.block_six_p = PASBlock(n_filters * 8, n_filters * 8, normalization=normalization, dilation_rate=dilation)
        self.block_six_up_p = PASBlock(n_filters * 8, n_filters * 4, normalization=normalization, upsampling=True, dilation_rate=dilation)

        self.block_six_n = NASBlock(n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up_n = NASBlock(n_filters * 8, n_filters * 4, normalization=normalization, upsampling=True)

        self.block_seven_p = PASBlock(n_filters * 4, n_filters * 4, normalization=normalization, dilation_rate=dilation)
        self.block_seven_up_p = PASBlock(n_filters * 4, n_filters * 2, normalization=normalization, upsampling=True, dilation_rate=dilation)

        self.block_seven_n = NASBlock(n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up_n = NASBlock(n_filters * 4, n_filters * 2, normalization=normalization, upsampling=True)

        self.block_eight_p = PASBlock(n_filters * 2, n_filters * 2, normalization=normalization, dilation_rate=dilation)
        self.block_eight_up_p = PASBlock(n_filters * 2, n_filters, normalization=normalization, upsampling=True, dilation_rate=dilation)

        self.block_eight_n = NASBlock(n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up_n = NASBlock(n_filters * 2, n_filters, normalization=normalization, upsampling=True)

        self.block_nine_p = PASBlock(n_filters, n_filters, normalization=normalization, dilation_rate=dilation)
        self.block_nine_n = NASBlock(n_filters, n_filters, normalization=normalization)

        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):

        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up_p = self.block_five_up_p(x5)
        x5_up_p = x5_up_p + x4
        x5_up_n = self.block_five_up_n(x5)
        x5_up_n = x5_up_n + x4

        x6p = self.block_six_p(x5_up_p)
        x6_up_p = self.block_six_up_p(x6p)
        x6_up_p = x6_up_p + x3
        x6n = self.block_six_n(x5_up_n)
        x6_up_n = self.block_six_up_n(x6n)
        x6_up_n = x6_up_n + x3

        x7p = self.block_seven_p(x6_up_p)
        x7_up_p = self.block_seven_up_p(x7p)
        x7_up_p = x7_up_p + x2
        x7n = self.block_seven_n(x6_up_n)
        x7_up_n = self.block_seven_up_n(x7n)
        x7_up_n = x7_up_n + x2

        x8p = self.block_eight_p(x7_up_p)
        x8_up_p = self.block_eight_up_p(x8p)
        x8_up_p = x8_up_p + x1
        x8n = self.block_eight_n(x7_up_n)
        x8_up_n = self.block_eight_up_n(x8n)
        x8_up_n = x8_up_n + x1

        x9p = self.block_nine_p(x8_up_p)
        x9n = self.block_nine_n(x8_up_n)
        out_p = self.out_conv(x9p)
        out_n = self.out_conv(x9n)
        return out_p, out_n

    def forward(self, input, turnoff_drop=True):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out_p, out_n = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out_p, out_n