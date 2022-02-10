import torch
import torch.nn as nn
import torch.nn.functional as F
# =============================
# Networks
# =============================


class NewERFAnet3(nn.Module):

    def __init__(self, in_ch, width, class_no, identity_add=True):

        super(NewERFAnet3, self).__init__()

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

        self.self_constrained_bridge = ContrastiveModule(in_ch=self.w4)

    def forward(self, _x, x, x_):

        x0 = self.econv0(x)
        x1 = self.econv1(x0)
        x2 = self.econv2(x1)
        x3 = self.econv3(x2)
        x4 = self.bridge(x3)

        _x = self.econv0(_x)
        _x = self.econv1(_x)
        _x = self.econv2(_x)
        _x = self.econv3(_x)
        _x = self.bridge(_x)

        x_ = self.econv0(x_)
        x_ = self.econv1(x_)
        x_ = self.econv2(x_)
        x_ = self.econv3(x_)
        x_ = self.bridge(x_)

        pseudo_x1_a, pseudo_x1_b, pseudo_x2, pseudo_x3_a, pseudo_x3_b, x1_value, x2_value, x3_value = self.self_constrained_bridge(_x, x4, x_)

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

        return y_fp, y_fn, pseudo_x1_a, pseudo_x1_b, pseudo_x2, pseudo_x3_a, pseudo_x3_b, x1_value, x2_value, x3_value
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

    def __init__(self, in_channels, out_channels, step, addition):

        super(FPA, self).__init__()
        self.addition = addition

        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=step, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

        self.attention_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=step, dilation=1, padding=0, bias=False),
            nn.Conv2d(in_channels=in_channels//8, out_channels=in_channels//8, kernel_size=3, stride=1, dilation=9, padding=9, bias=False),
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
        attention = self.attention_branch_3(f2) + f2 + f1
        attention = self.attention_branch_4(attention)
        features = self.main_branch(x)

        if self.addition is True:
            output = attention*features + features
        else:
            output = attention*features

        return output


class ContrastiveModule(nn.Module):

    def __init__(self, in_ch):
        super(ContrastiveModule, self).__init__()

        self.reduce_dim_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch // 8, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(num_features=in_ch // 8, affine=True),
            nn.ReLU()
        )

        self.reduce_dim_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch // 8, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(num_features=in_ch // 8, affine=True),
            nn.ReLU()
        )

        self.reduce_dim_3 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch // 8, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(num_features=in_ch // 8, affine=True),
            nn.ReLU()
        )

    def forward(self, x1, x2, x3):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        x1_query = self.reduce_dim_1(x1)
        x2_query = self.reduce_dim_1(x2)
        x3_query = self.reduce_dim_1(x3)

        x1_key = self.reduce_dim_2(x1)
        x2_key = self.reduce_dim_2(x2)
        x3_key = self.reduce_dim_2(x3)

        x1_value = self.reduce_dim_3(x1)
        x2_value = self.reduce_dim_3(x2)
        x3_value = self.reduce_dim_3(x3)

        b, c, h, w = x1_query.size()

        # Reshape tensors:

        x1_query = x1_query.view(b, c, h*w).permute(0, 2, 1).contiguous() # b x c x n
        x1_key = x1_key.view(b, c, h * w)

        x2_query = x2_query.view(b, c, h*w).permute(0, 2, 1).contiguous()
        x2_key = x2_key.view(b, c, h * w)

        x3_query = x3_query.view(b, c, h*w).permute(0, 2, 1).contiguous()
        x3_key = x3_key.view(b, c, h * w)

        # Affinity matrices:

        x1_query_x2_key = F.softmax(torch.bmm(x1_query, x2_key), dim=-1)
        x2_query_x1_key = F.softmax(torch.bmm(x2_query, x1_key), dim=-1)

        x2_query_x3_key = F.softmax(torch.bmm(x2_query, x3_key), dim=-1)
        x3_query_x2_key = F.softmax(torch.bmm(x3_query, x2_key), dim=-1)

        x1_query_x3_key = F.softmax(torch.bmm(x1_query, x3_key), dim=-1)
        # x3_query_x1_key = F.softmax(torch.bmm(x3_query, x1_key), dim=-1)

        # =====================================
        # Skew Anti-Symmetry (SAS) constraints:
        # =====================================
        # x1 --> x2 --> x3 --> x2 --> x1:
        pseudo_x1_a = torch.ones_like(x1_key).to(device=device)
        pseudo_x1_a = torch.bmm(pseudo_x1_a, x1_query_x2_key)
        pseudo_x1_a = torch.bmm(pseudo_x1_a, x2_query_x3_key)
        pseudo_x1_a = torch.bmm(pseudo_x1_a, x3_query_x2_key)
        pseudo_x1_a = torch.bmm(pseudo_x1_a, x2_query_x1_key)
        pseudo_x1_a = pseudo_x1_a.view(b, c, h, w)

        # x1 --> x2 --> x1:
        pseudo_x1_b = torch.ones_like(x1_key).to(device=device)
        pseudo_x1_b = torch.bmm(pseudo_x1_b, x1_query_x2_key)
        pseudo_x1_b = torch.bmm(pseudo_x1_b, x2_query_x1_key)
        pseudo_x1_b = pseudo_x1_b.view(b, c, h, w)

        # x2 --> x3 --> x2:
        pseudo_x2 = torch.ones_like(x1_key).to(device=device)
        pseudo_x2 = torch.bmm(pseudo_x2, x2_query_x3_key)
        pseudo_x2 = torch.bmm(pseudo_x2, x3_query_x2_key)
        pseudo_x2 = pseudo_x2.view(b, c, h, w)

        # =====================================
        # Jacobi Identity constraints:
        # =====================================
        # x1 --> x2 --> x3
        pseudo_x3_a = torch.ones_like(x1_key).to(device=device)
        pseudo_x3_a = torch.bmm(pseudo_x3_a, x1_query_x2_key)
        pseudo_x3_a = torch.bmm(pseudo_x3_a, x2_query_x3_key)
        pseudo_x3_a = pseudo_x3_a.view(b, c, h, w)

        # x1 --> x3
        pseudo_x3_b = torch.ones_like(x1_key).to(device=device)
        pseudo_x3_b = torch.bmm(pseudo_x3_b, x1_query_x3_key)
        pseudo_x3_b = pseudo_x3_b.view(b, c, h, w)

        return pseudo_x1_a, pseudo_x1_b, pseudo_x2, pseudo_x3_a, pseudo_x3_b, x1_value, x2_value, x3_value


def constrained_loss(x1, x2, x3, pseudo_x1_a, pseudo_x1_b, pseudo_x2, pseudo_x3_a, pseudo_x3_b, normalisation='softmax', mode='sum'):

    if normalisation == 'softmax':

        b, c, h, w = x1.size()

        label_x1 = x1.view(b, c, -1)
        label_x1 = F.softmax(label_x1, dim=-1)
        label_x1 = label_x1.view(b, c, h, w)

        label_x2 = x2.view(b, c, -1)
        label_x2 = F.softmax(label_x2, dim=-1)
        label_x2 = label_x2.view(b, c, h, w)

        label_x3 = x3.view(b, c, -1)
        label_x3 = F.softmax(label_x3, dim=-1)
        label_x3 = label_x3.view(b, c, h, w)

    elif normalisation == 'sigmoid':

        label_x1 = F.sigmoid(x1)
        label_x2 = F.sigmoid(x2)
        label_x3 = F.sigmoid(x3)

    # if mode == 'sum':
    #
    #     loss_x1a = torch.nn.MSELoss(reduction='sum')(pseudo_x1_a, label_x1) + torch.nn.MSELoss(reduction='sum')(label_x1, pseudo_x1_a)
    #     loss_x1a = loss_x1a * 0.5
    #     loss_x1b = torch.nn.MSELoss(reduction='sum')(pseudo_x1_b, label_x1) + torch.nn.MSELoss(reduction='sum')(label_x1, pseudo_x1_b)
    #     loss_x1b = loss_x1b * 0.5
    #     loss_x1a_x1b = torch.nn.MSELoss(reduction='sum')(pseudo_x1_a, pseudo_x1_b) + torch.nn.MSELoss(reduction='sum')(pseudo_x1_b, pseudo_x1_a)
    #     loss_x1a_x1b = loss_x1a_x1b * 0.5
    #
    #     loss_x2 = torch.nn.MSELoss(reduction='sum')(pseudo_x2, label_x2) + torch.nn.MSELoss(reduction='sum')(label_x2, pseudo_x2)
    #     loss_x2 = loss_x2 * 0.5
    #
    #     loss_x3a = torch.nn.MSELoss(reduction='sum')(pseudo_x3_a, label_x3) + torch.nn.MSELoss(reduction='sum')(label_x3, pseudo_x3_a)
    #     loss_x3a = loss_x3a * 0.5
    #     loss_x3b = torch.nn.MSELoss(reduction='sum')(pseudo_x3_b, label_x3) + torch.nn.MSELoss(reduction='sum')(label_x3, pseudo_x3_b)
    #     loss_x3b = loss_x3b * 0.5
    #     loss_x3a_x3b = torch.nn.MSELoss(reduction='sum')(pseudo_x3_a, pseudo_x3_b) + torch.nn.MSELoss(reduction='sum')(pseudo_x3_b, pseudo_x3_a)
    #     loss_x3a_x3b = loss_x3a_x3b * 0.5
    #
    # else:

    loss_x1a = torch.nn.MSELoss(reduction='mean')(pseudo_x1_a, label_x1) + torch.nn.MSELoss(reduction='mean')(label_x1, pseudo_x1_a)
    loss_x1a = loss_x1a*0.5
    loss_x1b = torch.nn.MSELoss(reduction='mean')(pseudo_x1_b, label_x1) + torch.nn.MSELoss(reduction='mean')(label_x1, pseudo_x1_b)
    loss_x1b = loss_x1b*0.5
    loss_x1a_x1b = torch.nn.MSELoss(reduction='mean')(pseudo_x1_a, pseudo_x1_b) + torch.nn.MSELoss(reduction='mean')(pseudo_x1_b, pseudo_x1_a)
    loss_x1a_x1b = loss_x1a_x1b*0.5

    loss_x2 = torch.nn.MSELoss(reduction='mean')(pseudo_x2, label_x2) + torch.nn.MSELoss(reduction='mean')(label_x2, pseudo_x2)
    loss_x2 = loss_x2*0.5

    loss_x3a = torch.nn.MSELoss(reduction='mean')(pseudo_x3_a, label_x3) + torch.nn.MSELoss(reduction='mean')(label_x3, pseudo_x3_a)
    loss_x3a = loss_x3a*0.5
    loss_x3b = torch.nn.MSELoss(reduction='mean')(pseudo_x3_b, label_x3) + torch.nn.MSELoss(reduction='mean')(label_x3, pseudo_x3_b)
    loss_x3b = loss_x3b * 0.5
    loss_x3a_x3b = torch.nn.MSELoss(reduction='mean')(pseudo_x3_a, pseudo_x3_b) + torch.nn.MSELoss(reduction='mean')(pseudo_x3_b, pseudo_x3_a)
    loss_x3a_x3b = loss_x3a_x3b * 0.5

    return loss_x1a, loss_x1b, loss_x1a_x1b, loss_x2, loss_x3a, loss_x3b, loss_x3a_x3b


def constrained_loss2(pseudo_x1_a, pseudo_x1_b, pseudo_x3_a, pseudo_x3_b):

    pseudo_x1_a = F.sigmoid(pseudo_x1_a)
    pseudo_x1_b = F.sigmoid(pseudo_x1_b)
    pseudo_x3_a = F.sigmoid(pseudo_x3_a)
    pseudo_x3_b = F.sigmoid(pseudo_x3_b)

    pseudo_x1_a = -pseudo_x1_a * torch.log(pseudo_x1_a)
    pseudo_x1_b = -pseudo_x1_b * torch.log(pseudo_x1_b)
    pseudo_x3_a = -pseudo_x3_a * torch.log(pseudo_x3_a)
    pseudo_x3_b = -pseudo_x3_b * torch.log(pseudo_x3_b)

    loss_x1a_x1b = torch.nn.MSELoss(reduction='mean')(pseudo_x1_a, pseudo_x1_b) + torch.nn.MSELoss(reduction='mean')(pseudo_x1_b, pseudo_x1_a)
    loss_x1a_x1b = loss_x1a_x1b * 0.5

    loss_x3a_x3b = torch.nn.MSELoss(reduction='mean')(pseudo_x3_a, pseudo_x3_b) + torch.nn.MSELoss(reduction='mean')(pseudo_x3_b, pseudo_x3_a)
    loss_x3a_x3b = loss_x3a_x3b * 0.5

    return loss_x1a_x1b + loss_x3a_x3b


def constrained_loss4(pseudo_x1_a, pseudo_x1_b, pseudo_x3_a, pseudo_x3_b):

    pseudo_x1_a = F.sigmoid(pseudo_x1_a)
    pseudo_x1_b = F.sigmoid(pseudo_x1_b)
    pseudo_x3_a = F.sigmoid(pseudo_x3_a)
    pseudo_x3_b = F.sigmoid(pseudo_x3_b)

    loss_x1a_x1b = pseudo_x1_a*torch.log(pseudo_x1_a / pseudo_x1_b) + pseudo_x1_b*torch.log(pseudo_x1_b / pseudo_x1_a)
    loss_x1a_x1b = loss_x1a_x1b.mean() * 0.5

    loss_x3a_x3b = pseudo_x3_a * torch.log(pseudo_x3_a / pseudo_x3_b) + pseudo_x3_b * torch.log(pseudo_x3_b / pseudo_x3_a)
    loss_x3a_x3b = loss_x3a_x3b.mean() * 0.5

    return loss_x1a_x1b + loss_x3a_x3b


def constrained_loss3(x1, x3, pseudo_x1_a, pseudo_x1_b, pseudo_x3_a, pseudo_x3_b):

    label_x1 = F.sigmoid(x1)
    label_x3 = F.sigmoid(x3)

    pseudo_x1_a = F.sigmoid(pseudo_x1_a)
    pseudo_x1_b = F.sigmoid(pseudo_x1_b)
    pseudo_x3_a = F.sigmoid(pseudo_x3_a)
    pseudo_x3_b = F.sigmoid(pseudo_x3_b)

    pseudo_x1_a = -pseudo_x1_a * torch.log(pseudo_x1_a)
    pseudo_x1_b = -pseudo_x1_b * torch.log(pseudo_x1_b)
    pseudo_x3_a = -pseudo_x3_a * torch.log(pseudo_x3_a)
    pseudo_x3_b = -pseudo_x3_b * torch.log(pseudo_x3_b)

    loss_x1a_x1b = torch.nn.MSELoss(reduction='mean')(pseudo_x1_a, pseudo_x1_b) + torch.nn.MSELoss(reduction='mean')(pseudo_x1_b, pseudo_x1_a)
    loss_x1a_x1b = loss_x1a_x1b * 0.5

    loss_x3a_x3b = torch.nn.MSELoss(reduction='mean')(pseudo_x3_a, pseudo_x3_b) + torch.nn.MSELoss(reduction='mean')(pseudo_x3_b, pseudo_x3_a)
    loss_x3a_x3b = loss_x3a_x3b * 0.5

    loss_x1a = torch.nn.MSELoss(reduction='mean')(pseudo_x1_a, label_x1) + torch.nn.MSELoss(reduction='mean')(label_x1, pseudo_x1_a)
    loss_x1a = loss_x1a*0.5
    loss_x1b = torch.nn.MSELoss(reduction='mean')(pseudo_x1_b, label_x1) + torch.nn.MSELoss(reduction='mean')(label_x1, pseudo_x1_b)
    loss_x1b = loss_x1b*0.5

    loss_x3a = torch.nn.MSELoss(reduction='mean')(pseudo_x3_a, label_x3) + torch.nn.MSELoss(reduction='mean')(label_x3, pseudo_x3_a)
    loss_x3a = loss_x3a*0.5
    loss_x3b = torch.nn.MSELoss(reduction='mean')(pseudo_x3_b, label_x3) + torch.nn.MSELoss(reduction='mean')(label_x3, pseudo_x3_b)
    loss_x3b = loss_x3b * 0.5

    return loss_x1a_x1b + loss_x3a_x3b + loss_x1a + loss_x1b + loss_x3a + loss_x3b