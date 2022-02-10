import torch
import torch.nn as nn
import torch.nn.functional as F


def pseudo_features(x1, x2, x3):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    x1_query = x1
    x2_query = x2
    x3_query = x3

    x1_key = x1
    x2_key = x2
    x3_key = x3

    b, c, h, w = x1.size()

    # Reshape tensors:

    x1_query = x1_query.view(b, c, h * w).permute(0, 2, 1).contiguous()  # b x c x n
    x1_key = x1_key.view(b, c, h * w)

    x2_query = x2_query.view(b, c, h * w).permute(0, 2, 1).contiguous()
    x2_key = x2_key.view(b, c, h * w)

    x3_query = x3_query.view(b, c, h * w).permute(0, 2, 1).contiguous()
    x3_key = x3_key.view(b, c, h * w)

    # Affinity matrices:

    x1_query_x2_key = F.softmax(torch.bmm(x1_query, x2_key), dim=-1)
    x2_query_x1_key = F.softmax(torch.bmm(x2_query, x1_key), dim=-1)

    x2_query_x3_key = F.softmax(torch.bmm(x2_query, x3_key), dim=-1)
    x3_query_x2_key = F.softmax(torch.bmm(x3_query, x2_key), dim=-1)

    x1_query_x3_key = F.softmax(torch.bmm(x1_query, x3_key), dim=-1)

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

    return pseudo_x1_a, pseudo_x1_b, pseudo_x3_a, pseudo_x3_b


# =================================================
# Loss function for DMNet: Difference Minimization
# Network for Semi-supervised
# Segmentation in Medical Images
# ==================================================


def dmnet_bce(A, B):
    ''' This function is the discriminator loss bce.
    Args:
        A:
        B:

    Returns:
    '''
    loss = -B * torch.log(A) - (1.0 - B) * torch.log(1.0 - A)

    return loss.sum()


def dmnet_adv_dis(pred1, pred2, target, label_available):
    ''' This function is the discriminator loss for DMNet (MICCAI 2020).
    Args:
        pred1 (torch:tensor): output of discriminator given output of branch 1
        pred2 (torch:tensor): output of discriminator given ouotput of branch 2
        target (torch:tensor): output of discriminator given label
    Returns:
        Loss (torch:value)
    '''
    # zeros_branch1_disc = torch.zeros_like(pred1).to(device)
    # zeros_branch2_disc = torch.zeros_like(pred2).to(device)
    # ones_label = torch.ones_like(target).to(device)

    # L_bce_branch1 = dmnet_bce(A=pred1, B=zeros_branch1_disc)
    # L_bce_branch2 = dmnet_bce(A=pred2, B=zeros_branch2_disc)

    L_bce_branch1 = dmnet_bce(A=pred1, B=0.0)
    L_bce_branch2 = dmnet_bce(A=pred2, B=0.0)

    if label_available is True:
        # L_bce_label = dmnet_bce(A=target, B=ones_label)
        L_bce_label = dmnet_bce(A=target, B=1.0)
        return L_bce_branch1 + L_bce_branch2 + L_bce_label
    else:
        return L_bce_branch1 + L_bce_branch2


def dmnet_adv_seg(pred1, pred2):
    # ones_branch1 = torch.ones_like(pred1).to(device)
    # ones_branch2 = torch.ones_like(pred2).to(device)
    # l1 = dmnet_bce(A=pred1, B=ones_branch1)
    # l2 = dmnet_bce(A=pred2, B=ones_branch2)
    l1 = dmnet_bce(A=pred1, B=1.0)
    l2 = dmnet_bce(A=pred2, B=1.0)
    return l1+l2


def dice_loss(input, target):
    smooth = 1
    # input = F.softmax(input, dim=1)
    # input = torch.sigmoid(input) #for binary
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum()
    # union = (torch.mul(iflat, iflat) + torch.mul(tflat, tflat)).sum()
    dice_score = (2.*intersection + smooth)/(union + smooth)
    return 1-dice_score


class focal_loss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def kt_loss(student_output_digits, teacher_output_digits, tempearture):
    # The KL Divergence for PyTorch comparing the softmaxs of teacher and student expects the input tensor to be log probabilities!
    knowledge_transfer_loss = nn.KLDivLoss()(F.logsigmoid(student_output_digits / tempearture), torch.sigmoid(teacher_output_digits / tempearture)) * (tempearture * tempearture)
    return knowledge_transfer_loss


def mse(x1, x2):
    loss = nn.MSELoss(reduction='mean')(x1, x2) + nn.MSELoss(reduction='mean')(x2, x1)
    return loss*0.5