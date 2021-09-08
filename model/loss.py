import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)

        if self.reduction:
            return loss.mean()
        else:
            return loss

def contrastive_loss(feature, center, t=5):
    feat_reshape = feature.unsqueeze(1).repeat(1, center.size(0), 1)
    prods = torch.exp(torch.sum(feat_reshape * center, dim=-1) / t)
    sum = torch.sum(prods, dim=-1)
    p = prods / sum.unsqueeze(1)
    contrastive_loss = -torch.sum(p * torch.log(p)) / p.size(0)

    return contrastive_loss

def cluster_cont_loss(feature, center, t=5):
    feat_reshape = feature.unsqueeze(1).repeat(1, center.size(0), 1)
    prods = torch.exp(torch.sum(feat_reshape * center, dim=-1) / t)
    sum = torch.sum(prods, dim=-1)
    p = prods / sum.unsqueeze(1)
    contrastive_loss = -torch.sum(p * torch.log(p)) / p.size(0)

    return contrastive_loss

def weigh_cont_loss(feature, inst, center, t=5):
    # feat_reshape = feature.unsqueeze(1).repeat(1, inst.size(0), 1)
    prods = torch.exp(torch.sum(feature.mm(inst.t()), dim=-1) / t)
    sum = torch.sum(prods, dim=-1)
    p = prods / sum

    # feat_reshape = feature.unsqueeze(1).repeat(1, center.size(0), 1)
    prods = torch.exp(torch.sum(feature.mm(center.t()), dim=-1) / 1)
    sum = torch.sum(prods, dim=-1)
    p_c = prods / sum

    w_contrastive_loss = -torch.sum(p_c * torch.log(p))

    return w_contrastive_loss

def entroy_mim(x, eps=1e-5):
    p = F.softmax(x, dim=-1)
    entroy = -torch.mean(torch.sum(p * torch.log(p + eps), 1))
    return entroy

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()

        return torch.sum(weight.view(-1, 1) * nn.BCEWithLogitsLoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)

class MomentumSoftmax:
    def __init__(self, num_class, m=1):
        self.softmax_vector = torch.zeros(num_class).detach() + 1.0 / num_class
        self.m = m
        self.num = m

    def update(self, mean_softmax, num=1):
        self.softmax_vector = (
            (self.softmax_vector * self.num) + mean_softmax * num
        ) / (self.num + num)
        self.num += num

    def reset(self):
        # print(self.softmax_vector)
        self.num = self.m

def adentropy(F1, feat, lamda, eta=1.0):
    _, _, out_softmax = F1(feat, reverse=True, eta=eta)
    loss_adent = lamda * torch.mean(torch.sum(out_softmax *
                                              (torch.log(out_softmax + 1e-5)), 1))
    return loss_adent
