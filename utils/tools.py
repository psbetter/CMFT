import os
import random
import numpy as np
import torch
from torch.autograd import Variable

# def seed_everything(seed=2021):
#
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
#     np.random.seed(seed)  # Numpy module.
#     random.seed(seed)  # Python random module.
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True

def seed_everything(seed=2021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def mask_outputs(outputs, mask, class_num, is_Training=True):
    mask = mask == 0
    if is_Training:
        mask = Variable(mask)
    batch_size = outputs.size()[0]
    mask_expand = mask.expand(batch_size, class_num)
    if is_Training:
        outputs = outputs.masked_fill(mask_expand, -1e9)
    else:
        outputs = outputs.masked_fill_(mask_expand, -1e9)
    return outputs

def update_center(feature_source, label_source, center, class_num):
    n_class = class_num

    onehot_s = torch.eye(n_class)[label_source].cuda()
    curr_center = torch.mm(feature_source.t(), onehot_s) / (onehot_s.sum(dim=0) + 1e-8)

    # Moving Centroid
    decay = 0.3
    center = (1 - decay) * center + decay * curr_center.t().clone()

    return center

def update_mem_feat(feature_source, feature_target, mem_feat_s, mem_feat_t, idx_s, idx_t):
    feature_source = feature_source / torch.norm(feature_source, p=2, dim=1, keepdim=True)
    # feature_source = F.normalize(feature_source, dim=1)
    mem_feat_s[idx_s] = feature_source.clone()

    feature_target = feature_target / torch.norm(feature_target, p=2, dim=1, keepdim=True)
    # feature_target = F.normalize(feature_target, dim=1)
    mem_feat_t[idx_t] = feature_target.clone()

    return mem_feat_s, mem_feat_t

def neighbor_prototype(feature, mem_feat, idx, k):
    dis = -torch.mm(feature.detach(), mem_feat.t())  # [36, 256]x[256, 4365]=[36, 4365]
    for di in range(dis.size(0)):
        dis[di, idx[di]] = torch.max(dis)
    _, p1 = torch.sort(dis, dim=1)
    w = torch.zeros(feature.size(0), mem_feat.size(0)).cuda()
    # k = 5
    for wi in range(w.size(0)):
        for wj in range(k):
            w[wi][p1[wi, wj]] = 1 / k
    feat_t = w.mm(mem_feat)
    return feat_t

# def neighbor_cluster(feature, mem_feat):
#     dis = -torch.mm(feature.detach(), mem_feat.t())
#
#     _, p1 = torch.sort(dis, dim=1)
#     w = torch.zeros(feature.size(0), mem_feat.size(0)).cuda()
#     k = 10
#     for wi in range(w.size(0)):
#         for wj in range(k):
#             w[wi][p1[wi, wj]] = 1 / k
#     feat_t = w.mm(mem_feat)
#     return feat_t