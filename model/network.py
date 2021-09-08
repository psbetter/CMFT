import torch.nn as nn
from torch.autograd import Variable

import model.backbone as backbone
import torch
import torch.nn.functional as F

from model import loss
from utils.tools import update_mem_feat, neighbor_prototype, update_center


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class Resnet(nn.Module):
    def __init__(self, base_net='ResNet50'):
        super(Resnet, self).__init__()
        ## set base network
        self.base_network = backbone.network_dict[base_net]()
        self.output_num = self.base_network.output_num()

    def forward(self, inputs):
        features = self.base_network(inputs)

        return features

class Classifier(nn.Module):
    def __init__(self, base_output_num=256, cfg=None):
        super(Classifier, self).__init__()
        self.bottleneck_layer = nn.Linear(base_output_num, cfg.bottleneck_dim)
        self.fc2 = nn.Linear(cfg.bottleneck_dim, cfg.class_num, bias=False)
        self.class_num = cfg.class_num
        self.temp = cfg.temp
        self.setting = cfg.setting
        if self.setting == 'NIUDA':
            self.fc1 = nn.Linear(cfg.bottleneck_dim, cfg.class_num, bias=False)
            self.share_class_num = cfg.share_class_num
            mask = [1 if i < self.share_class_num else 0 for i in range(self.class_num)]
            self.mask = torch.ByteTensor(mask).cuda()
            self.centroid = torch.zeros(self.class_num, cfg.bottleneck_dim).cuda()

    def forward(self, features):
        features = self.bottleneck_layer(features)
        # MFT
        if self.setting == 'NIUDA':
            outputs1 = self.fc1(features)
            softmax_outputs1 = self.mask_mechanism(outputs1)
            feat_oa = torch.matmul(softmax_outputs1, self.centroid)
            features_aug = features + feat_oa
        else:
            features_aug = features
            outputs1 = None
        # CDA
        x = F.normalize(features_aug)
        outputs2 = self.fc2(x)
        outputs2 = outputs2 / self.temp
        softmax_outputs = F.softmax(outputs2, dim=1)
        return features, features_aug, outputs1, outputs2, softmax_outputs

    def mask_mechanism(self, outputs):
        softmax_outputs = F.softmax(outputs, dim=1)
        _, predict = torch.max(softmax_outputs, 1)
        mask = self.mask == 1
        mask = Variable(mask)
        batch_size = outputs.size()[0]
        mask_expand = mask.expand(batch_size, self.class_num)
        for i in range(batch_size):
            if predict[i] > self.share_class_num-1:
                mask_expand[i][:] = 1
            mask_expand[i][predict[i]] = 0
        outputs = outputs.masked_fill(mask_expand, -1e9)
        softmax_outputs = F.softmax(outputs, dim=1)
        return softmax_outputs

class model(object):
    def __init__(self, cfg, use_gpu=True):
        self.base_net = Resnet(cfg.backbone)
        self.classifier = Classifier(self.base_net.output_num, cfg)

        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = cfg.class_num
        if self.use_gpu:
            self.base_net = self.base_net.cuda()
            self.classifier = self.classifier.cuda()
        self.bottleneck_dim = cfg.bottleneck_dim
        self.class_weight_src = torch.ones(cfg.class_num, ).cuda()
        self.smooth = cfg.smooth
        self.setting = cfg.setting
        self.k = cfg.k
        if self.setting == "NIUDA":
            mask = [1 if i < cfg.share_class_num else 0 for i in range(cfg.class_num)]
            self.mask = torch.ByteTensor(mask).cuda()
        else:
            self.mask = None
    # instance memory
    def init_mem(self, src_loader_len, tgt_loader_len):
        self.im_feat_s = torch.rand(src_loader_len, self.bottleneck_dim).cuda()
        self.im_feat_t = torch.rand(tgt_loader_len, self.bottleneck_dim).cuda()
    # init MIM
    def init_mom_softmax(self, train_loader_len):
        self.momentum_softmax_target = loss.MomentumSoftmax(
            self.class_num, m=train_loader_len)

    def get_loss(self, inputs, labels_source, idx_s, idx_t):
        bs = labels_source.size(0)

        features_ = self.base_net(inputs)
        features, features_aug, outputs1, outputs2, _ = self.classifier(features_)

        if self.setting == "NIUDA":
            feature_source = features_aug.narrow(0, 0, bs // 2)
            outputs_source = outputs2.narrow(0, 0, bs // 2)
            labels_share = labels_source.narrow(0, 0, bs // 2)
            # update centroid memory for MFT
            feature_source4center = features.narrow(0, 0, bs)
            up_center_s = update_center(feature_source4center, labels_source, self.classifier.centroid, self.class_num)
            self.classifier.centroid = up_center_s.detach()
            # lsr loss for MFT
            src_ = loss.CrossEntropyLabelSmooth(reduction='none', num_classes=self.class_num, epsilon=self.smooth)(
                outputs1.narrow(0, 0, inputs.size(0) - (bs // 2)), labels_source)
            weight_src = self.class_weight_src[labels_source].unsqueeze(0)
            classifier_loss = torch.sum(weight_src * src_) / (torch.sum(weight_src).item())
        else:
            feature_source = features_aug.narrow(0, 0, bs)
            outputs_source = outputs2.narrow(0, 0, bs)
            labels_share = labels_source
            classifier_loss = torch.tensor(0.0).cuda()

        feature_target = features_aug.narrow(0, bs, inputs.size(0) - bs)
        outputs_target = outputs2.narrow(0, bs, inputs.size(0) - bs)

        # update instance memory for CDA
        up_feat_s, up_feat_t = update_mem_feat(feature_source, feature_target, self.im_feat_s, self.im_feat_t, idx_s, idx_t)
        self.im_feat_s = up_feat_s.detach()
        self.im_feat_t = up_feat_t.detach()

        # label smooth regular loss
        src_ = loss.CrossEntropyLabelSmooth(reduction='none', num_classes=self.class_num, epsilon=self.smooth)(
            outputs_source, labels_share)
        weight_src = self.class_weight_src[labels_share].unsqueeze(0)
        classifier_loss += torch.sum(weight_src * src_) / (torch.sum(weight_src).item())

        # contrastive loss
        feat_t = neighbor_prototype(feature_target, self.im_feat_t, idx_t, self.k)
        feat_st = neighbor_prototype(feature_source, self.im_feat_t, idx_t, self.k)
        feat_ts = neighbor_prototype(feature_target, self.im_feat_s, idx_s, self.k)
        indomain_loss = loss.contrastive_loss(feature_target, feat_t, t=2)
        crossdomain_loss = loss.contrastive_loss(feature_source, feat_st, t=2)
        crossdomain_loss += loss.contrastive_loss(feature_target, feat_ts, t=2)
        con_loss = indomain_loss + crossdomain_loss

        # MIM loss
        min_entroy_loss = loss.entroy_mim(outputs_target)
        prob_unl = F.softmax(outputs_target, dim=1)
        prob_mean_unl = prob_unl.sum(dim=0) / outputs_target.size(0)
        self.momentum_softmax_target.update(prob_mean_unl.cpu().detach(), outputs_target.size(0)) # update momentum
        momentum_prob_target = (self.momentum_softmax_target.softmax_vector.cuda()) # get momentum probability
        entropy_cond = -torch.sum(prob_mean_unl * torch.log(momentum_prob_target + 1e-5))
        max_entroy_loss = -entropy_cond

        return classifier_loss, con_loss, min_entroy_loss, max_entroy_loss

    def predict(self, inputs):
        features = self.base_net(inputs)
        _, _, _, _, softmax_outputs = self.classifier(features)
        return softmax_outputs

    def set_train(self, mode):
        self.base_net.train(mode)
        self.classifier.train(mode)
        self.is_train = mode
