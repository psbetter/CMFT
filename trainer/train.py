import datetime
import os.path as osp
import argparse
from sklearn.metrics import f1_score
from torch import optim
from tqdm import trange
from torch.autograd import Variable
import torch

import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from preprocess.data_load import data_load
from utils.utils import lr_scheduler, print_args, Config
from utils.tools import seed_everything, mask_outputs

# evaluation on test data
def evaluate(model_instance, input_loader):
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)

    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True

    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        if model_instance.use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        probabilities = model_instance.predict(inputs)

        probabilities = probabilities.data.float()
        labels = labels.data.float()
        if first_test:
            all_probs = probabilities
            all_labels = labels
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
            all_labels = torch.cat((all_labels, labels), 0)

    # if model_instance.mask is not None:
    #     all_probs = mask_outputs(all_probs, model_instance.mask, model_instance.class_num)

    _, predict = torch.max(all_probs, 1)
    accuracy = float(torch.sum(torch.squeeze(predict).float() == all_labels)) / float(all_labels.size()[0])
    f1 = f1_score(all_labels.cpu(), predict.cpu(), average='macro')
    model_instance.set_train(ori_train_state)
    return accuracy, f1

def train(args, cfg, model_instance, optimizer_g, optimizer_f):
    best_acc, best_f1, best_index = 0.0, 0.0, 0.0
    best_model = None
    # init data and other param
    dset_loaders = data_load(args)
    train_source_loader = dset_loaders["source"]
    train_target_loader = dset_loaders["target"]
    test_target_loader = dset_loaders["test"]
    if cfg.setting == 'NIUDA':
        train_source_ns_loader = dset_loaders["source_ns"]
    else:
        train_source_ns_loader = None
    model_instance.init_mom_softmax(len(train_target_loader))
    model_instance.init_mem(len(dset_loaders["source"].dataset), len(dset_loaders["target"].dataset))
    # start train steps
    model_instance.set_train(True)
    for i in trange(cfg.max_iter):
        optimizer_g = lr_scheduler(optimizer_g, init_lr=args.lr * 0.1, iter_num=i, max_iter=cfg.max_iter)
        optimizer_f = lr_scheduler(optimizer_f, init_lr=args.lr, iter_num=i, max_iter=cfg.max_iter)

        # load a batch of data
        if i % len(train_source_loader) == 0:
            iter_source = iter(train_source_loader)
        if i % len(train_target_loader) == 0:
            iter_target = iter(train_target_loader)
            model_instance.momentum_softmax_target.reset()
        inputs_source, labels_source, idx_s = iter_source.next()
        inputs_target, _, idx_t = iter_target.next()
        inputs_source, inputs_target, labels_source = Variable(inputs_source).cuda(), Variable(
            inputs_target).cuda(), Variable(labels_source).cuda()
        if cfg.setting == "NIUDA":
            if i % len(train_source_ns_loader) == 0:
                iter_source_ns = iter(train_source_ns_loader)
            inputs_source_ns, labels_source_ns, _ = iter_source_ns.next()
            inputs_source_ns, labels_source_ns = Variable(inputs_source_ns).cuda(), Variable(labels_source_ns).cuda()

        lam = i / cfg.max_iter

        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        if cfg.setting == "NIUDA":
            inputs = torch.cat((inputs_source, inputs_source_ns, inputs_target), dim=0)
            labels_source = torch.cat((labels_source, labels_source_ns), dim=0)
        else:
            inputs = torch.cat((inputs_source, inputs_target), dim=0)
        classifier_loss, contrastive_loss, min_entroy_loss, max_entroy_loss  = model_instance.get_loss(inputs, labels_source, idx_s, idx_t)
        total_loss = classifier_loss + cfg.lam_con * lam * contrastive_loss + cfg.lam_mim.min * min_entroy_loss + cfg.lam_mim.max * max_entroy_loss
        total_loss.backward()
        optimizer_g.step()
        optimizer_f.step()

        # eval in the train
        if (i + 1) % cfg.eval_interval == 0:
            acc, f1 = evaluate(model_instance, test_target_loader)
            if acc > best_acc:
                best_acc = acc
                best_f1 = f1
                best_index = i
            log_str = "iter: {:05d}, acc: {:.5f}, f1: {:.5f}".format(i, acc, f1)
            args.out_file.write(log_str + "\n")
            args.out_file.flush()
            print(log_str)
    print('finish train')
    log_str = "The Best --> iter: {:05d}, acc: {:.5f}, f1: {:.5f}".format(best_index, best_acc, best_f1)
    args.out_file.write(log_str + "\n")
    args.out_file.flush()
    # if best_model is not None:
    #     output_dir = '../results/' + cfg.dataset + '/' + cfg.setting
    #     with open(osp.join(output_dir, 'MDD65_OLTR_AC.pkl'), 'wb') as f:
    #         torch.save(data, f)

if __name__ == '__main__':
    from model.network import model
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config/officehome_NIUDA.yml', help='all sets of configuration parameters')
    parser.add_argument('--method', type=str, default='CMFT', help='method')
    parser.add_argument('--src_address', default='../data/officehome/Art_small_25.txt', type=str,
                        help='address of image list of source dataset')
    parser.add_argument('--tgt_address', default='../data/officehome/Clipart_small_25.txt', type=str,
                        help='address of image list of target dataset')
    parser.add_argument('--src_ns_address', default='../data/officehome/Art_noshare_40.txt', type=str,
                        help='address of image list of target dataset')
    parser.add_argument('--lr', default='0.01', type=float, help='learning rate')
    parser.add_argument('--num_workers', default=4, type=int, help='num_workers')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # load config
    cfg = Config(args.config)
    args.batch_size = cfg.batch_size
    args.setting = cfg.setting

    seed_everything(seed=cfg.seed)
    # init model
    model_instance = model(cfg, use_gpu=True)

    optimizer_g = optim.SGD(model_instance.base_net.parameters(), lr=args.lr * 0.1)
    optimizer_f = optim.SGD(model_instance.classifier.parameters(), lr=args.lr)

    log_file = args.src_address.split('/')[-1][0] + args.tgt_address.split('/')[-1][0] + '.txt'
    output_dir = '../results/' + cfg.dataset + '/' + cfg.setting
    if not osp.exists(output_dir):
        os.system('mkdir -p ' + output_dir)
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    args.out_file = open(osp.join(output_dir, log_file), "a")
    args.out_file.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
    args.out_file.flush()

    print_args(args)
    train(args, cfg, model_instance, optimizer_g=optimizer_g, optimizer_f=optimizer_f)

