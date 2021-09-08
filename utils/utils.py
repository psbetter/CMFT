from torch.autograd import Function
import numpy as np
from easydict import EasyDict as edict
import yaml

def Config(filename):
    with open(filename, 'r') as f:
        parser = edict(yaml.load(f, Loader=yaml.FullLoader))
    for x in parser:
        print('{}: {}'.format(x, parser[x]))
    return parser

def lr_scheduler(optimizer, init_lr, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def print_args(args):
    message = ['src_address', 'tgt_address', 'src_ns_address']
    log_str = ("================= start ==================\n")
    for arg, content in args.__dict__.items():
        if arg not in message:
            continue
        if args.setting == 'UDA' and arg == 'src_ns_address':
            continue
        log_str += ("{}:{}\n".format(arg, content))
    print(log_str)
    args.out_file.write(log_str+'\n')
    args.out_file.flush()