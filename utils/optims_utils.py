import math
import numpy as np
from torch import nn
from torch.optim.adam import Adam
from torch.optim.sgd import SGD


def split_params(model: nn.Module):
    param_other, param_weight_decay, param_bias = list(), list(), list()  # optimizer parameter groups
    for k, v in model.named_parameters():
        if v.requires_grad:
            if '.bias' in k:
                param_bias.append(v)  # biases
            elif '.weight' in k and '.bn' not in k:
                param_weight_decay.append(v)  # apply weight decay
            else:
                param_other.append(v)  # all else
    return param_weight_decay, param_bias, param_other


def split_optimizer(model: nn.Module, cfg: dict):
    param_weight_decay, param_bias, param_other = split_params(model)
    if cfg['optimizer'] == 'Adam':
        optimizer = Adam(param_other, lr=cfg['lr'], betas=(cfg['momentum'], 0.999))
    elif cfg['optimizer'] == 'SGD':
        optimizer = SGD(param_other, lr=cfg['lr'], momentum=cfg['momentum'], nesterov=True)
    else:
        raise NotImplementedError("optimizer {:s} is not support!".format(cfg['optimizer']))
    optimizer.add_param_group(
        {'params': param_weight_decay, 'weight_decay': cfg['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': param_bias})
    return optimizer


class EpochWarmUpCosineDecayLRAdjust(object):
    def __init__(self,
                 init_lr=0.01,
                 epochs=300,
                 warm_up_epoch=1,
                 iter_per_epoch=1000,
                 gamma=1.0,
                 alpha=0.1,
                 bias_idx=None,
                 momentum=0.937):
        assert warm_up_epoch < epochs and epochs - warm_up_epoch >= 1
        self.init_lr = init_lr
        self.warm_up_epoch = warm_up_epoch
        self.iter_per_epoch = iter_per_epoch
        self.gamma = gamma
        self.alpha = alpha
        self.bias_idx = bias_idx
        self.flag = np.array([warm_up_epoch, epochs]).astype(np.int)
        self.flag = np.unique(self.flag)
        self.warm_up_iter = self.warm_up_epoch * iter_per_epoch
        self.momentum = momentum

    def cosine(self, current, total):
        return ((1 + math.cos(current * math.pi / total)) / 2) ** self.gamma * (1 - self.alpha) + self.alpha

    def get_lr(self, ite, epoch):
        current_iter = self.iter_per_epoch * epoch + ite
        if epoch < self.warm_up_epoch:
            up_lr = np.interp(current_iter, [0, self.warm_up_iter], [0, self.init_lr])
            down_lr = np.interp(current_iter, [0, self.warm_up_iter], [0.1, self.init_lr])
            momentum = np.interp(current_iter, [0, self.warm_up_iter], [0.9, self.momentum])
            return up_lr, down_lr, momentum
        num_pow = (self.flag <= epoch).sum() - 1
        cosine_ite = (epoch - self.flag[num_pow] + 1)
        cosine_all_ite = (self.flag[num_pow + 1] - self.flag[num_pow])
        cosine_weights = self.cosine(cosine_ite, cosine_all_ite)
        lr = cosine_weights * self.init_lr
        return lr, lr, self.momentum

    def __call__(self, optimizer, ite, epoch):
        ulr, dlr, momentum = self.get_lr(ite, epoch)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = dlr if self.bias_idx is not None and i == self.bias_idx else ulr
            if "momentum" in param_group:
                param_group['momentum'] = momentum
        return ulr, dlr, momentum
