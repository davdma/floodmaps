import torch
import torch.nn as nn

def get_weight_decay_param_groups(
        model, 
        weight_decay=1e-5, 
        skip_list=(nn.BatchNorm2d),
        ignore_bias=True):
    """Skip modules in skip_list and ignore bias parameters"""
    decay = []
    no_decay = []
    for module in model.modules():
        if isinstance(module, skip_list):
            no_decay.extend([p for p in module.parameters(recurse=False)])
            continue

        if ignore_bias:
            for p_name, p in module.named_parameters(recurse=False):
                if "bias" in p_name:
                    no_decay.append(p)
                else:
                    decay.append(p)
        else:
            decay.extend([p for p in module.parameters(recurse=False)])

    return [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay}]

def get_optimizer(model, cfg):
    if cfg.train.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    elif cfg.train.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.train.lr)
    elif cfg.train.optimizer == 'AdamW':
        # Do not weight decay batch norm and bias parameters
        param_groups = get_weight_decay_param_groups(model, weight_decay=cfg.train.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=cfg.train.lr)
    else:
        raise Exception('Optimizer not found.')

    return optimizer