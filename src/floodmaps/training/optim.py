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


def get_optimizer_with_ad(model, cfg, ad_cfg):
    """Create optimizer with separate learning rates for classifier and autodespeckler.
    
    Parameters
    ----------
    model : SARWaterDetector
        Model with classifier and autodespeckler components.
    cfg : obj
        Main training config with cfg.train.lr for classifier and cfg.train.optimizer.
    ad_cfg : obj
        Autodespeckler config with ad_cfg.train.lr for autodespeckler.
    
    Returns
    -------
    torch.optim.Optimizer
        Optimizer with separate parameter groups for classifier and autodespeckler.
    """
    # Create parameter groups with different learning rates
    classifier_params = list(model.classifier.parameters())
    autodespeckler_params = list(model.autodespeckler.parameters())
    
    if cfg.train.optimizer == 'Adam':
        param_groups = [
            {'params': classifier_params, 'lr': cfg.train.lr},
            {'params': autodespeckler_params, 'lr': ad_cfg.train.lr}
        ]
        optimizer = torch.optim.Adam(param_groups)
    elif cfg.train.optimizer == 'SGD':
        param_groups = [
            {'params': classifier_params, 'lr': cfg.train.lr},
            {'params': autodespeckler_params, 'lr': ad_cfg.train.lr}
        ]
        optimizer = torch.optim.SGD(param_groups)
    elif cfg.train.optimizer == 'AdamW':
        # For AdamW, handle weight decay separately for classifier and autodespeckler
        cls_decay_groups = get_weight_decay_param_groups(model.classifier, weight_decay=cfg.train.weight_decay)
        ad_decay_groups = get_weight_decay_param_groups(model.autodespeckler, weight_decay=cfg.train.weight_decay)
        
        # Set learning rates for each group
        for group in cls_decay_groups:
            group['lr'] = cfg.train.lr
        for group in ad_decay_groups:
            group['lr'] = ad_cfg.train.lr
        
        param_groups = cls_decay_groups + ad_decay_groups
        optimizer = torch.optim.AdamW(param_groups)
    else:
        raise Exception('Optimizer not found.')

    return optimizer