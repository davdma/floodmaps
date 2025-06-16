import torch

def get_scheduler(optimizer, cfg):
    """Supports epoch stepping schedulers (batch step needs to be implemented)."""
    if cfg.train.LR_scheduler is None or cfg.train.LR_scheduler == 'Constant':
        scheduler = None
    elif cfg.train.LR_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg.train.LR_patience)
    elif cfg.train.LR_scheduler == 'CosAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.train.LR_T_max, eta_min=0.000001)
    else:
        raise Exception('Scheduler not found.')
    return scheduler