import torch

def get_optimizer(model, cfg):
    if cfg.train.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    elif cfg.train.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.train.lr)
    elif cfg.train.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    else:
        raise Exception('Optimizer not found.')

    return optimizer