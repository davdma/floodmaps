import torch

def save_checkpoint(path, model, optimizer, epoch, scheduler=None, early_stopper=None, beta_scheduler=None, extra=None):
    """
    Save a checkpoint of the model, optimizer, scheduler, and stopper for later resuming training.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if early_stopper is not None:
        # should not be checkpointing if already stopped
        if early_stopper.is_stopped():
            raise ValueError("Cannot save checkpoint if early stopper is already stopped.")
        checkpoint['early_stopper_state_dict'] = early_stopper.state_dict()
    if beta_scheduler is not None:
        checkpoint['beta_scheduler_state_dict'] = beta_scheduler.state_dict()
    if extra is not None:
        checkpoint['extra'] = extra
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, early_stopper=None, beta_scheduler=None):
    """Load model, optimizer, scheduler, and early stopper from checkpoint.
    Can use without kwargs to just load the model state dict from save."""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if early_stopper is not None and 'early_stopper_state_dict' in checkpoint:
        early_stopper.load_state_dict(checkpoint['early_stopper_state_dict'])
    if beta_scheduler is not None and 'beta_scheduler_state_dict' in checkpoint:
        beta_scheduler.load_state_dict(checkpoint['beta_scheduler_state_dict'])
    return checkpoint
