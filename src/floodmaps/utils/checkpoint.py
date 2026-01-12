import torch

def save_checkpoint(path, model, optimizer, epoch, scheduler=None, early_stopper=None, 
                    beta_scheduler=None, discriminator=None, optimizer_d=None, extra=None):
    """
    Save a checkpoint of the model, optimizer, scheduler, and stopper for later resuming training.
    
    Parameters
    ----------
    path : str
        Path to save the checkpoint
    model : nn.Module
        Generator/main model to save
    optimizer : torch.optim.Optimizer
        Generator optimizer
    epoch : int
        Current epoch number
    scheduler : torch.optim.lr_scheduler, optional
        Learning rate scheduler
    early_stopper : ADEarlyStopper, optional
        Early stopping handler
    beta_scheduler : BetaScheduler, optional
        Beta annealing scheduler for VAE
    discriminator : nn.Module, optional
        Discriminator model for GAN training
    optimizer_d : torch.optim.Optimizer, optional
        Discriminator optimizer
    extra : dict, optional
        Additional data to save
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
    if discriminator is not None:
        checkpoint['discriminator_state_dict'] = discriminator.state_dict()
    if optimizer_d is not None:
        checkpoint['optimizer_d_state_dict'] = optimizer_d.state_dict()
    if extra is not None:
        checkpoint['extra'] = extra
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, early_stopper=None, 
                    beta_scheduler=None, discriminator=None, optimizer_d=None):
    """Load model, optimizer, scheduler, and early stopper from checkpoint.
    
    Can use without kwargs to just load the model state dict from save.
    Maintains backward compatibility with pure CVAE checkpoints (no discriminator).
    
    Parameters
    ----------
    path : str
        Path to the checkpoint file
    model : nn.Module
        Generator/main model to load weights into
    optimizer : torch.optim.Optimizer, optional
        Generator optimizer
    scheduler : torch.optim.lr_scheduler, optional
        Learning rate scheduler
    early_stopper : ADEarlyStopper, optional
        Early stopping handler
    beta_scheduler : BetaScheduler, optional
        Beta annealing scheduler for VAE
    discriminator : nn.Module, optional
        Discriminator model for GAN training
    optimizer_d : torch.optim.Optimizer, optional
        Discriminator optimizer
    
    Returns
    -------
    dict
        The loaded checkpoint dictionary
    """
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
    # GAN-specific: load discriminator if available in checkpoint
    if discriminator is not None and 'discriminator_state_dict' in checkpoint:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    if optimizer_d is not None and 'optimizer_d_state_dict' in checkpoint:
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    return checkpoint
