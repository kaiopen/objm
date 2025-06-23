from torch.optim import lr_scheduler


SCHEDULER = {
    'StepLR': lr_scheduler.StepLR,
    'CosineAnnealingLR': lr_scheduler.CosineAnnealingLR,
    'CosineAnnealingWarmRestarts': lr_scheduler.CosineAnnealingWarmRestarts
}
