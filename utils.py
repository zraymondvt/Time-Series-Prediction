import torch

def log_cosh_loss(y_true, y_pred):
    loss = torch.log(torch.cosh(y_pred - y_true))
    return torch.mean(loss)
