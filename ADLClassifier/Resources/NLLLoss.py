import torch
from torch import nn


def NLLLoss(y_pred, y_true):
    return nn.NLLLoss()(torch.log(y_pred), y_true)