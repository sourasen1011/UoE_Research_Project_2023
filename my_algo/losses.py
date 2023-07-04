import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

def nll_logistic_hazard(phi: Tensor, idx_durations: Tensor, events: Tensor) -> Tensor:
    """Negative log-likelihood of the discrete time hazard parametrized model LogisticHazard [1].
    
    Arguments:
        phi {torch.tensor} -- Estimates in (-inf, inf), where hazard = sigmoid(phi).
        idx_durations {torch.tensor} -- Event times represented as indices.
        events {torch.tensor} -- Indicator of event (1.) or censoring (0.).
            Same length as 'idx_durations'.
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    
    Returns:
        torch.tensor -- The negative log-likelihood.

    References:
    [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """
    if phi.shape[1] <= idx_durations.max(): # Kvamme et al uses lesser than equal to
        raise ValueError(f"Network output `phi` is too small for `idx_durations`."+
                        f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`,"+
                        f" but got `phi.shape[1] = {phi.shape[1]}`")
    # if events.dtype is torch.bool:
    #     events = events.float()
    events = events.view(-1, 1)
    idx_durations = idx_durations.view(-1, 1)
    y_bce = torch.zeros_like(phi).type(torch.int64).scatter(1, idx_durations.type(torch.int64), events.type(torch.int64))
    bce = F.binary_cross_entropy_with_logits(phi, y_bce.type(torch.float32), reduction='none')
    loss = bce.cumsum(1).gather(1, idx_durations.type(torch.int64)).view(-1)
    return loss.mean()