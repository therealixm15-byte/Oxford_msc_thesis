import torch
import torch.nn as nn
import torch.nn.functional as F


def interval_metrics(samples, targets, coverage_level=0.95):
    """
    Calculate the coverage and width of the prediction intervals.
    Args:
        samples: [num_samples, num_nodes] tensor of model predictions
        targets: [num_nodes] tensor of true values
        coverage_level: float, the desired coverage level (default: 0.95)
    Returns:
        coverage: float, the proportion of true values that fall within the prediction intervals
        width: float, the average width of the prediction intervals
    """
    lower_bound = torch.quantile(samples, (1 - coverage_level) / 2, dim=0)
    upper_bound = torch.quantile(samples, 1 - (1 - coverage_level) / 2, dim=0)

    coverage = ((targets >= lower_bound) & (targets <= upper_bound)).float().mean().item()
    width = (upper_bound - lower_bound).mean().item()

    return coverage, width


def crps(samples, targets):
    """
    Calculate sample-based CRPS for posterior predictive samples.
    Args:
        samples:
            Tensor of shape [num_samples, num_nodes] containing samples from
            the posterior predictive distribution.
        targets:
            Tensor of shape [num_nodes] containing the true values.
    Returns:
        Average CRPS across all nodes.
    """
    num_samples = samples.shape[0]
    # First term: E|X - y|
    absolute_error_term = torch.abs(samples - targets.unsqueeze(0)).mean(dim=0)
    # Second term: 0.5 * E|X - X'|
    sorted_samples, _ = torch.sort(samples, dim=0)
    coefficients = (2.0 * torch.arange(1,num_samples + 1,device=samples.device,dtype=samples.dtype)- num_samples- 1.0).unsqueeze(1)
    pairwise_difference_term = (sorted_samples * coefficients).sum(dim=0) / (num_samples ** 2)
    crps_values = (absolute_error_term- pairwise_difference_term)
    return crps_values.mean().item()