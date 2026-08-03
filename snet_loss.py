"""Likelihoods and training losses used by ADFlow.

The adaptive base distribution follows Eqs. (10)-(16) in the paper: a
zero-mean mixture of isotropic Gaussians for normal features, together with a
unit-covariance Gaussian for pseudo-anomalous features.
"""

import math

import torch
import torch.nn.functional as F


LOG_2PI = math.log(2.0 * math.pi)


def _prepare_mixture(z, variances, weights):
    if variances is None or weights is None:
        raise ValueError("Adaptive distribution requires variances and weights")

    variances = variances.reshape(-1).to(device=z.device, dtype=z.dtype)
    weights = weights.reshape(-1).to(device=z.device, dtype=z.dtype)
    if variances.numel() != weights.numel():
        raise ValueError("variances and weights must have the same length")

    variances = variances.clamp_min(torch.finfo(z.dtype).eps)
    weights = weights.clamp_min(torch.finfo(z.dtype).eps)
    weights = weights / weights.sum()
    return variances, weights


def normal_base_log_prob(z, variances=None, weights=None, distribution="adaptive"):
    """Return log p_Z(z) for either the AD mixture or N(0, I)."""
    feature_dim = z.shape[1]
    squared_norm = z.square().sum(dim=1, keepdim=True)

    if distribution == "standard":
        return -0.5 * (squared_norm.squeeze(1) + feature_dim * LOG_2PI)
    if distribution != "adaptive":
        raise ValueError("distribution must be 'adaptive' or 'standard'")

    variances, weights = _prepare_mixture(z, variances, weights)
    component_log_prob = (
        weights.log().unsqueeze(0)
        - 0.5 * squared_norm / variances.unsqueeze(0)
        - 0.5 * feature_dim * variances.log().unsqueeze(0)
        - 0.5 * feature_dim * LOG_2PI
    )
    return torch.logsumexp(component_log_prob, dim=1)


def get_logp(
    feature_dim,
    z,
    logdet_j,
    variances=None,
    weights=None,
    distribution="adaptive",
):
    """Return log p_Y(y) using the change-of-variables formula."""
    if z.ndim != 2 or z.shape[1] != feature_dim:
        raise ValueError(
            "Expected z with shape (N, {}), got {}".format(feature_dim, tuple(z.shape))
        )
    return normal_base_log_prob(z, variances, weights, distribution) + logdet_j


def adflow_training_loss(
    feature_dim,
    z,
    logdet_j,
    anomaly_mask,
    variances=None,
    weights=None,
    distribution="adaptive",
    lambda_bce=0.2,
):
    """Compute the per-patch ADFlow objective and its two components.

    NLL is evaluated on normal patches. The BCE term uses the log-likelihood
    ratio log p_Zn(z) - log p_Za(z), which is numerically equivalent to the
    probability ratio in Eq. (14) without explicitly forming probabilities.
    Losses are normalized by feature dimension so curves remain comparable
    across feature scales.
    """
    normal_base = normal_base_log_prob(z, variances, weights, distribution)
    normal_log_prob = normal_base + logdet_j

    if distribution == "adaptive":
        variances, _ = _prepare_mixture(z, variances, weights)
        anomaly_mean = variances.max().sqrt().detach()
    else:
        anomaly_mean = z.new_tensor(1.0)

    anomaly_base = -0.5 * (
        (z - anomaly_mean).square().sum(dim=1) + feature_dim * LOG_2PI
    )

    anomaly_target = anomaly_mask.reshape(-1).to(device=z.device, dtype=z.dtype)
    anomaly_target = anomaly_target.clamp(0.0, 1.0)
    normal_target = 1.0 - anomaly_target

    nll_per_patch = -normal_log_prob / feature_dim
    bce_per_patch = F.binary_cross_entropy_with_logits(
        normal_base - anomaly_base,
        normal_target,
        reduction="none",
    )

    # Eq. (13) is defined for normal features. Keep anomalous patches in the
    # BCE term only, and normalize the NLL by the number of normal patches.
    normal_count = normal_target.sum().clamp_min(1.0)
    nll = (nll_per_patch * normal_target).sum() / normal_count
    bce = bce_per_patch.mean()
    total = nll + lambda_bce * bce
    return total, nll.detach(), bce.detach()


def get_logp_NA(
    epoch,
    feature_dim,
    z,
    logdet_j,
    anomaly_mask,
    variances,
    weights,
):
    """Compatibility wrapper for the incomplete original training code."""
    del epoch, anomaly_mask
    return get_logp(feature_dim, z, logdet_j, variances, weights, "adaptive")
