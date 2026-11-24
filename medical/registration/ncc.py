"""Minibatch Pearson NCC and standard-normal moment matching."""
import math

import torch


def ncc_options(overrides=None):
    options = dict(moment_weight=.005, epsilon=1e-8)
    if overrides is not None and (not isinstance(overrides, dict) or set(overrides)-options.keys()):
        raise ValueError('Unknown ncc settings')
    options.update(overrides or {})
    for name in ("moment_weight", "epsilon"):
        value=options[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid ncc.{name}')
    if not 0 < options['epsilon'] < 1e-3:
        raise ValueError('ncc.epsilon must be in (0,1e-3)')
    return options


def correlation(prediction, target, epsilon):
    a, b = prediction-prediction.mean(), target-target.mean()
    return (a*b).mean()/((a.square().mean()+epsilon)*(b.square().mean()+epsilon)).sqrt()


def endpoint_statistics(source, target, source_truth, target_truth, *, epsilon, moment_weight):
    """Tensor-only endpoint NCC and moment matching; options validated by caller."""
    source_ncc = correlation(source, source_truth, epsilon)
    target_ncc = correlation(target, target_truth, epsilon)
    ncc_loss = 2-source_ncc-target_ncc
    means = torch.stack((source.mean(), target.mean()))
    variances = torch.stack((source.var(unbiased=False), target.var(unbiased=False)))
    stable_variances = variances+epsilon
    # KL between moment-matched univariate Gaussian and N(0,1), not a VAE.
    moment_loss = .5*(means.square()+stable_variances-stable_variances.log()-1).sum()
    weighted = moment_weight*moment_loss
    return dict(endpoint_ncc_loss=ncc_loss, source_ncc=source_ncc, target_ncc=target_ncc,
        moment_loss=moment_loss, weighted_moments=weighted,
        moment_weight=source.new_tensor(moment_weight),
        source_mean=means[0], target_mean=means[1],
        source_variance=variances[0], target_variance=variances[1])
