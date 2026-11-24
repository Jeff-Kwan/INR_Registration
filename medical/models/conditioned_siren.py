"""Single-field architecture ablations with an explicit time input or FiLM."""
from dataclasses import dataclass

import torch
from torch import nn

from medical.models.siren import Siren
from medical.models.siren_functional import PreparedSiren, siren_parameters

CONDITIONING_MODES = ('xyzt', 'film')


def time_derivative(parameters, conditioning):
    """Exact time derivatives of the affine maps after folding in conditioning.

    A fourth input column only affects the first affine's bias. FiLM acts on
    each sine activation and folds into the following affine, including output.
    The learned conditioning slopes are with respect to 2t-1.
    """
    architecture, slopes = conditioning
    first_weight, first_bias = parameters[0]
    if architecture == 'xyzt':
        return ((torch.zeros_like(first_weight), 2 * slopes), *(
            (torch.zeros_like(weight), torch.zeros_like(bias)) for weight, bias in parameters[1:]
        ))
    return ((torch.zeros_like(first_weight), torch.zeros_like(first_bias)), *(
        (2 * weight * modulation[0], 2 * (weight @ modulation[1]))
        for (weight, _), modulation in zip(parameters[1:], slopes)
    ))


def conditioned_affines(parameters, direction, time):
    """Fold the input or feature modulation into a spatial-only evaluation."""
    offset = time - 0.5
    return tuple((weight + offset * dw, bias + offset * db)
                 for (weight, bias), (dw, db) in zip(parameters, direction))


@dataclass(frozen=True)
class ConditionedField:
    parameters: tuple
    direction: tuple
    omegas: tuple

    def at(self, time):
        return PreparedSiren(conditioned_affines(self.parameters, self.direction, time),
                             self.direction, self.omegas)


class ConditionedSiren(nn.Module):
    def __init__(self, architecture, **model):
        super().__init__()
        if architecture not in CONDITIONING_MODES:
            raise ValueError(f'Unknown time-conditioned architecture: {architecture}')
        self.architecture = architecture
        self.model_config = dict(architecture=architecture, **model)
        self.spatial = Siren(dimensions=3, **model)
        shape = ((model['width'],) if architecture == 'xyzt'
                 else (model['layers'], 2, model['width']))
        self.slopes = nn.Parameter(torch.zeros(shape))

    def training_arguments(self):
        parameters, omegas = siren_parameters(self.spatial)
        return parameters, (self.architecture, self.slopes), omegas

    def prepare(self, *, temporal=True, compile_parameters=False):
        # Recovery compiles the derivative evaluation on each spatial chunk.
        # The small per-time affine preparation runs once outside that loop.
        parameters, conditioning, omegas = self.training_arguments()
        return ConditionedField(parameters, time_derivative(parameters, conditioning), omegas)
