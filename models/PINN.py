from typing import Callable
import torch
import torch.nn as nn
from lib.utils import *
from lib.modules import *
from models.MLP import MLP, MLPConfig


@dataclass(kw_only=True)
class PINNConfig(Config):
    eq: Callable[[nn.Module, torch.Tensor], torch.Tensor]
    t_eval: torch.Tensor  # must be same device
    alpha: float = 0.1
    p_params: List[nn.Parameter] = field(default_factory=list)


class PINN(Module):
    def __init__(self, c: PINNConfig):
        self.save_config(c, ignore="p_params")
        for i, param in enumerate(c.p_params):
            setattr(self, f"pp_{i}", param)

    def physics_loss(self):
        with torch.set_grad_enabled(True):
            errs = self.alpha * self.eq(self, self.t_eval)
            return torch.mean(errs**2)

    def data_loss(self, *args):
        return super().loss(*args)

    def loss(self, *args):
        return self.data_loss(*args) + self.physics_loss()


@dataclass(kw_only=True)
class MLPPINNConfig(PINNConfig, MLPConfig):
    pass


class MLPPINN(PINN, MLP):
    def __init__(self, c: MLPPINNConfig):
        MLP.__init__(self, MLPConfig.create(c))
        PINN.__init__(self, PINNConfig.create(c))

    def _make_hidden_layers(self, n_blks, hidden_size, dropout, activation):
        layers = []
        for _ in range(n_blks):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                activation(),
                nn.Dropout(dropout),  # Add dropout after ReLU
            ]
        if self.test_dropout is not None:
            layers[2] = nn.Dropout(self.test_dropout)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x
