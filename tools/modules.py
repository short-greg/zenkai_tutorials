import zenkai
import torch
from torch import nn
import typing


class Stochastic(nn.Module):
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return (torch.rand_like(x) <= x).type_as(x)


class Layer(zenkai.GradLearner):
    # # 3) Layer

    # A simple layer that uses

    # - dropout for denoising capabilities
    # - linear transformation
    # - normalization (if used)
    # - activation (if used)

    def __init__(
        self, in_features: int, out_features: int,
        use_norm: bool, act: typing.Optional[typing.Type[nn.Module]]=nn.LeakyReLU, dropout_p: float=None,
        in_act: typing.Optional[typing.Type[nn.Module]]=None, 
        x_lr: float=1., lr: float=1e-3
    ):

        super().__init__()
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else lambda x: x
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.BatchNorm1d(out_features) if use_norm else lambda x: x
        self.act = act() if act is not None else lambda x: x
        self.in_act = in_act() if in_act is not None else lambda x: x
        self._optim = torch.optim.Adam(self, lr=lr)

    @property
    def p(self) -> float:
        return self.dropout.p if self.dropout is not None else None

    @p.setter
    def p(self, p: typing.Optional[float]):
        if p is None:
            self.dropout = lambda x: x
        else:
            self.dropout = nn.Dropout(p)
        return p

    def accumulate(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State):
        loss = self._learn_criterion(state._y.f, t.f)
        loss.backward()
    
    def step_x(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State) -> zenkai.IO:
        return x.acc_grad(self._lr)    
    
    def step(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State):
        self._optim.step()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = self.in_act(x)
        y = self.dropout(y)
        y = self.linear(y)
        y = self.norm(y)
        y = self.act(y)
        return y