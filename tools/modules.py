import zenkai
import torch
from torch import nn
import typing


class Stochastic(nn.Module):
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return (torch.rand_like(x) <= x).type_as(x)


class Sign(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x.sign()


class Clamp(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x.clamp(-1.0, 1.0)


class NullModule(nn.Module):

    def forward(self, *x) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]:

        if len(x) == 1:
            return x
        return x


class Layer(nn.Module):

    def __init__(
        self, in_features: int, out_features: int, 
        in_activation: typing.Type[nn.Module]=None, 
        out_activation: typing.Type[nn.Module]=None,
        dropout_p: float=None,
        batch_norm: bool=False,
        x_lr: float=None,
        # lmode: zenkai.LMode=zenkai.LMode.Standard
    ):
        """initialize

        Args:
            in_features (int): The number of input features
            out_features (int): The number of output features
            lmode (zenkai.LMode, optional): The learning model to use. Defaults to zenkai.LMode.Default.
        """

        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.loss = nn.MSELoss(reduction='sum')
        self.in_activation = in_activation() if in_activation is not None else NullModule()
        self.activation = out_activation() if out_activation is not None else NullModule()
        self.norm = nn.BatchNorm1d(out_features) if batch_norm else NullModule()
        self.dropout = nn.Dropout(dropout_p) if dropout_p else NullModule()
        self.x_lr = x_lr

    def forward(self, x: torch.Tensor) -> typing.Union[typing.Any, None]:
        
        y = self.in_activation(x.f)
        y = self.dropout(y)
        y = self.linear(y)
        y = self.norm(y)
        return self.activation(y)


class LayerLearner(zenkai.LearningMachine):

    def __init__(
        self, in_features: int, out_features: int, 
        in_activation: typing.Type[nn.Module]=None, 
        out_activation: typing.Type[nn.Module]=None,
        dropout_p: float=None,
        batch_norm: bool=False,
        x_lr: float=None,
        lmode: zenkai.LMode=zenkai.LMode.Standard
    ):
        """initialize

        Args:
            in_features (int): The number of input features
            out_features (int): The number of output features
            lmode (zenkai.LMode, optional): The learning model to use. Defaults to zenkai.LMode.Default.
        """

        super().__init__(lmode)
        self.linear = nn.Linear(in_features, out_features)
        self.loss = nn.MSELoss(reduction='sum')
        self.in_activation = in_activation() if in_activation is not None else NullModule()
        self.activation = out_activation() if out_activation is not None else NullModule()
        self.norm = nn.BatchNorm1d(out_features) if batch_norm else NullModule()
        self.dropout = nn.Dropout(dropout_p) if dropout_p else NullModule()
        self.x_lr = x_lr

    def accumulate(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State, **kwargs):

        cost = self.loss(state._y.f, t.f)
        cost.backward()   
    
    def step_x(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State, **kwargs) -> zenkai.IO:
        return x.acc_grad(self.x_lr)

    def forward_nn(self, x: zenkai.IO, state: zenkai.State, **kwargs) -> typing.Union[typing.Any, None]:
        
        y = self.in_activation(x.f)
        y = self.dropout(y)
        y = self.linear(y)
        y = self.norm(y)
        return self.activation(y)


# class Layer(zenkai.GradLearner):
#     # # 3) Layer

#     # A simple layer that uses

#     # - dropout for denoising capabilities
#     # - linear transformation
#     # - normalization (if used)
#     # - activation (if used)

#     def __init__(
#         self, in_features: int, out_features: int,
#         use_norm: bool, act: typing.Optional[typing.Type[nn.Module]]=nn.LeakyReLU, dropout_p: float=None,
#         in_act: typing.Optional[typing.Type[nn.Module]]=None, 
#         x_lr: float=1., lr: float=1e-3
#     ):

#         super().__init__()
#         self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else lambda x: x
#         self.linear = nn.Linear(in_features, out_features)
#         self.norm = nn.BatchNorm1d(out_features) if use_norm else lambda x: x
#         self.act = act() if act is not None else lambda x: x
#         self.in_act = in_act() if in_act is not None else lambda x: x
#         self._optim = torch.optim.Adam(self, lr=lr)
#         self._x_lr = x_lr

#     @property
#     def p(self) -> float:
#         return self.dropout.p if self.dropout is not None else None

#     @p.setter
#     def p(self, p: typing.Optional[float]):
#         if p is None:
#             self.dropout = lambda x: x
#         else:
#             self.dropout = nn.Dropout(p)
#         return p

#     def accumulate(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State):
#         loss = self._learn_criterion(state._y.f, t.f)
#         loss.backward()
    
#     def step_x(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State) -> zenkai.IO:
#         return x.acc_grad(self._x_lr)    
    
#     def step(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State):
#         self._optim.step()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:

#         y = self.in_act(x)
#         y = self.dropout(y)
#         y = self.linear(y)
#         y = self.norm(y)
#         y = self.act(y)
#         return y
