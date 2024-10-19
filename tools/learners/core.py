import zenkai
import torch.nn as nn
import typing
from ..modules import Layer, NullModule
import torch


class LayerLearner(zenkai.LearningMachine):

    def __init__(
        self, in_features: int, out_features: int, 
        in_activation: typing.Type[nn.Module]=None, 
        out_activation: typing.Type[nn.Module]=None,
        dropout_p: float=None,
        batch_norm: bool=False,
        x_lr: float=None,
        lr: float=1e-3,
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
        if lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            self.optim = None
        self.x_lr = x_lr

    def accumulate(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State, **kwargs):

        cost = self.loss(state._y.f, t.f)
        cost.backward()   
    
    def step_x(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State, **kwargs) -> zenkai.IO:
        return x.acc_grad(self.x_lr)
    
    def step(self, x, t, state, **kwargs):
        if self.optim is not None:
            self.optim.step()
            self.optim.zero_grad()

    def forward_nn(self, x: zenkai.IO, state: zenkai.State, **kwargs) -> typing.Union[typing.Any, None]:
        
        y = self.in_activation(x.f)
        y = self.dropout(y)
        y = self.linear(y)
        y = self.norm(y)
        return self.activation(y)

