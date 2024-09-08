from typing import Any, Tuple
from zenkai import LearningMachine
import zenkai
from torch import nn
import torch


class Layer(LearningMachine):

    def __init__(
        self, in_features: int, out_features: int, 
        lmode: zenkai.LMode=zenkai.LMode.Default
    ):

        super().__init__(lmode)
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.LeakyReLU()
        self.loss = nn.MSELoss(reduction='sum')

    def accumulate(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State, **kwargs):

        cost = self.loss(state._y.f, t.f)
        cost.backward()   
    
    def step_x(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State, **kwargs) -> zenkai.IO:
        return x.acc_grad()

    def forward_nn(self, x: zenkai.IO, state: zenkai.State, **kwargs) -> Tuple | Any:
        
        y = self.linear(x.f)
        return self.activation(y)


class STELayer(LearningMachine):

    def __init__(self, in_features: int, out_features: int, lmode: zenkai.LMode=zenkai.LMode.Default):

        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.loss = nn.MSELoss(reduction='none')

    def accumulate(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State, **kwargs):

        # get the difference
        dx = (t.f - state._y.f)
        # propagate the difference "straight throug"
        t = state._yl + dx
        oob = (state._yl > 1.0) | (state._yl < -1.0)
        # calculate SSE for all values not
        # out of bounds
        cost = (
            (self.loss(state._yl, t.detach()) * oob)
        ).sum() * 0.5
        cost.backward()   
    
    def step_x(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State, **kwargs) -> zenkai.IO:
        return x.acc_grad()

    def forward_nn(self, x: zenkai.IO, state: zenkai.State, **kwargs) -> Tuple | Any:
        
        y = state._yl = self.linear(x.f)
        return torch.sign(y)


class Network(zenkai.GradLearner):

    def __init__(
        self, in_features: int, h1: int, h2: int, 
        out_features: int
    ):
        super().__init__()
        self.layer1 = Layer(in_features, h1)
        self.layer2 = Layer(h1, h2)
        self.layer3 = nn.Linear(h2, out_features)
        self.assessments = []
        self._optim = torch.optim.Adam(
            params=self.parameters()
        )
    
    def step(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State):
        self._optim.step()
        self._optim.zero_grad()
    
    def forward_nn(self, x: zenkai.IO, state: zenkai.State, **kwargs) -> torch.Tensor:

        y = self.layer1(x.f)
        y = self.layer2(y)
        return self.layer3(y)


class STENetwork(zenkai.GradLearner):

    def __init__(
        self, in_features: int, h1: int, h2: int, 
        out_features: int
    ):
        super().__init__()
        self.layer1 = STELayer(in_features, h1)
        self.layer2 = STELayer(h1, h2)
        self.layer3 = nn.Linear(h2, out_features)
        self.assessments = []
        self._optim = torch.optim.Adam(
            params=self.parameters()
        )
    
    def step(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State):
        self._optim.step()
        self._optim.zero_grad()

    def forward_nn(self, x: zenkai.IO, state: zenkai.State, **kwargs) -> torch.Tensor:

        y = self.layer1(x.f)
        y = self.layer2(y)
        return self.layer3(y)
