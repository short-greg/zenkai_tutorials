from typing import Any, Tuple
from zenkai import LearningMachine
import zenkai
from torch import nn
import torch
from torch.utils import data as torch_data
from zenkai import iou


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


class LoopLayer(LearningMachine):

    def __init__(
        self, in_features: int, out_features: int, 
        theta_loops: int=1, theta_batch_size: int=None,
        x_loops: int=1, lr: float=1e-3, activation: bool=False,
        use_batch_norm: bool=False
    ):
        """

        Args:
            in_features (int): 
            out_features (int): 
            theta_loops (int, optional): . Defaults to 1.
            theta_batch_size (int, optional): . Defaults to None.
            x_loops (int, optional): . Defaults to 1.
            lr (float, optional): . Defaults to 1e-3.
        """
        super().__init__(lmode=zenkai.LMode.StepPriority)
        self.linear = nn.Linear(in_features, out_features)
        self.loss = nn.MSELoss(reduction='none')
        self.activation = nn.LeakyReLU() if activation else lambda x: x
        self.learn_criterion = zenkai.NNLoss('MSELoss', reduction='sum', weight=0.5)
        self.theta_optim = torch.optim.Adam(self.linear.parameters(), lr=lr)
        self.theta_loops = theta_loops
        self.theta_batch_size = theta_batch_size
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.x_loops = x_loops
        self.lr = lr
        self.use_batch_norm = use_batch_norm
        self._optim = torch.optim.Adam(self.parameters(), lr=lr)

    def accumulate(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State, **kwargs):
        # Accumualte does nothing
        pass

    def step(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State, **kwargs):
        
        for _ in range(self.theta_loops):

            dataset = torch_data.TensorDataset(x.f, t.f)
            sub_state = state.sub('sub')
            for x_i, t_i in torch_data.DataLoader(
                dataset, batch_size=self.theta_batch_size, 
            ):
                y_i = self.forward_nn(iou(x_i), sub_state)
                sub_state.clear()
                
                self.theta_optim.zero_grad()
                cost = self.learn_criterion.assess(iou(y_i), iou(t_i))
                cost.backward()
                self.theta_optim.step()

    def step_x(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State, **kwargs) -> zenkai.IO:
        x_optim = torch.optim.Adam([x.f], lr=self.lr)
        sub_state = state.sub('sub')
        for _ in range(self.x_loops):
            y = self.forward_nn(x, sub_state)
            sub_state.clear()
            x_optim.zero_grad()
            cost = self.learn_criterion.assess(iou(y), t)
            cost.backward()
            x_optim.step()
        return x.acc_grad()

    def forward_nn(self, x: zenkai.IO, state: zenkai.State, **kwargs) -> Tuple | Any:
        
        y = state._yl = self.linear(x.f)
        if self.use_batch_norm:
            y = self.batch_norm(y)
        return self.activation(y)


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


class LoopNetwork(zenkai.GradLearner):

    def __init__(
        self, in_features: int, h1: int, h2: int, 
        out_features: int
    ):
        super().__init__()
        n_loops = 40
        self.layer1 = LoopLayer(in_features, h1, 1, 128, n_loops, lr=1e-3)
        self.layer2 = LoopLayer(h1, h2, 1, 128, n_loops, lr=1e-3)
        self.layer3 = nn.Linear(h2, out_features)
        self._optim = torch.optim.Adam(self.layer3.parameters(), lr=1e-3)
        self.assessments = []

    def step(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State):
        self._optim.step()
        self._optim.zero_grad()

    def forward_nn(self, x: zenkai.IO, state: zenkai.State, **kwargs) -> torch.Tensor:

        y = self.layer1(x.f)
        y = self.layer2(y)
        return self.layer3(y)
