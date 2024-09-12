import zenkai
import zenkai
from zenkai import State, IO, Idx, iou
from torch import Tensor
import zenkai
# from ..modules import Layer
import torch.nn as nn
import typing

import zenkai
from zenkai import State, IO, Idx
from torch import nn
# from ..modules import Layer
import torch
import typing


OPT_MODULE_TYPE = typing.Optional[typing.Type[nn.Module]]
MODULE_TYPE = typing.Type[nn.Module]


class NullModule(nn.Module):

    def forward(self, *x):

        if len(x) > 1:
            return x
        return x[0]


class Layer(zenkai.LearningMachine):

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



class AutoencoderLearner(zenkai.GradLearner):
    """Component for building a TargetPropagationLearner
    """

    def __init__(
        self, in_features: int, out_features: int, 
        rec_weight: float=1.0, dropout_p: float=None, 
        forward_act: OPT_MODULE_TYPE=nn.LeakyReLU,
        reverse_act: OPT_MODULE_TYPE=nn.LeakyReLU,
        rec_loss: MODULE_TYPE=nn.MSELoss,
        forward_in_act: typing.Type[nn.Module]=None,
        forward_norm: bool=True,
        reverse_norm: bool=True,
        targetf: OPT_MODULE_TYPE=None
    ):
        """Create an AutoencoderLearner with specificed input and output Features

        Args:
            in_features (int): number of input features
            out_features (int): number of output features
            rec_weight (float, optional): the weight on the reconstruction. Defaults to 1.0.
            dropout_p (float, optional): The amount of dropout to use. Defaults to None.
        """
        super().__init__()
        self._criterion = zenkai.NNLoss('MSELoss', reduction='sum')
        self._reverse_criterion = zenkai.NNLoss(rec_loss, reduction='sum')
        self.forward_on = True
        self.reverse_on = True
        self.feedforward = Layer(
            in_features, out_features, 
            forward_in_act, forward_act, dropout_p, forward_norm
        )
        self.targetf = targetf if targetf is not None else lambda x: x
        self.rec_weight = rec_weight

        self.feedback = Layer(
            out_features, in_features, None, reverse_act, None, reverse_norm
        )
        self.assessment = None
        self.r_assessment = None
        self._optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.i = 0

    def forward_nn(self, x: IO, state: State, batch_idx: Idx = None) -> Tensor:
        """Obtain the output of the learner

        Args:
            x (IO): The input
            state (State): The learning state
            batch_idx (Idx, optional): the index to use. Defaults to None.

        Returns:
            Tensor: The output of the function
        """
        return self.feedforward(x.f)

    def accumulate(self, x: IO, t: IO, state: State):
        """accumulate the gradients for the feedforward and feedback models

        Args:
            x (IO): the input
            t (IO): the output
            state (State): the learning state
        """
        z = self.feedback(state._y.f)
        z_loss = self._reverse_criterion.assess(x, iou(z))
        # print(state._y.f[0, :10], t.f[0, :10])
        t_loss = self._criterion.assess(state._y, t)
        (t_loss + self.rec_weight * z_loss).backward()
        self.i += 1
        self.assessment = z_loss
        self.r_assessment = t_loss

    def step(self, x: IO, t: IO, state: State):
        
        self._optim.step()
        self._optim.zero_grad()

    def step_x(self, x: IO, t: IO, state: State, batch_idx: Idx = None) -> IO:
        """Propagate the target by passing it throug 

        Args:
            x (IO): the input
            t (IO): the target
            state (State): the learning state
            batch_idx (Idx, optional): the index on the batch. Defaults to None.

        Returns:
            IO: 
        """
        return iou(self.feedback(self.targetf(t.f)))


class TargetPropLearner(zenkai.GradLearner):

    def __init__(
        self, in_features: int, h1_features: int,
        h2_features: int, h3_features: int, out_features: int,
        act: OPT_MODULE_TYPE=nn.LeakyReLU,
        reverse_act: OPT_MODULE_TYPE=nn.LeakyReLU,
        in_act: OPT_MODULE_TYPE=None,
        rec_loss: MODULE_TYPE=nn.MSELoss,
        targetf: OPT_MODULE_TYPE=None,
        out_x_lr: float=None
    ):
        """

        Args:
            in_features (int): the input features
            h1_features (int): number of features for layer 1
            h2_features (int): number of features for layer 2
            h3_features (int): number of features for layer 3
            out_features (int): the number of output features
        """
        super().__init__()
        self._criterion = zenkai.NNLoss('MSELoss', reduction='mean')
        self._learn_criterion = zenkai.NNLoss('MSELoss', reduction='sum', weight=0.5)
        self.forward_on = True
        self.reverse_on = True
        self.layer1 = AutoencoderLearner(
            in_features, h1_features, 1.0, 0.5, forward_act=act, reverse_act=nn.Tanh,
            rec_loss=nn.MSELoss,forward_norm=True, reverse_norm=False,
            targetf=targetf
        )
        self.layer2 = AutoencoderLearner(
            h1_features, h2_features, 1.0, dropout_p=0.25, forward_act=act, reverse_act=reverse_act,
            forward_in_act=in_act,
            rec_loss=rec_loss, forward_norm=True, reverse_norm=True,
            targetf=targetf
        )
        self.layer3 = AutoencoderLearner(
            h2_features, h3_features, 1.0, dropout_p=0.25, forward_act=act, reverse_act=reverse_act,
            forward_in_act=in_act,
            rec_loss=rec_loss, forward_norm=True, reverse_norm=True,
            targetf=targetf
        )
        self.layer4 = Layer(
            h3_features, out_features, None, None, None, False, x_lr=out_x_lr
        )
        self._optim = torch.optim.Adam(self.layer4.parameters(), lr=1e-3)
        self.assessments = []
        self.r_assessments = []

    def step(self, x: IO, t: IO, state: State):
        
        self._optim.step()
        self._optim.zero_grad()

    def accumulate(self, x: IO, state: State, batch_idx: Idx=None):
        super().accumulate(x, state, batch_idx)
        self.assessments = [layer.assessment for layer in [self.layer1, self.layer2, self.layer3]]
        self.r_assessments = [layer.r_assessment for layer in [self.layer1, self.layer2, self.layer3]]

    def forward_nn(self, x: IO, state: State, batch_idx: Idx = None) -> Tensor:

        y = self.layer1(x.f)
        y = self.layer2(y)
        y = self.layer3(y)
        return self.layer4(y)


class BaselineLearner1(zenkai.GradLearner):
    """Learner to use for comparison
    """

    def __init__(
        self, in_features: int, h1_features: int,
        h2_features: int, h3_features: int, out_features: int
    ):
        super().__init__()
        self._criterion = zenkai.NNLoss('MSELoss', reduction='mean')
        self._learn_criterion = zenkai.NNLoss('MSELoss', reduction='sum', weight=0.5)
        self.forward_on = True
        self.reverse_on = True

        self.layer1 = Layer(
            in_features, h1_features, None, nn.ReLU, 0.5, True
        )
        self.layer2 = Layer(
            h1_features, h2_features, None, nn.ReLU, 0.5, True
        )
        self.layer3 = Layer(
            h2_features, h3_features, None, nn.ReLU, 0.25, True
        )
        self.layer4 = Layer(
            h3_features, out_features, None, None, 0.15, False
        )
        self._optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.assessments = []
        self.r_assessments = []

    def step(self, x: IO, t: IO, state: State):
        
        self._optim.step()
        self._optim.zero_grad()

    def forward_nn(self, x: IO, state: State, batch_idx: Idx = None) -> Tensor:

        y = self.layer1(x.f)
        y = self.layer2(y)
        y = self.layer3(y)
        return self.layer4(y)


class DiffAutoencoderLearner(AutoencoderLearner):
    
    def __init__(
        self, in_features: int, out_features: int, rec_weight: float = 1, 
        dropout_p: float = None, x_lr: float=1.0, 
        forward_act: typing.Type[nn.Module]=nn.LeakyReLU,
        reverse_act: typing.Type[nn.Module]=nn.LeakyReLU,
        forward_in_act: typing.Type[nn.Module]=None,
        forward_norm: bool=True,
        reverse_norm: bool=True,
        rec_loss: typing.Type[nn.Module]=nn.MSELoss,
        targetf: OPT_MODULE_TYPE=None
    ):
        super().__init__(
            in_features, out_features, rec_weight, 
            dropout_p, forward_act=forward_act, reverse_act=reverse_act, rec_loss=rec_loss,
            forward_norm=forward_norm, reverse_norm=reverse_norm,
            forward_in_act=forward_in_act,
            targetf=targetf
          )
        self.x_lr = x_lr

    def step(self, x: IO, t: IO, state: State):
        
        self._optim.step()
        self._optim.zero_grad()

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        """Propagate the target and the output back and calculate the difference

        Args:
            x (IO): the input
            t (IO): the target
            state (State): the learning state

        Returns:
            IO: the target for the incoming layer
        """
        yx = self.feedback(state._y.f)
        tx = self.feedback(self.targetf(t.f))

        return x.acc_dx([yx - tx], self.x_lr).detach()


class DiffTargetPropLearner(zenkai.GradLearner):

    def __init__(
      self, in_features: int, h1_features: int,
      h2_features: int, h3_features: int, out_features: int,
      x_lr: float=1.0,
      act: OPT_MODULE_TYPE=nn.LeakyReLU,
      reverse_act: OPT_MODULE_TYPE=nn.LeakyReLU,
      forward_in_act: OPT_MODULE_TYPE=None,
      rec_loss: MODULE_TYPE=nn.MSELoss
    ):
        # Same as TargetPropLearner
        # but uses the DiffAutoencoderLearner
        super().__init__()
        self._criterion = zenkai.NNLoss('MSELoss', reduction='mean')
        self._learn_criterion = zenkai.NNLoss('MSELoss', reduction='sum', weight=0.5)
        self.forward_on = True
        self.reverse_on = True

        self.layer1 = DiffAutoencoderLearner(
          in_features, h1_features, 1.0, 0.5, x_lr=x_lr, 
          forward_act=act, reverse_act=nn.Tanh, rec_loss=nn.MSELoss,
          forward_norm=True, reverse_norm=False,
          forward_in_act=None
        )
        self.layer2 = DiffAutoencoderLearner(
          h1_features, h2_features, 1.0, dropout_p=0.25, 
          x_lr=x_lr, forward_act=act, reverse_act=reverse_act, rec_loss=rec_loss,
          forward_norm=True, reverse_norm=True,
          forward_in_act=forward_in_act
        )
        self.layer3 = DiffAutoencoderLearner(
          h2_features, h3_features, 1.0, dropout_p=0.25,
          x_lr=x_lr, forward_act=act, reverse_act=reverse_act, rec_loss=rec_loss,
          forward_norm=True, reverse_norm=True,
          forward_in_act=forward_in_act
        )
        self.layer4 = Layer(
          h3_features, out_features,
          False, None, None
        )
        self._optim = torch.optim.Adam(self.layer4.parameters(), lr=1e-3)
        self.assessments = []
        self.r_assessments = []
    
    def accumulate(self, x: IO, state: State, batch_idx: Idx=None):
        super().accumulate(x, state, batch_idx)
        self.assessments = [layer.assessment for layer in [self.layer1, self.layer2, self.layer3]]
        self.r_assessments = [layer.r_assessment for layer in [self.layer1, self.layer2, self.layer3]]

    def forward_nn(self, x: IO, state: State, batch_idx: Idx = None) -> torch.Tensor:

        y = self.layer1(x.f)
        y = self.layer2(y)
        y = self.layer3(y)
        return self.layer4(y)


class DiffTargetPropLearner2(zenkai.GradLearner):

    def __init__(
      self, in_features: int, h1_features: int,
      h2_features: int, h3_features: int, out_features: int,
      x_lr: float=1.0, rec_weight: float=1.0, out_rec_weight: float=1.0,
      act: OPT_MODULE_TYPE=nn.LeakyReLU,
      reverse_act: OPT_MODULE_TYPE=nn.LeakyReLU,
      forward_in_act: OPT_MODULE_TYPE=None,
      rec_loss: MODULE_TYPE=nn.MSELoss,
    ):
        # Same as TargetPropLearner
        # but uses the DiffAutoencoderLearner
        super().__init__()
        self._criterion = zenkai.NNLoss('MSELoss', reduction='mean')
        self._learn_criterion = zenkai.NNLoss('MSELoss', reduction='sum', weight=0.5)
        self.forward_on = True
        self.reverse_on = True

        self.layer1 = DiffAutoencoderLearner(
          in_features, h1_features, rec_weight, 0.5, x_lr=x_lr, 
          forward_act=act, reverse_act=nn.Sigmoid, rec_loss=nn.MSELoss,
          forward_norm=True, reverse_norm=False,
          forward_in_act=None
        )
        self.layer2 = DiffAutoencoderLearner(
          h1_features, h2_features, rec_weight, dropout_p=0.25, 
          x_lr=x_lr, forward_act=act, reverse_act=reverse_act, rec_loss=rec_loss,
          forward_norm=True, reverse_norm=True,
          forward_in_act=forward_in_act
        )
        self.layer3 = DiffAutoencoderLearner(
          h2_features, h3_features, rec_weight, dropout_p=0.25,
          x_lr=x_lr, forward_act=act, reverse_act=reverse_act, rec_loss=rec_loss,
          forward_norm=True, reverse_norm=True,
          forward_in_act=forward_in_act
        )
        self.layer4 = DiffAutoencoderLearner(
          h2_features, h3_features, out_rec_weight, dropout_p=0.0,
          x_lr=x_lr, forward_act=None, reverse_act=reverse_act, 
          rec_loss=rec_loss, 
          forward_norm=True, reverse_norm=True,
          forward_in_act=forward_in_act
        )
        self._optim = torch.optim.Adam(self.layer4.parameters(), lr=1e-3)
        self.assessments = []
        self.r_assessments = []
    
    def accumulate(self, x: IO, state: State, batch_idx: Idx=None):
        super().accumulate(x, state, batch_idx)
        self.assessments = [layer.assessment for layer in [self.layer1, self.layer2, self.layer3]]
        self.r_assessments = [layer.r_assessment for layer in [self.layer1, self.layer2, self.layer3]]

    def forward_nn(self, x: IO, state: State, batch_idx: Idx = None) -> torch.Tensor:

        y = self.layer1(x.f)
        y = self.layer2(y)
        y = self.layer3(y)
        return self.layer4(y)
