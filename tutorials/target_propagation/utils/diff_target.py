import zenkai
from zenkai import State, IO, Idx
from torch import nn
from .utils import Layer
import torch
import typing

from .target import (
    AutoencoderLearner, 
    OPT_MODULE_TYPE, MODULE_TYPE
)


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
        self._optim = zenkai.OptimFactory('Adam', lr=1e-3).comp()
        self._optim.prep_theta()
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
        self._optim = zenkai.OptimFactory('Adam', lr=1e-3).comp()
        self._optim.prep_theta([self.layer4])
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

