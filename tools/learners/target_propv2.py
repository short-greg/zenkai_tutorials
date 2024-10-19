import zenkai
import zenkai
from zenkai import State, IO, iou
from torch import Tensor
import zenkai
import torch.nn as nn
import torch
import typing
from .. import utils
import math
from abc import abstractmethod
from ..modules import Layer, NullModule
from .core import LayerLearner
from itertools import chain
from functools import partial


OPT_MODULE_TYPE = typing.Optional[typing.Type[nn.Module]]
MODULE_TYPE = typing.Type[nn.Module]


class BaseTPStepX(zenkai.StepX):

    def __init__(
        self, 
        train_reconstruction: bool=True, 
        train_predictor: bool=True
    ):
        super().__init__()
        self.train_reconstruction = train_reconstruction
        self.train_predictor = train_predictor

    @abstractmethod
    def step_x(self, x, t, state, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def factory(cls, train_reconstruction: bool=True, train_predictor: bool=True) -> 'TPStepX':
        pass


class TPStepX(BaseTPStepX):

    def __init__(
        self, 
        feedforward: nn.Module,
        feedback: nn.Module,
        train_reconstruction: bool=True, 
        train_predictor: bool=True
    ):
        super().__init__(train_reconstruction, train_predictor)
        self.feedforward = feedforward
        self.feedback = feedback

    def step_x(self, x, t: zenkai.IO, state, y: zenkai.IO=None):
        
        return t.on(self.feedback)

    @classmethod
    def factory(cls, train_reconstruction: bool=True, train_predictor: bool=True) -> 'TPStepX':
        def _(feedforward: nn.Module, feedback: nn.Module):
            return TPStepX(
                feedforward, feedback, train_reconstruction, train_predictor
            )
        return _


class DiffTPStepX(BaseTPStepX):

    def __init__(
        self, 
        feedforward: nn.Module,
        feedback: nn.Module,
        lr: float=1.0,
        train_reconstruction: bool=True, 
        train_predictor: bool=True
    ):
        super().__init__(train_reconstruction, train_predictor)
        self.feedforward = feedforward
        self.feedback = feedback
        self.lr = lr

    def step_x(self, x: zenkai.IO, t: zenkai.IO, state, y: zenkai.IO=None):
        
        if y is None:
            y = x.on(self.feedforward)
        rec_t = t.on(self.feedback)
        rec_y = y.on(self.feedback)
        
        dx = iou(rec_t.f - rec_y.f)
        return x.acc_dx(
            dx, self.lr
        )
    
    @classmethod
    def factory(cls, train_reconstruction: bool=True, train_predictor: bool=True) -> 'TPStepX':
        def _(feedforward: nn.Module, feedback: nn.Module):
            return DiffTPStepX(
                feedforward, feedback, train_reconstruction, train_predictor
            )
        return _


class BaseTPStepTheta(zenkai.StepTheta):
    
    def __init__(
        self, 
        train_reconstruction: bool=True, 
        train_predictor: bool=True
    ):
        super().__init__()
        self.train_reconstruction = train_reconstruction
        self.train_predictor = train_predictor
    
    @abstractmethod
    def accumulate(self, x, t, state, **kwargs):
        pass
    
    @abstractmethod
    def step(self, x, t, state, **kwargs):
        pass


class TPStepTheta(BaseTPStepTheta):
    
    def __init__(
        self, 
        feedforward: nn.Module,
        feedback: nn.Module,
        criterion: zenkai.Criterion,
        reverse_criterion: zenkai.Criterion,
        lr: float=1e-3,
        train_reconstruction: bool=True, 
        train_predictor: bool=True,
        rec_weight: float=1.0,
        pred_weight: float=1.0
    ):
        super().__init__(train_reconstruction, train_predictor)
        self.feedforward = feedforward
        self.feedback = feedback
        self._criterion = criterion
        self._reverse_criterion = reverse_criterion
        self.optim = torch.optim.Adam(
            chain(self.feedforward.parameters(), self.feedback.parameters()), lr=lr
        )
        self.assessment = None
        self.r_assessment = None
        self.rec_weight = rec_weight
        self.pred_weight = pred_weight
    
    def accumulate(self, x, t, state, **kwargs):        
        
        if self.train_reconstruction:
            z = utils.AmplifyGrad.apply(state._y.f, self.rec_weight)
            z = self.feedback(z)
            z_loss = self._reverse_criterion.assess(x, iou(z))
        else:
            z_loss = 0.
        if self.train_predictor:
            y = utils.AmplifyGrad.apply(state._y.f, self.pred_weight)
            t_loss = self._criterion.assess(iou(y), t)
        else:
            t_loss = 0.0
    
        p_loss = 0

        if not (isinstance(t_loss, float) and isinstance(z_loss, float)):
            (t_loss + z_loss + p_loss).backward()

        # self.i += 1
        self.assessment = z_loss
        self.r_assessment = t_loss
    
    def step(self, x, t, state, **kwargs):
        
        self.optim.step()
        self.optim.zero_grad()

    @classmethod
    def factory(cls, 
        criterion: zenkai.Criterion,
        reverse_criterion: zenkai.Criterion,
        lr: float=1e-3,
        train_reconstruction: bool=True, 
        train_predictor: bool=True,
        rec_weight: float=1.0,
        pred_weight: float=1.0):
        
        def _(feedforward: nn.Module, feedback: nn.Module):
            return cls(
                feedforward, feedback,
                criterion, reverse_criterion, lr, 
                train_reconstruction, train_predictor,
                rec_weight, pred_weight
            )
        return _


class TPAltStepTheta(BaseTPStepTheta):
    
    def __init__(
        self, 
        feedforward: nn.Module,
        feedback: nn.Module,
        criterion: zenkai.Criterion,
        reverse_criterion: zenkai.Criterion,
        lr: float=1e-3,
        train_reconstruction: bool=True, 
        train_predictor: bool=True,
        rec_weight: float=1.0,
        pred_weight: float=1.0
    ):
        super().__init__(train_reconstruction, train_predictor)
        self.feedforward = feedforward
        self.feedback = feedback
        self.criterion = criterion
        self.reverse_criterion = reverse_criterion
        self.pred_optim = torch.optim.Adam(
            self.feedforward, lr=lr
        )
        self.rec_optim = torch.optim.Adam(
            chain(self.feedforward.parameters(), self.feedback.parameters()), lr=lr
        )
        self.assessment = None
        self.r_assessment = None
        self.rec_weight = rec_weight
        self.pred_weigth = pred_weight
    
    def accumulate(self, x, t, state, **kwargs):        
        
        if self.train_reconstruction:
            z = utils.AmplifyGrad.apply(state._y.f, self.rec_weight)
            z = self.feedback(z)
            z_loss = self._reverse_criterion.assess(x, iou(z))
            z_loss.backward()
            self.rec_optim.step()
            self(x)
    
        if self.train_predictor:
            y = utils.AmplifyGrad.apply(state._y.f, self.pred_weight)
            t_loss = self.criterion(
                iou(y), t
            )
            t_loss.backward()
        self.assessment = t_loss
        self.r_assessment = z_loss
    
    def step(self, x, t, state, **kwargs):
        
        if self.train_predictor:
            self.pred_optim.step()
            self.pred_optim.zero_grad()

    @classmethod
    def factory(cls, 
        criterion: zenkai.Criterion,
        reverse_criterion: zenkai.Criterion,
        lr: float=1e-3,
        train_reconstruction: bool=True, 
        train_predictor: bool=True,
        rec_weight: float=1.0,
        pred_weight: float=1.0):
        
        def _(feedforward: nn.Module, feedback: nn.Module):
            return cls(
                feedforward, feedback,
                criterion, reverse_criterion, lr, 
                train_reconstruction, train_predictor,
                rec_weight, pred_weight
            )
        return _


STEPX_FACTORY = typing.Callable[[nn.Module, nn.Module], BaseTPStepX]
STEP_THETA_FACTORY = typing.Callable[[nn.Module, nn.Module], BaseTPStepTheta]


class LinearTPLearner(zenkai.LearningMachine):

    def __init__(
        self, in_features: int, out_features: int,  
        step_theta: STEP_THETA_FACTORY,
        step_x: STEPX_FACTORY,
        dropout_p: float=None, 
        forward_act: OPT_MODULE_TYPE=nn.LeakyReLU,
        reverse_act: OPT_MODULE_TYPE=nn.LeakyReLU,
        forward_in_act: typing.Type[nn.Module]=None,
        forward_norm: bool=True,
        reverse_norm: bool=True,
        targetf: OPT_MODULE_TYPE=None
    ):
        super().__init__()
        self.feedforward = Layer(
            in_features, out_features, 
            forward_in_act, forward_act, dropout_p, forward_norm
        )
        self.targetf = targetf or NullModule()
        self.feedback = Layer(
            out_features, in_features, None, 
            reverse_act, None, reverse_norm
        )
        self._step_theta = step_theta(self.feedforward, self.feedback)
        self._step_x = step_x(self.feedforward, self.feedback)
    
    def step(self, x: zenkai.IO, t: zenkai.IO, state, **kwargs):
        t = t.on(self.targetf)
        self._step_theta.step(x, t, state, **kwargs)
    
    def accumulate(self, x, t, state, **kwargs):
        t = t.on(self.targetf)
        self._step_theta.accumulate(x, t, state, **kwargs)

    def step_x(self, x, t, state, **kwargs):
        t = t.on(self.targetf)
        return self._step_x.step_x(x, t, state, **kwargs)
    
    def train_mode(self, train_reconstruction: bool=True, train_predictor: bool=True):
        """

        Args:
            train_reconstruction (bool, optional): . Defaults to True.
            train_predictor (bool, optional): . Defaults to True.
        """
        self.train_reconstruction = train_reconstruction
        self.train_predictor = train_predictor

    def forward_nn(self, x: zenkai.IO, state: State):

        x = x.clone()
        return self.feedforward(x.f)
    
    @property
    def assessment(self):
        return self._step_theta.assessment
    
    @property
    def r_assessment(self):
        return self._step_theta.r_assessment
    

class DeepTPLearner(zenkai.LearningMachine):

    def accumulate(self, x, t, state, **kwargs):
        
        ((state._y.f - t.f).pow(2).sum() * 0.5).backward()

    def step(self, x, t, state, **kwargs):
        pass

    def step_x(self, x, t, state, **kwargs):
        return x.acc_grad()


class DeepLinearTPLearner(DeepTPLearner):

    def __init__(self, in_features: int, h1_features: int,
      h2_features: int, h3_features: int, out_features: int,
      step_theta: TPStepTheta,
      step_x: TPStepX,
      out_x_lr: float=1.0,
      act: OPT_MODULE_TYPE=nn.LeakyReLU,
      reverse_act: OPT_MODULE_TYPE=nn.LeakyReLU,
      forward_in_act: OPT_MODULE_TYPE=None,
      dropout_p: float=0.1
    ):
        super().__init__()
        self.layer1 = LinearTPLearner(
            in_features, h1_features, step_theta, step_x,
            dropout_p, act, nn.Tanh, reverse_norm=False,
            forward_norm=True
        )
        self.layer2 = LinearTPLearner(
            h1_features, h2_features, step_theta, step_x,
            dropout_p, act, reverse_act, forward_in_act, reverse_norm=True,
            forward_norm=True
        )
        self.layer3 = LinearTPLearner(
            h2_features, h3_features, step_theta, step_x,
            dropout_p, act, reverse_act, forward_in_act, reverse_norm=True,
            forward_norm=True
        )
        self.layer4 = LayerLearner(
            h3_features, out_features, None,
            act, batch_norm=False, x_lr=out_x_lr
        )
    
    @property
    def assessments(self):
        return [layer.assessment for layer in [self.layer1, self.layer2, self.layer3]]

    @property
    def r_assessments(self):
        return [layer.r_assessment for layer in [self.layer1, self.layer2, self.layer3]]

    def forward_nn(self, x, state, **kwargs):
        
        y = self.layer1(x.f)
        y = self.layer2(y)
        y = self.layer3(y)
        return self.layer4(y)