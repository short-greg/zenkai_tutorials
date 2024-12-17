import zenkai
import zenkai
from zenkai import State, iou
import zenkai
import torch.nn as nn
import torch
import typing
from .. import utils
from abc import abstractmethod
from ..modules import Layer, NullModule
from .core import LayerLearner
from itertools import chain
from torch import Tensor
from zenkai import IO

from functools import partial
from ..modules import Sign, Clamp, Sampler, Binary


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

    def assessments(self) -> typing.Dict:
        return {}

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
        
        return t.on(self.feedback).detach()

    @classmethod
    def factory(cls, train_reconstruction: bool=True, train_predictor: bool=True) -> 'TPStepX':
        def _(feedforward: nn.Module, feedback: nn.Module):
            return TPStepX(
                feedforward, feedback, train_reconstruction, train_predictor
            )
        return _
    
    def assessments(self) -> typing.Dict:
        return {}


class BaselineLearner1(zenkai.GradLearner):
    """Learner to use for comparison
    """
    def __init__(
        self, in_features: int, h1_features: int,
        h2_features: int, h3_features: typing.Optional[int], out_features: int,
        activation: typing.Callable[[], nn.Module]=nn.LeakyReLU,
        dropout_p: float=0.5,
        lr: float=1e-3
    ):
        """

        Args:
            in_features (int): The number of input features
            h1_features (int): The hidden 1 features
            h2_features (int): The hidden 2 features
            h3_features (int): The hidden 3 features
            out_features (int): The out features
            activation (typing.Callable[[], nn.Module], optional): The activation. Defaults to nn.LeakyReLU.
            dropout_p (float, optional): The dropout value to use. Defaults to 0.5.
            lr (float, optional): The learning rate for the learner. Defaults to 1e-3.
        """
        super().__init__()
        print(in_features)
        self._criterion = zenkai.NNLoss('MSELoss', reduction='mean')
        self._learn_criterion = zenkai.NNLoss('MSELoss', reduction='sum', weight=0.5)
        self.forward_on = True

        self.layer1 = LayerLearner(
            in_features, h1_features, None, 
            activation, dropout_p, True
        )
        self.layer2 = LayerLearner(
            h1_features, h2_features, None, 
            activation, dropout_p * 0.5, True
        )
        if h3_features is not None:
            self.layer3 = LayerLearner(
                h2_features, h3_features, None, 
                activation, dropout_p * 0.5, True
            )
        else:
            self.layer3 = NullModule()
            h3_features = h2_features
        self.layer4 = LayerLearner(
            h3_features, out_features,
            None, None, None, False
        )
        self._optim = torch.optim.Adam(
            self.parameters(), lr=lr
        )

    def step(self, x: IO, t: IO, state: State):
        """Update 

        Args:
            x (IO): The input
            t (IO): The target
            state (State): The learning state
        """
        self._optim.step()
        self._optim.zero_grad()

    def forward_nn(self, x: IO, state: State) -> Tensor:

        y = self.layer1(x.f)
        y = self.layer2(y)
        y = self.layer3(y)
        return self.layer4(y)

    def assessments(self) -> typing.Dict:

        result = {}
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            if hasattr(layer, 'assessments'):
                for k, v in layer.assessments().items():
                    result[f'{i}_{k}'] = v
        return result


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
        self.x_grad = None
        self.x_diff = None
        self.x_sim = None

    def step_x(self, x: zenkai.IO, t: zenkai.IO, state, y: zenkai.IO=None):
        
        x.freshen_()
        x.zero_grad()
        ((self.feedforward(x.f) - t.f).pow(2).sum() * 2).backward()

        if x.f.grad is not None:
            self.x_grad = x.f.grad.abs().mean()
        else:
            self.x_grad = None

        if y is None:
            y = x.on(self.feedforward)
        rec_t = t.on(self.feedback)
        rec_y = y.on(self.feedback)

        dx = iou(rec_t.f - rec_y.f)
        if self.x_grad is not None:
            self.x_sim = torch.cosine_similarity(x.f.grad, dx.f).mean()
        else:
            self.x_sim = None
        self.x_diff = dx.f.abs().mean()
        return x.acc_dx(
            dx, self.lr
        )
    
    def assessments(self) -> typing.Dict:
        res = {}
        if self.x_grad is not None:
            res['x_grad'] = self.x_grad.item()
        if self.x_diff is not None:
            res['x_diff'] = self.x_diff.item()
        if self.x_sim is not None:
            res['x_sim'] = self.x_sim.item()
        
        return res

    @classmethod
    def factory(cls, train_reconstruction: bool=True, train_predictor: bool=True, lr: float=1.0) -> 'TPStepX':
        def _(feedforward: nn.Module, feedback: nn.Module):
            return DiffTPStepX(
                feedforward, feedback, lr, train_reconstruction, train_predictor
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

    def assessments(self) -> typing.Dict:
        return {}


class NullTPStepTheta(BaseTPStepTheta):
    
    def __init__(
        self, feedforward, feedback
    ):
        super().__init__(False, False)
        self.feedforward = feedforward
        self.feedback = feedback
    
    def accumulate(self, x, t, state, **kwargs):

        self.assessment = None
        self.r_assessment = None
    
    def step(self, x, t, state, **kwargs):
        return x.detach()

    @classmethod
    def factory(cls):
        
        def _(feedforward: nn.Module, feedback: nn.Module):
            return cls(
                feedforward, feedback,
            )
        return _


class TPStepTheta(BaseTPStepTheta):
    
    def __init__(
        self, 
        feedforward: nn.Module,
        feedback: nn.Module,
        criterion: zenkai.Criterion,
        reverse_criterion: zenkai.Criterion,
        lr: float=1e-3,
        rec_lr: float=1e-3,
        train_reconstruction: bool=True, 
        train_predictor: bool=True,
        rec_weight: float=1.0,
        pred_weight: float=1.0,
        # gaussian_noise: float=None
    ):
        super().__init__(train_reconstruction, train_predictor)
        self.feedforward = feedforward
        self.feedback = feedback
        self._criterion = criterion
        self._reverse_criterion = reverse_criterion
        self.pred_optim = torch.optim.Adam(
            self.feedforward.parameters(), lr=lr
        )
        self.rec_optim = torch.optim.SGD(
            self.feedback.parameters(), lr=rec_lr, momentum=0.7
        )
        self.assessment = None
        self.r_assessment = None
        self.rec_weight = rec_weight
        self.pred_weight = pred_weight
        # self.gaussian_noise = gaussian_noise
    
    def accumulate(self, x, t, state, **kwargs):        
        
        # if self.gaussian_noise is not None:
        #     y = self.feedforward(x.f)
        # else:
        y = state._y.f
            
        if self.train_reconstruction:

            z = utils.AmplifyGrad.apply(
                y, self.rec_weight
            )
            z = self.feedback(z)
            z_loss = self._reverse_criterion.assess(x, iou(z))
        else:
            z_loss = None
        if self.train_predictor:
            y = utils.AmplifyGrad.apply(y, self.pred_weight)
            t_loss = self._criterion.assess(iou(y), t)
        else:
            t_loss = None
    
        p_loss = 0

        if not (isinstance(t_loss, float) and isinstance(z_loss, float)):
            # print(t_loss, z_loss)
            ((t_loss or 0.0) + (z_loss or 0.0) + p_loss).backward()

        # self.i += 1
        self.assessment = z_loss
        self.r_assessment = t_loss
    
    def assessments(self) -> typing.Dict:
        res = {}
        if self.assessment is not None:
            res['loss'] = self.assessment.item()
        if self.r_assessment is not None:
            res['r_loss'] = self.r_assessment.item()
        return res

    def step(self, x, t, state, **kwargs):
        
        self.rec_optim.step()
        self.rec_optim.zero_grad()
        self.pred_optim.step()
        self.pred_optim.zero_grad()

    @classmethod
    def factory(cls, 
        criterion: zenkai.Criterion,
        reverse_criterion: zenkai.Criterion,
        lr: float=1e-3,
        rec_lr: float=1e-3,
        train_reconstruction: bool=True, 
        train_predictor: bool=True,
        rec_weight: float=1.0,
        pred_weight: float=1.0):
        
        def _(feedforward: nn.Module, feedback: nn.Module):
            return cls(
                feedforward, feedback,
                criterion, reverse_criterion, lr, rec_lr,
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
        rec_lr: float=1e-3,
        train_reconstruction: bool=True, 
        train_predictor: bool=True,
        rec_weight: float=1.0,
        pred_weight: float=1.0,
    ):
        super().__init__(train_reconstruction, train_predictor)
        self.feedforward = feedforward
        self.feedback = feedback
        self.criterion = criterion
        self.reverse_criterion = reverse_criterion
        self.pred_optim = torch.optim.Adam(
            self.feedforward.parameters(), lr=lr, weight_decay=1e-6
        )
        self.rec_optim = torch.optim.SGD(
            chain(
                self.feedforward.parameters(), 
                self.feedback.parameters()
            ), 
            lr=rec_lr, momentum=0.75, weight_decay=1e-6
        )
        self.assessment = None
        self.r_assessment = None
        self.rec_weight = rec_weight
        self.pred_weight = pred_weight
    
    def accumulate(self, x, t, state, **kwargs):        

        t_loss = None
        z_loss = None
        y = state._y.f

        if self.train_reconstruction:
            
            z = utils.AmplifyGrad.apply(y, self.rec_weight)
            z = self.feedback(z)
            z_loss = self.reverse_criterion.assess(x, iou(z))
            self.rec_optim.zero_grad()
            z_loss.backward()
            self.rec_optim.step()
        else:
            z_loss = None
    
        if self.train_predictor:
            if self.train_reconstruction:
                y = self.feedforward(x.f)
            self.pred_optim.zero_grad()
            y = utils.AmplifyGrad.apply(y, self.pred_weight)
            t_loss = self.criterion(
                iou(y), t
            )
            t_loss.backward()
        else:
            t_loss = None
        self.assessment = t_loss
        self.r_assessment = z_loss
    
    def assessments(self) -> typing.Dict:
        res = {}
        if self.assessment is not None:
            res['loss'] = self.assessment.item()
        if self.r_assessment is not None:
            res['r_loss'] = self.r_assessment.item()
        return res

    def step(self, x, t, state, **kwargs):
        
        if self.train_predictor:
            self.pred_optim.step()
            self.pred_optim.zero_grad()

    @classmethod
    def factory(cls, 
        criterion: zenkai.Criterion,
        reverse_criterion: zenkai.Criterion,
        lr: float=1e-3,
        rec_lr: float=1e-3,
        train_reconstruction: bool=True, 
        train_predictor: bool=True,
        rec_weight: float=1.0,
        pred_weight: float=1.0):
        
        def _(feedforward: nn.Module, feedback: nn.Module):
            return cls(
                feedforward, feedback,
                criterion, reverse_criterion, lr, rec_lr,
                train_reconstruction, train_predictor,
                rec_weight, pred_weight
            )
        return _



class TPRegAltStepTheta(BaseTPStepTheta):
    
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
        pred_weight: float=1.0,
        reg: float=1e-3
        # gaussian_noise: float=None
    ):
        super().__init__(train_reconstruction, train_predictor)
        self.feedforward = feedforward
        self.feedback = feedback
        self.criterion = criterion
        self.reverse_criterion = reverse_criterion
        self.pred_optim = torch.optim.Adam(
            self.feedforward.parameters(), lr=lr
        )
        self.rec_optim = torch.optim.Adam(
            chain(self.feedforward.parameters(), self.feedback.parameters()), lr=lr
        )
        self.assessment = None
        self.r_assessment = None
        self.rec_weight = rec_weight
        self.pred_weight = pred_weight
        self.reg = reg
        # self.gaussian_noise = gaussian_noise
    
    def accumulate(self, x, t, state, **kwargs):        

        t_loss = None
        z_loss = None
        
        y = state._y.f

        if self.train_reconstruction:
            
            y = self.feedforward(x.f)
            y2 = self.feedforward(x.f)
            reg_loss = (y - y2).pow(2).mean() * self.reg
            z = utils.AmplifyGrad.apply(y, self.rec_weight)
            z = self.feedback(z)
            z_loss = self.reverse_criterion.assess(x, iou(z))
            (z_loss + reg_loss).backward()
            self.rec_optim.step()
        else:
            z_loss = None
    
        if self.train_predictor:
            if self.train_reconstruction:
                y = self.feedforward(x.f)
            else:
                y = state._y.f

            y = utils.AmplifyGrad.apply(y, self.pred_weight)
            t_loss = self.criterion(
                iou(y), t
            )
            t_loss.backward()
        else:
            t_loss = None
        self.assessment = t_loss
        self.r_assessment = z_loss
    
    def assessments(self) -> typing.Dict:
        res = {}
        if self.assessment is not None:
            res['loss'] = self.assessment.item()
        if self.r_assessment is not None:
            res['r_loss'] = self.r_assessment.item()
        return res

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
        pred_weight: float=1.0, reg: float=1e-3):
        
        def _(feedforward: nn.Module, feedback: nn.Module):
            return cls(
                feedforward, feedback,
                criterion, reverse_criterion, lr, 
                train_reconstruction, train_predictor,
                rec_weight, pred_weight, reg=reg
            )
        return _


class TPMultAltStepTheta(BaseTPStepTheta):
    
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
        pred_weight: float=1.0,
        k: int = 8
        # gaussian_noise: float=None
    ):
        super().__init__(train_reconstruction, train_predictor)
        self.feedforward = feedforward
        self.feedback = feedback
        self.criterion = criterion
        self.reverse_criterion = reverse_criterion
        self.pred_optim = torch.optim.Adam(
            self.feedforward.parameters(), lr=lr
        )
        self.rec_optim = torch.optim.Adam(
            chain(self.feedforward.parameters(), self.feedback.parameters()), lr=lr
        )
        self.assessment = None
        self.r_assessment = None
        self.rec_weight = rec_weight
        self.pred_weight = pred_weight
        self.k = k
        # self.gaussian_noise = gaussian_noise
    
    def accumulate(self, x, t, state, **kwargs):        

        t_loss = None
        z_loss = None
        
        y = state._y.f
        shape = x.f.shape

        viewed = x.f.view(1, *shape)
        viewed = viewed.repeat(self.k, *[1] * len(shape)).reshape(
            self.k * shape[0], *shape[1:]
        )

        if self.train_reconstruction:
            y = self.feedforward(viewed)
            z = utils.AmplifyGrad.apply(y, self.rec_weight)
            z = self.feedback(z)
            z = z.view(self.k, -1, *z.shape[1:])
            z_loss = self.reverse_criterion.assess(
                iou(x.f.view(1, *shape)), iou(z)
            )
            z_loss.backward()
            self.rec_optim.step()
        else:
            z_loss = None
    
        if self.train_predictor:
            #if self.train_reconstruction:
            y = self.feedforward(viewed)
            # else:
            #     y = viewed
            #     y = state._y.f

            t = iou(
                t.f.view(1, *t.f.shape)
            )
            y = utils.AmplifyGrad.apply(y, self.pred_weight)
            y = y.view(self.k, -1, *y.shape[1:])
            t_loss = self.criterion(
                iou(y), t
            )
            t_loss.backward()
        else:
            t_loss = None
        self.assessment = t_loss
        self.r_assessment = z_loss
    
    def assessments(self) -> typing.Dict:
        res = {}
        if self.assessment is not None:
            res['loss'] = self.assessment.item()
        if self.r_assessment is not None:
            res['r_loss'] = self.r_assessment.item()
        return res

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
        gaussian_noise: float=None,
        forward_act: OPT_MODULE_TYPE=nn.LeakyReLU,
        reverse_act: OPT_MODULE_TYPE=nn.LeakyReLU,
        forward_in_act: typing.Type[nn.Module]=None,
        forward_norm: bool=True,
        reverse_norm: bool=True,
        targetf: OPT_MODULE_TYPE=None,
        two_layer: bool=False,
        share_weights: bool=False
    ):
        super().__init__()
        if share_weights and two_layer:
            raise RuntimeError('Cannot have both share weights and two layer')
        self.feedforward = Layer(
            in_features, out_features,
            forward_in_act, forward_act, dropout_p, forward_norm,
        )
        self.targetf = targetf or NullModule()
        self.gaussian_noise = gaussian_noise

        if two_layer:
            self.feedback = nn.Sequential(Layer(
                out_features, out_features, None, 
                reverse_act, None, True
            ), Layer(
                out_features, in_features, None, 
                reverse_act, None, reverse_norm
            ))
        else:
            self.feedback = Layer(
                out_features, in_features, None, 
                reverse_act, None, reverse_norm
            )
            if share_weights:
                self.feedback.linear.weight = nn.parameter.Parameter(
                    self.feedforward.linear.weight.T
                )
        self._step_theta = step_theta(
            self.feedforward, 
            self.feedback
        )
        self._step_x = step_x(self.feedforward, self.feedback)
    
    def step(self, x: zenkai.IO, t: zenkai.IO, state, **kwargs):
        t = t.on(self.targetf)
        self._step_theta.step(state._x_e, t, state, **kwargs)
    
    def accumulate(self, x, t, state, **kwargs):
        t = t.on(self.targetf)
        self._step_theta.accumulate(state._x_e, t, state, **kwargs)

    def step_x(self, x, t, state, **kwargs):
        t = t.on(self.targetf)

        x_t = self._step_x.step_x(state._x_e, t, state, **kwargs)
        if state._noise is not None:
            return iou(x_t.f - state._noise)
        return x_t
    
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
        if self.gaussian_noise is not None:
            state._noise = torch.randn_like(x.f) * self.gaussian_noise
            state._x_e = iou(
                x.f + state._noise
            )
        else:
            state._noise = None
            state._x_e = x
        state._x_e.freshen_()
        return self.feedforward(state._x_e.f)
    
    def assessments(self) -> typing.Dict:
        return {
            **self._step_theta.assessments(),
            **self._step_x.assessments()
        }

    
class DeepTPLearner(zenkai.LearningMachine):

    def accumulate(self, x, t, state, **kwargs):
        
        ((state._y.f - t.f).pow(2).mean() * 0.5).backward()

    def step(self, x, t, state, **kwargs):
        pass

    def step_x(self, x, t, state, **kwargs):
        return x.acc_grad()

    def assessments(self) -> typing.Dict:
        return {}



class DeepLinearTPLearner(DeepTPLearner):

    def __init__(self, in_features: int, h1_features: int,
      h2_features: int, h3_features: typing.Optional[int], out_features: int,
      step_theta: TPStepTheta,
      step_x: TPStepX,
      out_x_lr: float=1.0,
      act: OPT_MODULE_TYPE=nn.LeakyReLU,
      reverse_act: OPT_MODULE_TYPE=nn.LeakyReLU,
      forward_in_act: OPT_MODULE_TYPE=None,
      use_norm: bool=True,
      two_layer: bool=False,
      lr: float=1e-3,
      dropout_p: float=0.1, 
      gaussian_noise: float=None,
      share_weights: bool=False,
      
    ):
        super().__init__()
        self.layer1 = LinearTPLearner(
            in_features, h1_features, step_theta, step_x,
            dropout_p, gaussian_noise, act, nn.Tanh,
            reverse_norm=False,
            forward_norm=use_norm,
            two_layer=two_layer,
            share_weights=share_weights,
        )
        self.layer2 = LinearTPLearner(
            h1_features, h2_features, 
            step_theta, step_x,
            dropout_p, gaussian_noise, act, 
            reverse_act,
            forward_in_act, 
            reverse_norm=use_norm,
            forward_norm=use_norm,
            two_layer=two_layer,
            share_weights=share_weights,
        )
        if h3_features is not None:
            self.layer3 = LinearTPLearner(
                h2_features, h3_features, step_theta, step_x,
                dropout_p, gaussian_noise,
                act, 
                reverse_act, forward_in_act, 
                reverse_norm=use_norm,
                forward_norm=use_norm,
                two_layer=two_layer,
                share_weights=share_weights,
            )
        else:
            self.layer3 = NullModule()
            h3_features = h2_features
        
        self.layer4 = LayerLearner(
            h3_features, out_features, None,
            None, dropout_p=None, batch_norm=False, x_lr=out_x_lr,
            lr=lr
        )

    def assessments(self) -> typing.Dict:

        res = {}
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            if hasattr(layer, 'assessments'):
                for k, v in layer.assessments().items():
                    res[f'{i}_{k}'] = v
        return res

    def step_thetas(self) -> typing.Iterator:
        for layer in [self.layer1, self.layer2, self.layer3]:
            if hasattr(layer, '_step_theta'):
                yield layer._step_theta

    def step_xs(self) -> typing.Iterator:
        for layer in [self.layer1, self.layer2, self.layer3]:
            if hasattr(layer, '_step_x'):
                yield layer._step_x

    # def x_grads(self):

    #     return [
    #         layer.x_grad for layer in 
    #         [self.layer1, self.layer2, self.layer3]
    #     ]

    # def x_diffs(self):

    #     return [
    #         layer.x_diff for layer in 
    #         [self.layer1, self.layer2, self.layer3]
    #     ]

    # @property
    # def r_assessments(self):
    #     return [
    #         layer.r_assessment for layer in 
    #         [self.layer1, self.layer2, self.layer3]]

    def forward_nn(self, x, state, **kwargs):
        
        y = self.layer1(x.f)
        y = self.layer2(y)
        y = self.layer3(y)
        return self.layer4(y)


def select_act(act: str, rec_weight) -> typing.Tuple:

    if act == 'leaky_relu':
        key = 'Target Prop - Leaky ReLU'
        if rec_weight is not None:
            key = f'{key} - {rec_weight}'
        activation = nn.LeakyReLU
        in_act = None
    elif act == 'sign':
        key = 'TargetProp - Sign'
        if rec_weight is not None:
            key = f'{key} - {rec_weight}'
        activation = nn.Tanh
        in_act = partial(Sign, False)
    elif act == 'binary':
        key = 'TargetProp - Binary'
        if rec_weight is not None:
            key = f'{key} - {rec_weight}'
        activation = nn.Sigmoid
        in_act = Binary
    elif act == 'sigmoid':
        key = 'TargetProp - Sigmoid'
        if rec_weight is not None:
            key = f'{key} - {rec_weight}'
        activation = nn.Sigmoid
        in_act = None
    elif act == 'sampler':
        key = 'Target Prop - Sampler'
        if rec_weight is not None:
            key = f'{key} - {rec_weight}'
        in_act = partial(Sampler, False, 1.0)
        activation = nn.Sigmoid
    elif act == 'clamp':
        key = 'Target Prop - Clamp'
        if rec_weight is not None:
            key = f'{key} - {rec_weight}'
        in_act = partial(Clamp, -1.0, 1.0, False)
        activation = partial(nn.LeakyReLU, 0.9)
    else:
        raise ValueError(f'Cannot use act {act}')
    
    return key, activation, in_act


def filter_acts(module: nn.Module, to_filter: typing.Type[nn.Module], seen: typing.Set=None):

    seen = seen or set()

    if isinstance(module, to_filter):
        yield module

    for child in module.children():
        for filtered in filter_acts(child, to_filter, seen):
            yield filtered

