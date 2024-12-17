import zenkai
import torch
from torch import nn
import typing
from itertools import chain
from functools import reduce


class SignSTE(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass of the Binary Step function.
        """
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x < -1) | (x > 1)] = 0
        return grad_input


def sign_ste(x: torch.Tensor) -> torch.Tensor:
    """Execute the sign function

    Args:
        x (torch.Tensor): the input

    Returns:
        torch.Tensor: -1 for values less than 0 otherwise 1
    """
    return SignSTE.apply(x)


# class StochasticSTE(torch.autograd.Function):
#     """Use to clip the grad between two values
#     Useful for smooth maximum/smooth minimum
#     """

#     @staticmethod
#     def forward(ctx, x):
#         """
#         Forward pass of the Binary Step function.
#         """
#         ctx.save_for_backward(x)
#         x = torch.sigmoid(x)
#         return (x <= torch.rand_like(x)).type_as(x)

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         Backward pass of the Binary Step function using the Straight-Through Estimator.
#         """
#         (x,) = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad_input[(x < 0) | (x > 1)] = 0
#         return grad_input


class ClampSTE(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """
    @staticmethod
    def forward(ctx, x, lower: float=-1.0, upper: float=1.0, g: float=0.01):
        """
        Forward pass of the Binary Step function.
        """
        ctx.save_for_backward(x)
        ctx.g = g
        ctx.lower = lower
        ctx.upper = upper
        return torch.clamp(x, lower, upper)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x < ctx.lower) | (x > ctx.upper)] = ctx.g
        return grad_input, None, None, None


class L2Reg(nn.Module):

    def __init__(self, lam: float=1e-3, reduction: str='sum'):
        super().__init__()

        self._lam = lam
        self.reduction = reduction

    def forward(self, mods: typing.List[nn.Module]):

        return reduce(
            lambda p, a: (p + a.sum() if self.reduction == 'sum' else a.mean()),
            (p for p in chain(mods.parameters() for mod in mods)), 0.0
        )




# def stochastic_ste(x: torch.Tensor) -> torch.Tensor:
#     """Execute the sign function

#     Args:
#         x (torch.Tensor): the input

#     Returns:
#         torch.Tensor: -1 for values less than 0 otherwise 1
#     """
#     return StochasticSTE.apply(x)


def clamp_ste(x: torch.Tensor, lower: float=-1.0, upper: float=1.0, g: float=0.01) -> torch.Tensor:
    """Execute the sign function

    Args:
        x (torch.Tensor): the input

    Returns:
        torch.Tensor: -1 for values less than 0 otherwise 1
    """
    return ClampSTE.apply(x, lower, upper, g)


class SamplerSTE(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x):
        """
        Forward pass of the Binary Step function.
        """
        ctx.save_for_backward(x)
        return (x <= torch.rand_like(x)).type_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x < 0) | (x > 1)] = 0
        return grad_input


class Sampler(nn.Module):

    def __init__(self, use_ste: bool=False, temperature: float=1.0):
        super().__init__()
        self._use_ste = use_ste
        self.temperature = temperature
        self.binary = Binary()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.temperature != 1.0:
            x = (x * 2) - 1.
            s = torch.sign(x)
            a = torch.abs(x)

            x = ((s * a ** self.temperature) + 1) / 2.0
        if self._use_ste:
            return SamplerSTE.apply(x)
        
        y = ((torch.rand_like(x) <= x).type_as(x))

        return y


# class Stochastic(nn.Module):

#     def __init__(self, use_ste: bool=False, use_tanh: bool=False, temperature: float=1.0):
#         super().__init__()
#         self._use_ste = use_ste
#         self._use_tanh = use_tanh
#         self.temperature = temperature
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:

#         if self._use_ste:
#             return SamplerSt(x)
#         if self._use_tanh:
#             x = torch.tanh(x)

#         # if self.temperature != 1.0:
#         # x = (x * 2) - 1
#         s = torch.sign(x)
#         a = torch.abs(x)

#         x = ((s * a ** self.temperature) + 1) / 2.0
#         y = ((torch.rand_like(x) <= x).type_as(x) * 2.0) - 1.0
#         return y


class Clamp(nn.Module):

    def __init__(
        self, lower: float=-1.0, upper: float=1.0, 
        use_ste: bool=False, g: float=0.01
    ) -> None:
        super().__init__()
        self._use_ste = use_ste
        self._lower = lower
        self._upper = upper
        self._g = g

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self._use_ste:
            return clamp_ste(x, self._lower, self._upper, self._g)
        return x.clamp(self._lower, self._upper)


class Sign(nn.Module):

    def __init__(self, use_ste: bool=False) -> None:
        super().__init__()
        self._use_ste = use_ste

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self._use_ste:
            return sign_ste(x)
        return x.sign()


class Binary(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # self._use_ste = use_ste

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # if self._use_ste:
        #     return binary_ste(x)
        return (x >= 0.5).type_as(x)


# class Clamp(nn.Module):

#     def forward(self, x: torch.Tensor) -> torch.Tensor:

#         return x.clamp(-1.0, 1.0)


class NullModule(nn.Module):

    def forward(self, *x) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]:

        if len(x) == 1:
            return x[0]
        return x


class Layer(nn.Module):

    def __init__(
        self, in_features: int, out_features: int, 
        in_activation: typing.Type[nn.Module]=None, 
        out_activation: typing.Type[nn.Module]=None,
        dropout_p: float=None,
        batch_norm: bool=False,
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

    def forward(self, x: torch.Tensor) -> typing.Union[typing.Any, None]:
        
        y = self.in_activation(x)
        y = self.dropout(y)
        y = self.linear(y)
        y = self.norm(y)
        return self.activation(y)


# class Layer(nn.Module):
#     """A layer consisting of an in_activtion, an out_activation, dropout, and
#     linear operations
#     """

#     def __init__(
#         self, in_features: int, out_features: int, 
#         in_activation: typing.Type[nn.Module]=None, 
#         out_activation: typing.Type[nn.Module]=None,
#         dropout_p: float=None,
#         batch_norm: bool=False,
#     ):
#         """Create a normal nn.Module layer

#         Args:
#             in_features (int): The number of input features
#             out_features (int): The number of output features
#             in_activation (typing.Type[nn.Module], optional): The activation on the input. Defaults to None.
#             out_activation (typing.Type[nn.Module], optional): The activation on the output. Defaults to None.
#             dropout_p (float, optional): The amount to dropout by. Defaults to None.
#             batch_norm (bool, optional): Whether to use batch norm. Defaults to False.
#         """

#         super().__init__()
#         self.linear = nn.Linear(in_features, out_features)
#         self.loss = nn.MSELoss(reduction='sum')
#         self.in_activation = in_activation() if in_activation is not None else NullModule()
#         self.activation = out_activation() if out_activation is not None else NullModule()
#         self.norm = nn.BatchNorm1d(out_features) if batch_norm else NullModule()
#         self.dropout = nn.Dropout(dropout_p) if dropout_p else NullModule()

#     def forward(self, x: torch.Tensor) -> typing.Union[typing.Any, None]:
#         """Compute the output of the laeyr

#         Args:
#             x (torch.Tensor): 

#         Returns:
#             typing.Union[typing.Any, None]: 
#         """
        
#         y = self.in_activation(x)
#         y = self.dropout(y)
#         y = self.linear(y)
#         y = self.norm(y)
#         return self.activation(y)


# class LayerLearner(zenkai.LearningMachine):
#     """A Layer that implements learning functionality
#     """

#     def __init__(
#         self, in_features: int, out_features: int, 
#         in_activation: typing.Type[nn.Module]=None, 
#         out_activation: typing.Type[nn.Module]=None,
#         dropout_p: float=None,
#         batch_norm: bool=False,
#         x_lr: float=None,
#         lmode: zenkai.LMode=zenkai.LMode.Standard
#     ):
#         """Create a layer that implements learning functionality

#         Args:
#             in_features (int): The number of input features
#             out_features (int): The number of output features
#             in_activation (typing.Type[nn.Module], optional): The activation on the input. Defaults to None.
#             out_activation (typing.Type[nn.Module], optional): The activation on the output. Defaults to None.
#             dropout_p (float, optional): The amount to dropout by. Defaults to None.
#             batch_norm (bool, optional): Whether to use batch norm. Defaults to False.
#         """

#         super().__init__(lmode)
#         self.linear = nn.Linear(in_features, out_features)
#         self.loss = nn.MSELoss(reduction='sum')
#         self.in_activation = in_activation() if in_activation is not None else NullModule()
#         self.activation = out_activation() if out_activation is not None else NullModule()
#         self.norm = nn.BatchNorm1d(out_features) if batch_norm else NullModule()
#         self.dropout = nn.Dropout(dropout_p) if dropout_p else NullModule()
#         self.x_lr = x_lr
#         self.max_norm = 1.0
#         self._optim = torch.optim.Adam(self.parameters(), lr=1e-3)

#     def accumulate(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State, **kwargs):
#         """Calculate the cost and execute the backward function on the cost

#         Args:
#             x (zenkai.IO): The input
#             t (zenkai.IO): The target
#             state (zenkai.State): The state
#         """
#         cost = self.loss(state._y.f, t.f)
#         cost.backward()   
    
#     def step(self, x: IO, t: IO, state: State):
#         """Clip the gradients and then step

#         Args:
#             x (IO): The input
#             t (IO): The target
#             state (State): The learning state
#         """
#         torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_norm)
#         self._optim.step()
#         self._optim.zero_grad()

#     def step_x(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State, **kwargs) -> zenkai.IO:
#         """Accumulate the gradient

#         Args:
#             x (zenkai.IO): The input
#             t (zenkai.IO): The target
#             state (zenkai.State): The learning state

#         Returns:
#             zenkai.IO: 
#         """
#         return x.acc_grad(self.x_lr)

#     def forward_nn(self, x: zenkai.IO, state: zenkai.State, **kwargs) -> typing.Union[typing.Any, None]:
#         """

#         Args:
#             x (zenkai.IO): The input
#             state (zenkai.State): The learning state

#         Returns:
#             typing.Union[typing.Any, None]: 
#         """
#         y = self.in_activation(x.f)
#         y = self.dropout(y)
#         y = self.linear(y)
#         y = self.norm(y)
#         return self.activation(y)



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
