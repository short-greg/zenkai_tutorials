import zenkai
import zenkai
from zenkai import State, IO, Idx, iou
from torch import Tensor
import zenkai
from .utils import Layer


class AutoencoderLearner(zenkai.GradLearner):
    """Component for building a TargetPropagationLearner
    """

    def __init__(self, in_features: int, out_features: int, rec_weight: float=1.0, dropout_p: float=None):
        """Create an AutoencoderLearner with specificed input and output Features

        Args:
            in_features (int): number of input features
            out_features (int): number of output features
            rec_weight (float, optional): the weight on the reconstruction. Defaults to 1.0.
            dropout_p (float, optional): The amount of dropout to use. Defaults to None.
        """
        super().__init__()
        self._criterion = zenkai.NNLoss('MSELoss', reduction='sum')
        self._reverse_criterion = zenkai.NNLoss('MSELoss', reduction='sum')
        self.forward_on = True
        self.reverse_on = True
        self.feedforward = Layer(
            in_features, out_features, True, True, dropout_p
        )
        self.rec_weight = rec_weight

        self.feedback = Layer(
            out_features, in_features, False, True
        )
        self.assessment = None
        self.r_assessment = None
        # Considering changing this
        self._optim = zenkai.OptimFactory('Adam', lr=1e-3).comp()
        self._optim.prep_theta(self)
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
        t_loss = self._criterion.assess(state._y, t)
        (t_loss + self.rec_weight * z_loss).backward()
        self.i += 1
        self.assessment = z_loss
        self.r_assessment = t_loss

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
        return iou(self.feedback(t.f))



class TargetPropLearner(zenkai.GradLearner):

    def __init__(
        self, in_features: int, h1_features: int,
        h2_features: int, h3_features: int, out_features: int
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
            in_features, h1_features, 1.0, 0.5
        )
        self.layer2 = AutoencoderLearner(
            h1_features, h2_features, 1.0, dropout_p=0.25
        )
        self.layer3 = AutoencoderLearner(
            h2_features, h3_features, 1.0, dropout_p=0.25
        )
        self.layer4 = Layer(
            h3_features, out_features, False, False
        )
        self._optim = zenkai.OptimFactory('Adam', lr=1e-3).comp()
        self._optim.prep_theta([self.layer4])
        self.assessments = []
        self.r_assessments = []

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
            in_features, h1_features, True, True, 0.5
        )
        self.layer2 = Layer(
            h1_features, h2_features, True, True, dropout_p=0.25
        )
        self.layer3 = Layer(
            h2_features, h3_features, True, True, dropout_p=0.25
        )
        self.layer4 = Layer(
            h3_features, out_features, False, False
        )
        self._optim = zenkai.OptimFactory('Adam', lr=1e-3).comp()
        self._optim.prep_theta(self)
        self.assessments = []
        self.r_assessments = []

    def forward_nn(self, x: IO, state: State, batch_idx: Idx = None) -> Tensor:

        y = self.layer1(x.f)
        y = self.layer2(y)
        y = self.layer3(y)
        return self.layer4(y)

