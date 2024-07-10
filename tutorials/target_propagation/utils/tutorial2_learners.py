import zenkai
import zenkai
from zenkai import State, IO, Idx
import zenkai
from .utils import Layer

from .tutorial1_learners import AutoencoderLearner


class DiffAutoencoderLearner(AutoencoderLearner):

    def step_x(self, x: IO, t: IO, state: State, batch_idx: Idx = None) -> IO:
        """Propagate the target and the output back and calculate the difference

        Args:
            x (IO): the input
            t (IO): the target
            state (State): the learning state

        Returns:
            IO: the target for the incoming layer
        """

        yx = self.feedback(state._y.f)
        tx = self.feedback(t.f)

        return x.acc_dx([yx - tx]).detach()


class DiffTargetPropLearner(zenkai.GradLearner):

  def __init__(
    self, in_features: int, h1_features: int,
    h2_features: int, h3_features: int, out_features: int
  ):
    # Same as TargetPropLearner
    # but uses the DiffAutoencoderLearner
    super().__init__()
    self._criterion = zenkai.NNLoss('MSELoss', reduction='mean')
    self._learn_criterion = zenkai.NNLoss('MSELoss', reduction='sum', weight=0.5)
    self.forward_on = True
    self.reverse_on = True

    self.layer1 = DiffAutoencoderLearner(
      in_features, h1_features, 1.0, 0.5
    )
    self.layer2 = DiffAutoencoderLearner(
      h1_features, h2_features, 1.0, dropout_p=0.25
    )
    self.layer3 = DiffAutoencoderLearner(
      h2_features, h3_features, 1.0, dropout_p=0.25
    )
    self.layer4 = Layer(
      h3_features, out_features, False, False
    )
    self._optim = zenkai.OptimFactory('Adam', lr=1e-3).comp()
    self._optim.prep_theta([self.layer4])
    self.assessments = []
    self.r_assessments = []
