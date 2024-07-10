import zenkai
import zenkai
import torch
from torch import nn
from torch.utils import data as torch_data
import numpy as np
import tqdm
from zenkai import LearningMachine


class Layer(nn.Module):
    # # 3) Layer

    # A simple layer that uses

    # - dropout for denoising capabilities
    # - linear transformation
    # - normalization (if used)
    # - activation (if used)

    def __init__(
        self, in_features: int, out_features: int,
        use_norm: bool, use_act: bool, dropout_p: float=None
    ):

        super().__init__()
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else lambda x: x
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.BatchNorm1d(out_features) if use_norm else lambda x: x
        self.act = nn.LeakyReLU() if use_act else lambda x: x

    @property
    def p(self) -> float:
        return self.dropout.p if self.dropout is not None else None

    @p.setter
    def p(self, p: float):
        if p is None:
            self.dropout = lambda x: x
        else:
            self.dropout = nn.Dropout(p)
        return p

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = self.dropout(x)
        y = self.linear(y)
        y = self.norm(y)
        y = self.act(y)
        return y


def train(
    learner: LearningMachine,
    dataset: torch_data.Dataset,
    n_epochs: int, device='cpu'
):
    learner = learner.to(device)
    loss = nn.CrossEntropyLoss(reduction='mean')

    zenkai.set_lmode(learner, zenkai.LMode.WithStep)

    for i in range(n_epochs):

        dataloader = torch_data.DataLoader(dataset, batch_size=128, shuffle=True)

        with tqdm.tqdm(total=len(dataloader)) as pbar:

            results = {'loss': []}

            for j, (x, x1_t) in enumerate(dataloader):
                x = x.to(device)
                x1_t = x1_t.to(device)

                y = learner(x.view(x.shape[0], -1))
                assessment = loss(y, x1_t)
                assessment.backward()
                results['loss'].append(assessment.item())
                assessments = {i: v.item() for i, v in enumerate(learner.assessments)}
                r_assessments = {i: v.item() for i, v in enumerate(learner.r_assessments)}

                for i, v in assessments.items():
                    if str(i) not in results:
                        results[str(i)] = []
                    results[str(i)].append(v)

                for i, v in r_assessments.items():
                    if f'r_{i}' not in results:
                        results[f'r_{i}'] = []
                    results[f'r_{i}'].append(v)

                pbar.set_postfix({
                    k: np.mean(v) for k, v in results.items()
                })
                pbar.update(1)
