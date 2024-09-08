from zenkai import LearningMachine
import zenkai
import tqdm
from torch.utils import data as torch_data
import torch.nn as nn
import torch
import numpy as np


def train(
    learner: LearningMachine,
    dataset: torch_data.Dataset,
    n_epochs: int, device='cpu'
):
    learner = learner.to(device)
    loss = nn.CrossEntropyLoss(reduction='mean')

    zenkai.set_lmode(learner, zenkai.LMode.WithStep)
    losses = []

    for i in range(n_epochs):

        dataloader = torch_data.DataLoader(dataset, batch_size=128, shuffle=True)

        with tqdm.tqdm(total=len(dataloader)) as pbar:

            results = {'loss': []}

            for j, (x, x1_t) in enumerate(dataloader):
                x = x.to(device)
                x1_t = x1_t.to(device)

                before = zenkai.params.to_pvec(learner)
                y = learner(x.view(x.shape[0], -1))
                assessment = loss(y, x1_t)
                assessment.backward()
                assert (before != zenkai.params.to_pvec(learner)).any()
                results['loss'].append(assessment.item())
                losses.append(assessment.item())
                assessments = {i: v.item() for i, v in enumerate(learner.assessments)}

                for i, v in assessments.items():
                    if str(i) not in results:
                        results[str(i)] = []
                    results[str(i)].append(v)

                pbar.set_postfix({
                    k: np.mean(v) for k, v in results.items()
                })
                pbar.update(1)
    return losses

def classify(
    learner: LearningMachine,
    dataset: torch_data.Dataset,
    device='cpu',
    # transform,
):
    dl = torch_data.DataLoader(dataset, len(dataset))
    images, labels = next(iter(dl))
    images = images.flatten(1)
    # images, labels = dataset[:]

    # images = transform(images)
    y = learner(images)
    outputs = torch.argmax(y, dim=-1)

    return (outputs == labels).float().sum() / len(labels)

