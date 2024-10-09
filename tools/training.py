from zenkai import LearningMachine
import zenkai
import typing
import tqdm
from torch.utils import data as torch_data
import torch.nn as nn
import torch
import numpy as np

from zenkai import LearningMachine
from torch.utils import data as torch_data
import torch
import numpy as np
import matplotlib.pyplot as plt


def train(
    learner: LearningMachine,
    dataset: torch_data.Dataset,
    n_epochs: int, device='cpu',
    batch_size: int=128,
    validate: bool=False,
    callback: typing.Callable[[int, int], None]=None,
    flatten: bool=True
):
    learner = learner.to(device)
    loss = nn.CrossEntropyLoss(reduction='mean')

    zenkai.set_lmode(learner, zenkai.LMode.WithStep)
    losses = []

    total_iters = 0
    for i in range(n_epochs):

        dataloader = torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        with tqdm.tqdm(total=len(dataloader)) as pbar:

            results = {'loss': []}

            for j, (x, x1_t) in enumerate(dataloader):
                x = x.to(device)
                x1_t = x1_t.to(device)

                if validate:
                    before = zenkai.params.to_pvec(learner)
                if flatten:
                    x = x.flatten(1)
                
                y = learner(x)
                assessment = loss(y, x1_t)
                assessment.backward()
                if validate:
                    assert (before != zenkai.params.to_pvec(learner)).any()
                results['loss'].append(assessment.item())
                losses.append(assessment.item())
                assessments = {i: v.item() if isinstance(v, torch.Tensor) else v for i, v in enumerate(learner.assessments)}

                for i, v in assessments.items():
                    if str(i) not in results:
                        results[str(i)] = []
                    results[str(i)].append(v)

                pbar.set_postfix({
                    k: np.mean(v) for k, v in results.items()
                })
                pbar.update(1)
                if callback is not None:
                    callback(i, j, total_iters)
                total_iters += 1
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


def plot_loss_line(loss_values, names: typing.List, title='Training Loss Over Epochs', save_file: typing.Optional[str]=None):
    """
    Plots training loss as a line plot for a variable number of models.

    Args:
    - loss_values (list of lists): A list of lists, where each sublist contains the loss values for a model over epochs.
    - title (str): Title of the plot.
    """
    # Create figure and axis
    fig, ax = plt.subplots()
    
    # Colormap and linestyle options to differentiate between models
    colormaps = ['blue', 'green', 'red', 'purple', 'orange']
    linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    
    # Plot loss for each model
    for i, (name, losses) in enumerate(zip(names, loss_values)):
        epochs = np.arange(1, len(losses) + 1)  # Create a sequence of epochs (x-axis)
        color = colormaps[i % len(colormaps)]
        linestyle = linestyles[i % len(linestyles)]
        
        ax.plot(epochs, losses, label=name, color=color, linestyle=linestyle, linewidth=2)
    
    # Set labels and title
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    
    # Create legend
    ax.legend(loc='upper right')

    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight')
    # Show plot
    plt.show()
