from zenkai import LearningMachine
from torch.utils import data as torch_data
import torch
import numpy as np
import matplotlib.pyplot as plt


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


def plot_loss_line(loss_values, title='Training Loss Over Epochs'):
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
    for i, losses in enumerate(loss_values):
        epochs = np.arange(1, len(losses) + 1)  # Create a sequence of epochs (x-axis)
        color = colormaps[i % len(colormaps)]
        linestyle = linestyles[i % len(linestyles)]
        
        ax.plot(epochs, losses, label=f'Model {i+1}', color=color, linestyle=linestyle, linewidth=2)
    
    # Set labels and title
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    
    # Create legend
    ax.legend(loc='upper right')

    # Show plot
    plt.show()
