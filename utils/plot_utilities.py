import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
import numpy as np

def create_complex_grid(grid_size, coordinates, plot_functions):
    """This function is IDENTICAL to the one in the previous answer."""
    fig = plt.figure(figsize=(24, 12))
    grid = gs.GridSpec(grid_size[0], grid_size[1], figure=fig)

    for i, plot_func in enumerate(plot_functions):
        row, cols = coordinates[i]
        ax = fig.add_subplot(grid[row, cols[0]:cols[1]])
        plot_func(ax) # <-- This works because plot_func is now a partial waiting for `ax`

    plt.tight_layout()
    plt.show()
    return fig

def plot_confusion_matrix_heatmap_normalized(ax, cm, title):
    # receives a confusion matrix and an axis
    # returns the axis with the plot

    labels_real = [1, 2, 3, 4, 5]     # classes reais (linhas)
    labels_pred = [0, 1, 2, 3, 4, 5]  # classes preditas (colunas)

    # Normalizar por linha (cada linha soma 1)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_percent = cm_normalized * 100  # converter para %

    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels_pred, yticklabels=labels_real, ax=ax)
    ax.set_xlabel('Classe Predita')
    ax.set_ylabel('Classe Real')
    ax.set_title(title)

def simple_plot(cm, title):
    fig, ax = plt.subplots()
    plot_confusion_matrix_heatmap_normalized(ax, cm, title)