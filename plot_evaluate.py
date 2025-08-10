import os
from typing import Dict, Union, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import ast
import seaborn as sn

from mpl_toolkits import axes_grid1
import matplotlib.pyplot as plt

def add_colorbar(im, aspect=10, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def string_to_tensor(cka: str) -> torch.Tensor:
    """
    This function will get a string to convert it to tensor.
    :param cka: The cka row that stored in string.
    :return:
    Torch
    """
    list_str = cka.replace('tensor(', '').rstrip(')')
    return torch.tensor(ast.literal_eval(list_str))

def string_to_list(layers:str) -> List[str]:
    """
    This function will get a string to convert it to list of string.
    :param layers: The layers that stored in string.
    :return:
    List of strings for the layers.
    """
    return ast.literal_eval(layers)

def stacking_layers(dataframe:pd.DataFrame) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """
    A dictionary with specified layers and stacked tensors.
    :param dataframe:
    :return:
    A dictionary with specified layers and stacked tensors.
    """
    num_layers = len(dataframe)
    result = {
        'model1_layers': [],
        'model2_layers': dataframe.iloc[0]['model2_layers'],
        'CKA': torch.zeros((num_layers, len(dataframe.iloc[0]['model2_layers'])))
    }
    for index, row in dataframe.iterrows():
        result['model1_layers'].append(row['model1_layers'][0] if isinstance(row['model1_layers'], list) else row['model1_layers'])
        cka_tensor = row['CKA']
        if cka_tensor.ndim == 2 and cka_tensor.shape[0] == 1:
            cka_tensor = cka_tensor.squeeze(0)
        result['CKA'][index, :] = cka_tensor
    return result

def plot_heat_map(cka:Dict[str, Union[torch.Tensor, List[str]]], title:str=None, save_path:str=None) -> None:
    """
    This function will plot a heatmap of the CKA.
    :param cka: The CKA tensor.
    :param title: The title of the figure.
    :param save_path: The path to save the figure.
    :return:
    The figure.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(cka['CKA'], origin='lower', cmap='magma')
    # ax.set_xlabel(f"Layers {cka['model1_layers']}", fontsize=15)
    # ax.set_ylabel(f"Layers {cka['model2_layers']}", fontsize=15)
    if title is not None:
        ax.set_title(f"{title}", fontsize=18)
    add_colorbar(im)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_cka(
    cka_matrix: torch.Tensor,
    first_layers: list[str],
    second_layers: list[str],
    first_name: str = "First Model",
    second_name: str = "Second Model",
    save_path: str=None,
    title: str=None,
    vmin: float=None,
    vmax: float= None,
    cmap: str = "magma",
    show_ticks_labels: bool = False,
    short_tick_labels_splits: int = None,
    use_tight_layout: bool = True,
    show_annotations: bool = True,
    show_img: bool = True,
    show_half_heatmap: bool = False,
    invert_y_axis: bool = True,
) -> None:
    """Plot the CKA matrix obtained by calling CKA class __call__() method.

    Args:
        cka_matrix (torch.Tensor): the CKA matrix.
        first_layers (list[str]): list of the names of the first model's layers.
        second_layers (list[str]): list of the names of the second model's layers.
        first_name (str): name of the first model (default="First Model").
        second_name (str): name of the second model (default="Second Model").
        save_path (str | None): where to save the plot, if None then the plot will not be saved (default=None).
        title (str | None): the plot title, if None then a simple text with the name of both models will be used
            (default=None).
        vmin (float | None): values to anchor the colormap, otherwise they are inferred from the data and other keyword
            arguments (default=None).
        vmax (float | None): values to anchor the colormap, otherwise they are inferred from the data and other keyword
            arguments (default=None).
        cmap (str): the name of the colormap to use (default="magma").
        show_ticks_labels (bool): whether to show the tick labels (default=False).
        short_tick_labels_splits (int | None): only works when show_tick_labels is True. If it is not None, the tick
            labels will be shortened to the defined sublayer starting from the deepest level. E.g.: if the layer name
            is 'encoder.ff.linear' and this parameter is set to 1, then only 'linear' will be printed on the heatmap
            (default=None).
        use_tight_layout (bool): whether to use a tight layout in order not to cut any label in the plot (default=True).
        show_annotations (bool): whether to show the annotations on the heatmap (default=True).
        show_img (bool): whether to show the plot (default=True).
        show_half_heatmap (bool): whether to mask the upper left part of the heatmap since those valued are duplicates
            (default=False).
        invert_y_axis (bool): whether to invert the y-axis of the plot (default=True).

    Raises:
        ValueError: if ``vmax`` or ``vmin`` are not defined together or both equal to None.
    """
    # Deal with vmin and vmax
    if (vmin is not None) ^ (vmax is not None):
        raise ValueError("'vmin' and 'vmax' must be defined together or both equal to None.")

    vmin = min(vmin, torch.min(cka_matrix).item()) if vmin is not None else vmin
    vmax = max(vmax, torch.max(cka_matrix).item()) if vmax is not None else vmax

    # Build the mask
    mask = torch.tril(torch.ones_like(cka_matrix, dtype=torch.bool), diagonal=-1) if show_half_heatmap else None

    # Build the heatmap
    if mask:
        ax = sn.heatmap(cka_matrix.cpu(), vmin=vmin, vmax=vmax, annot=show_annotations, cmap=cmap, mask=mask.cpu().numpy())
    else:
        ax = sn.heatmap(cka_matrix.cpu(), vmin=vmin, vmax=vmax, annot=show_annotations, cmap=cmap)
    if invert_y_axis:
        ax.invert_yaxis()

    ax.set_xlabel(f"{second_name} layers", fontsize=12)
    ax.set_ylabel(f"{first_name} layers", fontsize=12)

    # Deal with tick labels
    if show_ticks_labels:
        if short_tick_labels_splits is None:
            ax.set_xticklabels(second_name)
            ax.set_yticklabels(first_name)
        else:
            ax.set_xticklabels(["-".join(module.split(".")[-short_tick_labels_splits:]) for module in second_layers])
            ax.set_yticklabels(["-".join(module.split(".")[-short_tick_labels_splits:]) for module in first_layers])

        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

    # Put the title if passed
    if title is not None:
        ax.set_title(title, fontsize=14)
    else:
        title = f"{first_name} vs {second_name}"
        ax.set_title(title, fontsize=14)

    # Set the layout to tight if the corresponding parameter is True
    if use_tight_layout:
        plt.tight_layout()

    # Save the plot to the specified path if defined
    if save_path is not None:
        title = title.replace(" ", "_").replace("/", "-")
        path_rel = f"{save_path}/{title}.png"
        plt.savefig(path_rel, dpi=400, bbox_inches="tight")

    # Show the image if the user chooses to do so
    if show_img:
        plt.show()

if __name__ == '__main__':
    arrow_df = pd.read_csv("arrow-results.csv")
    gks_df = pd.read_csv("gks-results.csv")

    arrow_df['CKA'] = arrow_df['CKA'].apply(string_to_tensor)
    gks_df['CKA'] = gks_df['CKA'].apply(string_to_tensor)

    arrow_df['model1_layers'] = arrow_df['model1_layers'].apply(string_to_list)
    arrow_df['model2_layers'] = arrow_df['model2_layers'].apply(string_to_list)

    gks_df['model1_layers'] = gks_df['model1_layers'].apply(string_to_list)
    gks_df['model2_layers'] = gks_df['model2_layers'].apply(string_to_list)

    arrow_cka = stacking_layers(arrow_df)
    gks_cka = stacking_layers(gks_df)

    # plot_heat_map(arrow_cka, title="CKA for Phi3 and Phi3-Arrow")
    # plot_heat_map(gks_cka, title="CKA for Phi3 and Phi3-GKS")


    arrow_df = pd.read_csv("arrow-results (1).csv")
    gks_df = pd.read_csv("gks-results (1).csv")
    arrow_cka = torch.zeros((len(arrow_df), len(arrow_df)))
    gks_cka = torch.zeros((len(gks_df), len(gks_df)))
    for index, row in arrow_df.iterrows():
        arrow_cka[index, :] = string_to_tensor(row[f'layer:model.layers.{index}'])
    for index, row in gks_df.iterrows():
        gks_cka[index, :] = string_to_tensor(row[f'layer:model.layers.{index}'])

    os.makedirs("results", exist_ok=True)
    plot_cka(
        arrow_cka.to('cuda'),
        first_layers=[f"model.layers.{i}" for i in range(32)],
        second_layers=[f"base_model.model.model.layers.{i}" for i in range(32)],
        first_name="Phi3",
        second_name="Arrow",
        save_path="results/",
        show_annotations=False,
    )

    plot_cka(
        gks_cka.to('cuda'),
        first_layers=[f"model.layers.{i}" for i in range(32)],
        second_layers=[f"base_model.model.model.layers.{i}" for i in range(32)],
        first_name="Phi3",
        second_name="GKS",
        save_path="results/",
        show_annotations=False,
    )

    mean_diag_arrow = torch.mean(torch.diag(arrow_cka))
    mean_diag_gks = torch.mean(torch.diag(gks_cka))
    print("Mean Diagonal CKA: Arrow", mean_diag_arrow.item(), "GKS", mean_diag_gks.item())

    whole_mean_arrow = torch.mean(arrow_cka)
    whole_mean_gks = torch.mean(gks_cka)

    print(f"Mean CKA: Arrow", whole_mean_arrow.item(), "GKS", whole_mean_gks.item())

    norm_arrow_cka = (arrow_cka - arrow_cka.min()) / (arrow_cka.max() - arrow_cka.min())
    norm_gks_cka = (gks_cka - gks_cka.min()) / (gks_cka.max() - gks_cka.min())

    plot_cka(
        norm_arrow_cka.to('cuda'),
        first_layers=[f"model.layers.{i}" for i in range(32)],
        second_layers=[f"base_model.model.model.layers.{i}" for i in range(32)],
        first_name="Phi3",
        second_name="Normalized Arrow",
        save_path="results/",
        show_annotations=False,
    )

    plot_cka(
        norm_gks_cka.to('cuda'),
        first_layers=[f"model.layers.{i}" for i in range(32)],
        second_layers=[f"base_model.model.model.layers.{i}" for i in range(32)],
        first_name="Phi3",
        second_name="Normalized GKS",
        save_path="results/",
        show_annotations=False,
    )

    mean_diag_arrow = torch.mean(torch.diag(norm_arrow_cka))
    mean_diag_gks = torch.mean(torch.diag(norm_gks_cka))
    print("Mean Diagonal CKA: Normalized Arrow", mean_diag_arrow.item(), " Normalized GKS", mean_diag_gks.item())

    whole_mean_arrow = torch.mean(norm_arrow_cka)
    whole_mean_gks = torch.mean(norm_gks_cka)

    print(f"Mean CKA: Normalized Arrow", whole_mean_arrow.item(), " Normalized GKS", whole_mean_gks.item())