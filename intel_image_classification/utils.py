from dataclasses import dataclass

import seaborn as sns
import torch
import torchvision
from matplotlib.figure import Figure
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification.confusion_matrix import MulticlassConfusionMatrix
from torchmetrics.collections import MetricCollection


@dataclass
class DataAndLoaders:
    train_loader: DataLoader
    valid_loader: DataLoader
    test_loader: DataLoader
    classes: list[str]


@dataclass
class OptimizationSuite:
    criterion: torch.nn.modules.loss._Loss
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler


@dataclass
class LoggingSuite:
    metrics: MetricCollection
    conf_matrix: MulticlassConfusionMatrix
    writer: SummaryWriter


@dataclass
class TrainerConfig:
    checkpoint_path: str


# def figure_of_confusion_matrix(matrix, class_names):
#     fig = px.imshow(
#         matrix,
#         labels=dict(x="predicted", y="true class"),
#         x=class_names,
#         y=class_names,
#         aspect="equal",
#         text_auto=True,
#         width=600,
#         height=600,
#     )
#     fig.update_xaxes(side="top")
#     return fig


def figure_of_confusion_matrix(matrix, class_names) -> Figure:
    ax = sns.heatmap(
        matrix, annot=True, xticklabels=class_names, yticklabels=class_names
    )
    ax.set(xlabel="predicted", ylabel="true labels")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    fig = ax.get_figure(root=True)
    # Чтобы pyright не жаловался на fig: Figure | None
    assert fig is not None, "Seaborn failed to create a figure"
    return fig


def get_datasets_and_loaders(
    train_transform,
    test_transform,
    train_batch_size=32,
    test_batch_size=64,
    num_workers=4,
):
    train_set = torchvision.datasets.ImageFolder(
        "data/seg_train", transform=train_transform
    )
    full_test_set = torchvision.datasets.ImageFolder(
        "data/seg_test", transform=test_transform
    )
    valid_set, test_set = torch.utils.data.random_split(
        full_test_set, [0.5, 0.5], torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=test_batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )
    classes = train_set.classes

    return DataAndLoaders(train_loader, valid_loader, test_loader, classes)


if __name__ == "__main__":
    pass
