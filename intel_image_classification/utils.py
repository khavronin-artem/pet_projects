import seaborn as sns
import torch
import torchvision
from matplotlib.figure import Figure
from torchvision.transforms import v2

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
    train_batch_size=32, valid_batch_size=64, train_num_workers=4, valid_num_workers=4
):
    train_transform = v2.Compose(
        [
            v2.Resize(size=140),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomCrop(size=(130, 130)),
            v2.ColorJitter(brightness=0.1, contrast=0.1),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    train_set = torchvision.datasets.ImageFolder(
        "data/seg_train", transform=train_transform
    )
    full_test_set = torchvision.datasets.ImageFolder(
        "data/seg_test",
        transform=v2.Compose(
            [
                v2.Resize(size=140),
                v2.CenterCrop(size=(130, 130)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
    )
    valid_set, test_set = torch.utils.data.random_split(
        full_test_set, [0.5, 0.5], torch.Generator().manual_seed(42)
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=train_num_workers,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=valid_batch_size, num_workers=valid_num_workers
    )

    return train_set, train_loader, valid_set, valid_loader


if __name__ == "__main__":
    pass
