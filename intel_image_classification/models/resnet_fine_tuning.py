import torchvision.models
from torch import nn


def get_resnet_model(classes):
    resnet = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT
    )
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features=in_features, out_features=len(classes))
    return resnet


if __name__ == "__main__":
    pass
