import torch
from torch import nn


class Net(nn.Module):
    """Архитектура на базе DenseNet со сквозными связями (skip connections).

    Реализует кастомные блоки, где каждый последующий слой получает
    карты признаков всех предыдущих слоев в рамках блока.
    """

    def __init__(self):
        super().__init__()
        self.num_of_basic_blocks = 3
        self.block0 = nn.Sequential(
            # 3x130x130
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2),
            # 64x63x63
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # 64x31x31
        )
        self.basic_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_features=64),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_features=64),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_features=64),
                    nn.ReLU(),
                )
                for i in range(self.num_of_basic_blocks)
            ]
        )
        self.conv1x1_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
        )
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
        )
        self.final_pool = nn.AvgPool2d(kernel_size=31)
        self.final_linear = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=32, out_features=6),
        )

    def forward(self, x0):
        x0 = self.block0(x0)
        x1 = x0
        x0 = self.basic_blocks[0](x0)
        x0 = torch.cat((x0, x1), dim=1)
        x0 = self.conv1x1_0(x0)
        x2 = x0
        x0 = self.basic_blocks[1](x0)
        x0 = torch.cat((x0, x1, x2), dim=1)
        x0 = self.conv1x1_1(x0)
        x3 = x0
        x0 = self.basic_blocks[2](x0)
        x0 = torch.cat((x0, x2, x3), dim=1)
        x0 = self.conv1x1_2(x0)
        x0 = self.final_pool(x0)
        x0 = torch.flatten(x0, start_dim=1)
        x0 = self.final_linear(x0)
        return x0


if __name__ == "__main__":
    pass
