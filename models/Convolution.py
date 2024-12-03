import torch.nn as nn
import torch

from lib.modules import Classifier
from lib.utils import *


@dataclass(kw_only=True)
class CNNConfig(Config):
    n_blks: int  # num primary blocks
    n_classes: int
    dropout: float = 0.1
    hidden_size: int = 64  # first MLP
    channels: int = 32


class CNN(Classifier):
    """
    Simple CNN model
    """

    def __init__(self, c: CNNConfig) -> None:
        super().__init__()
        self.save_config(c)

        self.conv1 = nn.Sequential(
            nn.LazyConv2d(c.channels, 5, 3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(c.dropout),
        )
        self.maxpool = nn.MaxPool2d(
            2, 1, 1
        )  # reduce spatial dims but not present here, also increases translation invariance
        self.features = self.make_feature_layers(c.n_blks)  # shrinking
        self.conv2 = nn.Sequential(  # c.hidden_size features
            nn.Conv2d(c.channels, c.hidden_size, 1, bias=False),
            nn.BatchNorm2d(c.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(c.dropout),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Squeeze into hidden_sizex1x1
        self.classifier = nn.Sequential(
            nn.Linear(c.hidden_size, c.hidden_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_size, c.n_classes),
        )

    def make_feature_layers(self, n_blks: int) -> nn.Sequential:
        layers = []
        for _ in range(n_blks):
            layers += [
                nn.Conv2d(self.channels, self.channels, 3),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True),
            ]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.conv1(x))
        x = self.features(x)
        x = self.avgpool(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


# __all__ = ["CNN", "CNNConfig"]