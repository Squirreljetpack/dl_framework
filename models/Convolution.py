import torch.nn as nn
import torch

from lib.modules import Classifier


class CNN(Classifier):
    """
    Simple CNN model
    """

    def __init__(self, n_classes, conv_layers, dropout, conv1_outc=32) -> None:
        super().__init__()
        self.save_attr()

        self.conv1 = nn.Sequential(
            nn.LazyConv2d(conv1_outc, 5, 3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.maxpool = nn.MaxPool2d(2, 1, 1)
        self.features = self.make_feature_layers(conv_layers)
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv1_outc, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def make_feature_layers(self, conv_layers: int) -> nn.Sequential:
        layers = []
        for _ in range(conv_layers):
            layers += [
                nn.Conv2d(self.conv1_outc, self.conv1_outc, 3),
                nn.BatchNorm2d(self.conv1_outc),
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
