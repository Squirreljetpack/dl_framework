import torch
import torch.nn as nn
from lib.utils import *
from lib.modules import *



class MLPClassifier(Classifier):
    def __init__(self, hidden_size, output_size, n_hidden_layers):
        super(MLPClassifier, self).__init__()
        self.input_layer = nn.Sequential(
            nn.LazyLinear(hidden_size), nn.ReLU(inplace=True)
        )
        self.hidden_layers = self._make_hidden_layers(n_hidden_layers, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def _make_hidden_layers(self, n_hidden_layers, hidden_size):
        layers = []
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x
