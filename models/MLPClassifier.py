import torch
import torch.nn as nn
from lib.Utils import *
from lib.modules import *



class MLPClassifier(Classifier):
    def __init__(
        self, features, hidden_size, layers, flavor="rnn", dropout=0.2, y_len=1
    ):