import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from lib import *
from lib.utils import *
import logging
import importlib

# import torcheval.metrics as ms
# import torch.utils.data as td

sns.set_theme(style="darkgrid")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using: {device}. Device: {torch.cuda.get_device_name()}")


if is_notebook():
    from IPython.display import display, clear_output
