import numpy as np
import torch
import seaborn as sns
from lib import *
from lib.Utils import *
import importlib

sns.set_theme(style="darkgrid")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using: {device}. Device: {torch.cuda.get_device_name()}")


if is_notebook():
    from IPython.display import display, clear_output
