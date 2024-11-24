import inspect
from contextlib import contextmanager
import os
from typing import List
import torch

from functools import reduce
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def is_notebook():
    # credit -> https://stackoverflow.com/a/39662359
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def apply_kwargs(func, kwargs):
    """
    Filters a dictionary of keyword arguments to only include those
    accepted by the given function, then applies them to the function.

    Parameters:
        func (callable): The function to which arguments will be applied.
        kwargs (dict): Dictionary of keyword arguments.

    Returns:
        The result of calling `func` with the filtered keyword arguments.
    """
    sig = inspect.signature(func)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(**valid_kwargs)


class Base:
    def save_parameters(self, ignore=[]):
        """Save function arguments into self.p"""

        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.p = {
            k: v
            for k, v in local_vars.items()
            if k not in set(ignore + ["self"]) and not k.startswith("_")
        }
        # for k, v in self.hparams.items():
        # setattr(self, k, v)

    def save_attr(self, ignore=[]):
        """Save function arguments into class attributes."""

        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.p = {
            k: v
            for k, v in local_vars.items()
            if k not in set(ignore + ["self"]) and not k.startswith("_")
        }
        for k, v in self.p.items():
            setattr(self, k, v)

    @property
    def device(self):
        try:
            return self.gpus[0]
        except:  # noqa: E722
            return "cpu"

    def defined_or_self(self, variable_names):
        """
        Checks if the variable corresponding to `variable_name` is None.
        If it's None, sets it to the value of `self.variable_name` using getattr(self, variable_name).

        Args:
            variable_name (str): The name of the variable to check and potentially update.

        Returns:
            The value of the variable, or `None` if it's not defined.
        """
        # Get the current local variables in the function
        frame = inspect.currentframe().f_back
        local_vars = frame.f_locals

        if not isinstance(variable_names, list):
            variable_names = [variable_names]

        for vn in variable_names:
            # Check if the variable is defined and is None
            value = local_vars.get(vn, None)

            if value is None:
                # If it's None, set it to self.variable_name using getattr(self, variable_name)
                value = getattr(self, vn, None)
                local_vars[vn] = value


def get_gpus(gpus: int | List[int] = -1, vendor="cuda"):
    """Given num_gpus or array of ids, returns a list of torch devices

    Args:
        gpus (int | List[int], optional): [] for cpu. Defaults to -1 for all gpus.
        vendor (str, optional): vendor_string. Defaults to "cuda".

    Returns:
        _type_: _description_
    """
    if isinstance(gpus, list):
        assert [int(i) for i in gpus]
    elif gpus == -1:
        gpus = range(torch.cuda.device_count())
    else:
        assert gpus <= torch.cuda.device_count()
        gpus = range(gpus)
    return [torch.device(f"{vendor}:{i}") for i in gpus]


@contextmanager
def change_dir(target_dir):
    original_dir = os.getcwd()
    try:
        os.chdir(target_dir)
        yield
    finally:
        os.chdir(original_dir)


def mean_of_dicts(dict_list):
    keys = dict_list[0].keys()
    total_dict = reduce(
        lambda acc, val: {key: acc.get(key, 0) + val.get(key, 0) for key in keys},
        dict_list,
    )
    mean_dict = {key: total_dict[key] / len(dict_list) for key in keys}

    return mean_dict


def k_level_list(q, k=1):
    test = q
    while k > 0 and isinstance(test, list):
        if len(test) == 0:
            break
        test = test[0]
        k -= 1
    for _ in range(k):
        q = [q]
    return q
