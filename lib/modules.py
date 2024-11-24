from torch import nn
from torch.nn import functional as F

from .Utils import *


class Module(nn.Module, Base):
    """The base class of models."""

    def __init__(self):
        super().__init__()
        self.save_attr()

    ## Training section
    # Defaults for classification are used and should likely be overridden

    def loss(self, outputs, y):
        loss = F.mse_loss(outputs.squeeze(), y.squeeze(), reduction="mean")
        return loss

    def forward(self, X):
        assert hasattr(self, "net"), "Neural network is defined"
        return self.net(X)

    @property
    def filename(self):
        # join key-value pairs from self.p
        param_str = "__".join([f"{k}_{v}" for k, v in self.p.items()])
        return f"{self.__class__.__name__}_{param_str}"


class Classifier(Module):
    """
    The base class of classification models.

    Defined in :numref:`sec_classification`
    """

    def accuracy(self, outputs, Y, averaged=True):
        """
        Compute the number of correct predictions.

        Defined in :numref:`sec_classification`
        """
        outputs = torch.reshape(outputs, (-1, outputs.shape[-1]))
        preds = torch.argmax(outputs, axis=1)
        compare = torch.astype(preds == torch.reshape(Y, -1), torch.float32)
        return torch.reduce_mean(compare) if averaged else compare

    def loss(self, outputs, Y, averaged=True):
        """Defined in :numref:`sec_softmax_concise`"""
        outputs = outputs.reshape(-1, outputs.shape[-1])
        Y = Y.reshape(
            -1,
        )
        return F.cross_entropy(outputs, Y, reduction="mean" if averaged else "none")

    # For metrics
    def _pred(output):
        F.argmax(dim=1)

    def layer_summary(self, X_shape):
        """Defined in :numref:`sec_lenet`"""
        X = torch.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, "output shape:\t", X.shape)


class MultiClassifier(Module):
    """
    The base class of classification models.

    Defined in :numref:`sec_classification`
    """

    def accuracy(self, outputs, Y, *, averaged=True):
        """
        Compute the number of correct predictions.

        Defined in :numref:`sec_classification`
        """
        outputs = torch.reshape(outputs, (-1, outputs.shape[-1]))
        preds = torch.argmax(outputs, axis=1)
        compare = torch.astype(preds == torch.reshape(Y, -1), torch.float32)
        return torch.reduce_mean(compare) if averaged else compare

    def loss(self, outputs, Y, averaged=True):
        """Defined in :numref:`sec_softmax_concise`"""
        outputs = outputs.reshape(-1, outputs.shape[-1])
        Y = Y.reshape(
            -1,
        )
        return F.cross_entropy(outputs, Y, reduction="mean" if averaged else "none")

    def _pred(output):
        F.softmax(output, dim=-1)
