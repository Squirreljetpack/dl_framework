from torch import nn
from torch.nn import functional as F

from .utils import *

# Do not use properties beginning with _
class Module(nn.Module, Base):
    """The base class of models."""

    def __init__(self):
        super().__init__()
        self.save_attr()

    ## Training section
    # Defaults for classification are used and should likely be overridden

    def loss(self, outputs, y):
        loss = F.mse_loss(outputs, y.reshape(outputs.shape), reduction="mean")
        return loss

    def forward(self, X):
        assert hasattr(self, "net"), "Neural network is defined"
        return self.net(X)

    @property
    def filename(self):
        # join key-value pairs from self.p
        param_str = "__".join([f"{k}_{v}" for k, v in self.p.items()])
        return f"{self.__class__.__name__}_{param_str}"

    def pred(self, output):
        return output.squeeze(-1)

    def layer_summary(self, X_shape):
        """Displays model output dimensions for each layer given input X-shape.

        Args:
            X_shape (tuple)
        """
        X = torch.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, "output shape:\t", X.shape)

    def show(self):
        for p in self.named_parameters():
            print(p)


# Just a reminder of basic field names, better to declare all fields when subclassing and subclass Config instead
@dataclass(kw_only=True)
class ClassifierConfig(Config):
    n_blks: int  # num primary blocks
    n_classes: int
    dropout: float = 0.1
    hidden_size: int  # first MLP


class Classifier(Module):
    """
    The base class of classification models.
    """

    def loss(self, outputs, Y, averaged=True):
        outputs = outputs.reshape(-1, outputs.shape[-1])
        Y = Y.reshape(
            -1,
        )
        loss = F.cross_entropy(outputs, Y, reduction="mean" if averaged else "none")
        return loss

    # Used in evaluation/metrics steps, which compare pred(output) and label
    def pred(self, output):
        return output.argmax(dim=-1).to(torch.int64)

    # test manual computation agrees
    def _softmax(self, logits):
        exp_logits = np.exp(
            logits - np.max(logits, axis=1, keepdims=True)
        )  # Stability trick
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # probs = self.softmax(outputs.cpu().detach().numpy())

        # true_probs = probs[np.arange(len(Y)), Y.cpu().numpy()]

        # nloss = -np.mean(np.log(true_probs))


class MultiClassifier(Module):
    """
    The base class of classification models.
    """

    def loss(self, outputs, Y, averaged=True):
        outputs = outputs.reshape(-1, outputs.shape[-1])
        Y = Y.reshape(
            -1,
        )
        return F.cross_entropy(outputs, Y, reduction="mean" if averaged else "none")

    def pred(self, output):
        return output.argmax(dim=1)
