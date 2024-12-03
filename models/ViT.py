import collections
import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.modules import Classifier
from models.Attention import MultiHeadAttention
from lib.utils import *


class ViTMLP(nn.Module):
    def __init__(self, mlp_n_hiddens, mlp_n_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_n_hiddens)
        self.gelu = nn.GELU()  # SOTA tend to use GELU
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_n_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(self.dense1(x)))))


class ViTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        mlp_n_hiddens,
        n_heads,
        dropout,
        use_bias=False,
    ):
        super().__init__()
        norm_shape = hidden_size  # unsure if this is ever different
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = MultiHeadAttention(hidden_size, n_heads, dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_n_hiddens, hidden_size, dropout)

    def forward(self, X, valid_lens=None):
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        return X + self.mlp(self.ln2(X))


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, emb_size):
        super().__init__()
        image_size = (
            image_size
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )
        patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )
        self.n_patches = (image_size[1] // patch_size[1]) * (
            image_size[0] // patch_size[0]
        )

        self.conv = nn.LazyConv2d(  # infers img size and channels
            emb_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )  # patches may not cover the whole img if they don't line up

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)

@dataclass(kw_only=True)
class ViTConfig(Config):
    img_size: int
    patch_size: int = 16  # 16x16 for 224x224 in original
    n_classes: int
    n_blks: int  # num primary blocks
    dropout: float = 0.1
    emb_size: int
    n_heads: int
    hidden_size: int = 1024  # MLP hidden size
    emb_dropout: float = 0.2
    class_freqs: Any = None
    attn_use_bias: bool = False


class ViT(Classifier):
    def __init__(self, c: ViTConfig):
        # n_patches/seq_len is inferred from img_size
        super().__init__()
        self.save_config(c)
        self.patch_embedding = PatchEmbedding(c.img_size, c.patch_size, c.emb_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, c.emb_size))
        n_steps = self.patch_embedding.n_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(torch.randn(1, n_steps, c.emb_size))
        self.dropout = nn.Dropout(c.emb_dropout)
        self.blks = nn.Sequential()
        for i in range(c.n_blks):
            self.blks.add_module(
                f"{i}",
                ViTBlock(
                    c.emb_size,
                    c.hidden_size,
                    c.n_heads,
                    c.dropout,
                    c.attn_use_bias,
                ),
            )
        self.head = nn.Sequential(
            nn.LayerNorm(c.emb_size), nn.Linear(c.emb_size, c.n_classes)
        )
        if c.class_freqs is not None:
            self.head[1].bias = nn.Parameter(torch.log(c.class_freqs))  # aids imbalance
            # self.apply(init_weights)

    def forward(self, X):
        X = self.patch_embedding(X)
        X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])  # vs head(x.mean(dim=1))  ?


def init_weights(module):
    """
    Initialise weights of given module using Kaiming Normal initialisation for linear and
    convolutional layers, and zeros for bias.
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
