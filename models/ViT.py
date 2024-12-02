import torch.nn as nn
import torch
import torch.nn.functional as F

from lib.modules import Classifier


class Head(nn.Module):
    def __init__(self, head_size, n_embed, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        k, q, v = self.key(x), self.query(x), self.value(x)
        out = F.softmax(q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5, dim=-1)
        out = self.dropout(out)
        out = out @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, n_heads, n_embed, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embed) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class MLP(nn.Module):
    def __init__(self, n_embed, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(inplace=True),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_heads, n_embed, dropout=0.2):
        super().__init__()
        head_size = n_embed // n_heads
        self.attention = MultiHeadAttention(head_size, n_heads, n_embed, dropout)
        self.ffwd = MLP(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            emb_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformer(Classifier):
    def __init__(
        self,
        in_channels,
        patch_size,
        n_patches,
        emb_size,
        n_heads,
        n_layers,
        n_classes,
        class_freqs,
        dropout=0.2,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + n_patches, emb_size))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_heads, emb_size, dropout) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, n_classes)

        self.head.bias = nn.Parameter(torch.log(class_freqs))
        self.apply(init_weights)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
        x = x.mean(dim=1)  # vs taking mean?
        x = self.head(x)
        return x


def init_weights(module):
    """
    Initialise weights of given module using Kaiming Normal initialisation for linear and
    convolutional layers, and zeros for bias.
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
