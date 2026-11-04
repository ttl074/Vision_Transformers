"""
Vision Transformer (ViT) for CIFAR-100, implemented from scratch.

Defines two tokenizers (a linear patch embedding and a ResNet-56 residual
CNN tokenizer), the encoder building blocks (multi-head self-attention, MLP,
stochastic depth, and a pre-norm transformer block), and the full model,
configured through ``ViTConfig``. With the patch embedding, a linear
projection of non-overlapping patches with no convolution, the model is a
plain ViT; with the ResNet tokenizer, it is a ResNet-Transformer hybrid with
the same encoder design.
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ViTConfig:
    """
    Hyperparameters for the ViT model and its hybrid variant.

    Defaults give a compact encoder in the style of CCT-7 (d_model 256,
    7 blocks, 4 heads, MLP ratio 2) on 32x32 inputs with patch size 4. Each
    transformer block has 527,104 parameters. With ``tokenizer="patch"``
    (plain ViT) the model has 3,744,868 parameters at depth 7; with
    ``tokenizer="resnet"`` (hybrid), a ResNet-56 feature extractor replaces
    the patch embedding, for 4,604,084 parameters at depth 7. ``depth`` is
    used to match the two models' sizes (a plain ViT at depth 9 has
    4,799,076 parameters, within 5% of the hybrid).

    Attributes:
        img_size (int): Height and width of the square input images.
        patch_size (int): Height and width of each square patch. Must divide
            ``img_size``. Used only when ``tokenizer="patch"``.
        n_channels (int): Number of input image channels.
        num_classes (int): Number of output classes.
        d_model (int): Embedding dimension of each token.
        depth (int): Number of transformer blocks.
        n_heads (int): Number of attention heads. Must divide ``d_model``.
        mlp_ratio (int): Hidden size of each MLP as a multiple of ``d_model``.
        dropout (float): Dropout probability for attention weights and MLPs.
        drop_path_rate (float): Stochastic-depth rate of the last transformer
            block; rates rise linearly from 0 at the first block.
        tokenizer (str): How images become tokens: ``"patch"`` for the linear
            patch embedding (plain ViT) or ``"resnet"`` for the ResNet
            tokenizer (hybrid).
        blocks_per_stage (int): Residual blocks in each of the ResNet
            tokenizer's three stages; 9 gives ResNet-56 (depth 6n + 2).
            Used only when ``tokenizer="resnet"``.
    """
    # pylint: disable=too-many-instance-attributes
    img_size: int = 32
    patch_size: int = 4
    n_channels: int = 3
    num_classes: int = 100
    d_model: int = 256
    depth: int = 7
    n_heads: int = 4
    mlp_ratio: int = 2
    dropout: float = 0.0
    drop_path_rate: float = 0.1
    tokenizer: str = "patch"
    blocks_per_stage: int = 9


class PatchEmbedding(nn.Module):
    """
    Split images into non-overlapping patches and project each to a token.

    Each patch is flattened to ``n_channels * patch_size * patch_size`` values
    and mapped to ``d_model`` by one shared linear layer.

    Args:
        d_model (int): Embedding dimension of each token.
        img_size (int): Height and width of the square input images.
        patch_size (int): Height and width of each square patch.
        n_channels (int): Number of input image channels.

    Raises:
        ValueError: If ``img_size`` is not divisible by ``patch_size``.
    """
    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"{img_size} is not divisible by {patch_size}")
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.linear_proj = nn.Linear(n_channels * patch_size * patch_size, d_model)

    def forward(self, x):
        """
        Convert a batch of images to patch tokens.

        Args:
            x (torch.Tensor): Images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Tokens of shape (B, num_patches, d_model), one per
            patch in row-major order.
        """
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, self.num_patches, C * p * p)
        x = self.linear_proj(x)
        return x


class ResidualBlock(nn.Module):
    """
    Basic residual block: two 3x3 convolutions plus a shortcut connection.

    Computes ReLU(F(x) + shortcut(x)), where F is conv, batch norm, ReLU,
    conv, batch norm. The first convolution applies ``stride``. The shortcut
    is the identity when the shape is unchanged; otherwise it is a 1x1
    convolution with the same stride, followed by batch norm, projecting the
    input to the output's shape so the two can be added.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        stride (int): Stride of the first convolution and the projection
            shortcut; 2 halves the height and width.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        """
        Apply the residual block.

        Args:
            x (torch.Tensor): Feature maps of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Feature maps of shape
            (B, out_channels, H / stride, W / stride).
        """
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class ResNetTokenizer(nn.Module):
    """
    ResNet feature extractor that turns images into transformer tokens.

    A CIFAR-style ResNet (He et al., 2016): a 3x3 convolution stem to 16
    channels, then three stages of ``blocks_per_stage`` residual blocks at
    16, 32, and 64 channels, the last two downsampling 2x. With the default
    of 9 blocks per stage this is ResNet-56. A 1x1 convolution then projects
    each cell of the 8x8 feature map to ``d_model``, and each cell becomes
    one token. Output shape matches ``PatchEmbedding``, so the two are
    interchangeable in ``ViT``.

    Args:
        d_model (int): Embedding dimension of each token.
        img_size (int): Height and width of the square input images.
        n_channels (int): Number of input image channels.
        blocks_per_stage (int): Residual blocks in each stage.

    Raises:
        ValueError: If ``img_size`` is not divisible by ``DOWNSAMPLE``, or
            ``blocks_per_stage`` is less than 1.
    """
    # two stride-2 stages: 32 → 16 → 8
    DOWNSAMPLE = 4

    def __init__(self, d_model, img_size, n_channels, blocks_per_stage=9):
        super().__init__()
        if img_size % self.DOWNSAMPLE != 0:
            raise ValueError(f"{img_size} not divisible by {self.DOWNSAMPLE}")
        if blocks_per_stage < 1:
            raise ValueError(f"blocks_per_stage {blocks_per_stage} < 1")
        self.num_patches = (img_size // self.DOWNSAMPLE) ** 2
        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.stages = nn.Sequential(
            self._make_stage(16, 16, blocks_per_stage, 1),
            self._make_stage(16, 32, blocks_per_stage, 2),
            self._make_stage(32, 64, blocks_per_stage, 2)
        )
        self.proj = nn.Conv2d(64, d_model, kernel_size=1)

    @staticmethod
    def _make_stage(in_channels, out_channels, n_blocks, stride):
        """
        Build one stage of residual blocks.

        The first block applies ``stride`` and changes the channel count; the
        remaining ``n_blocks - 1`` keep the shape.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels of every block.
            n_blocks (int): Number of residual blocks.
            stride (int): Stride of the first block.

        Returns:
            nn.Sequential: The stage.
        """
        blocks = [ResidualBlock(in_channels, out_channels, stride)]
        blocks += [ResidualBlock(out_channels, out_channels)
                   for _ in range(n_blocks - 1)]
        return nn.Sequential(*blocks)

    def forward(self, x):
        """
        Convert a batch of images to tokens.

        Args:
            x (torch.Tensor): Images of shape (B, n_channels, H, W).

        Returns:
            torch.Tensor: Tokens of shape (B, num_patches, d_model), one per
            cell of the downsampled feature map in row-major order.
        """
        x = self.stem(x)
        x = self.stages(x)
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head scaled dot-product self-attention.

    Projects each token to queries, keys, and values, computes attention
    independently in each head, then concatenates the heads and applies an
    output projection.

    Args:
        d_model (int): Embedding dimension of each token.
        n_heads (int): Number of attention heads.
        dropout (float): Dropout probability applied to the attention weights.

    Raises:
        ValueError: If ``d_model`` is not divisible by ``n_heads``.
    """
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"{d_model} is not divisible by {n_heads}!")
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Apply self-attention across the tokens of each image.

        Args:
            x (torch.Tensor): Tokens of shape (B, N, d_model).

        Returns:
            torch.Tensor: Attention output of shape (B, N, d_model).
        """
        B, N, D = x.shape
        x = self.qkv(x)
        x = x.reshape(B, N, 3, self.n_heads, self.head_dim)
        x = x.permute(2, 0, 3, 1, 4)
        q, k, v = x.unbind(0)
        scores = q @ k.transpose(-2, -1) * self.scale
        weights = scores.softmax(dim=-1)
        weights = self.attn_dropout(weights)
        attn_out = weights @ v
        attn_out = attn_out.transpose(1, 2)
        attn_out = attn_out.reshape(B, N, D)
        attn_out = self.proj(attn_out)
        return attn_out


class MLP(nn.Module):
    """
    Two-layer feed-forward network applied to each token independently.

    Expands each token to ``mlp_ratio * d_model`` features, applies GELU,
    and projects back to ``d_model``.

    Args:
        d_model (int): Embedding dimension of each token.
        mlp_ratio (int): Hidden size as a multiple of ``d_model``.
        dropout (float): Dropout probability after each layer.
    """
    def __init__(self, d_model, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, int(mlp_ratio * d_model))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(mlp_ratio * d_model), d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Apply the feed-forward network to each token.

        Args:
            x (torch.Tensor): Tokens of shape (B, N, d_model).

        Returns:
            torch.Tensor: Tokens of shape (B, N, d_model).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DropPath(nn.Module):
    """
    Stochastic depth: randomly drop a residual branch, per sample, in training.

    During training, each sample's input is zeroed with probability ``p`` and
    otherwise scaled by 1 / (1 - p), so the expected output is unchanged.
    Applied to a residual branch, a dropped sample skips the block. In
    evaluation mode, or when ``p`` is 0, the input is returned unchanged.

    Args:
        p (float): Drop probability, in [0, 1).

    Raises:
        ValueError: If ``p`` is not in [0, 1).
    """
    def __init__(self, p=0.0):
        super().__init__()
        if p >= 1 or p < 0:
            raise ValueError(f"p {p} not in [0,1)")
        self.p = p

    def forward(self, x):
        """
        Drop or rescale each sample's input.

        Args:
            x (torch.Tensor): Tensor whose first dimension is the batch.

        Returns:
            torch.Tensor: Tensor of the same shape as ``x``.
        """
        if not self.training or self.p == 0.0:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer encoder block.

    Applies layer normalization before self-attention and before the MLP,
    each wrapped in a residual connection. Each residual branch passes
    through stochastic depth before being added back.

    Args:
        config (ViTConfig): Model hyperparameters (uses d_model, n_heads,
            mlp_ratio, and dropout).
        drop_path (float): Stochastic-depth rate for both residual branches.
    """
    def __init__(self, config, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(config.d_model, config.n_heads, config.dropout)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config.d_model, config.mlp_ratio, config.dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        """
        Apply one transformer block.

        Args:
            x (torch.Tensor): Tokens of shape (B, N, d_model).

        Returns:
            torch.Tensor: Tokens of shape (B, N, d_model).
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViT(nn.Module):
    """
    Vision Transformer for image classification, plain or hybrid.

    Converts images to tokens with the tokenizer chosen by
    ``config.tokenizer`` (``"patch"``: linear patch embedding, the plain ViT;
    ``"resnet"``: ResNet-56 tokenizer, the hybrid), adds learned position
    embeddings, applies a stack of transformer blocks (with stochastic depth
    rising linearly from 0 to ``config.drop_path_rate``) and a final layer norm,
    averages the tokens (mean pooling), and maps the result to class logits.
    Everything after the tokenizer has the same design in both variants;
    only ``depth`` may differ, to match parameter counts.

    Args:
        config (ViTConfig): Model hyperparameters.

    Raises:
        ValueError: If ``config.tokenizer`` is not ``"patch"`` or ``"resnet"``.
    """
    def __init__(self, config):
        super().__init__()
        if config.tokenizer == "patch":
            self.tokenizer = PatchEmbedding(config.d_model, config.img_size,
                                            config.patch_size, config.n_channels)
        elif config.tokenizer == "resnet":
            self.tokenizer = ResNetTokenizer(config.d_model, config.img_size,
                                             config.n_channels, config.blocks_per_stage)
        else:
            raise ValueError(f"Unknown tokenizer: {config.tokenizer}")
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.tokenizer.num_patches, config.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        rates = torch.linspace(0, config.drop_path_rate, config.depth).tolist()
        self.blocks = nn.Sequential(*[
            TransformerBlock(config, drop_path=rate)
            for rate in rates])
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize a submodule's weights; called on every submodule by ``apply``.

        Linear layers get truncated-normal weights (std 0.02) and zero biases;
        LayerNorms get unit weights and zero biases. Convolutions and batch
        norms in the ResNet tokenizer keep PyTorch's defaults (Kaiming
        initialization for convolutions). The position embedding is a
        parameter, not a module, so it is initialized separately.

        Args:
            module (nn.Module): The submodule to initialize.
        """
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Classify a batch of images.

        Args:
            x (torch.Tensor): Normalized images of shape
                (B, n_channels, img_size, img_size).

        Returns:
            torch.Tensor: Class logits of shape (B, num_classes), before
            softmax.
        """
        x = self.tokenizer(x)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)
