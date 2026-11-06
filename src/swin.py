"""
Swin Transformer for CIFAR-100, implemented from scratch.

Window partitioning and the shifted-window mask, windowed multi-head
attention with a relative position bias, Swin blocks, patch merging, and
the full model, configured through ``SwinConfig``. Reuses the patch
embedding, attention, MLP, stochastic depth, and weight initialization
from ``src.models``.
"""

from dataclasses import dataclass

import torch
from torch import nn

from src.models import PatchEmbedding, MultiHeadSelfAttention, MLP, DropPath, init_weights


def window_partition(x, window_size):
    """
    Split a grid of tokens into non-overlapping square windows.

    Each window becomes its own batch row, so attention can run within
    windows using ordinary batched attention. Windows are ordered row-major
    (left to right, top to bottom) within each image, and images in batch
    order; ``window_reverse`` relies on this order.

    Args:
        x (torch.Tensor): Tokens of shape (B, H, W, C).
        window_size (int): Height and width of each window. Must divide H
            and W.

    Returns:
        torch.Tensor: Windows of shape (B * num_windows, window_size ** 2, C),
        where num_windows = (H / window_size) * (W / window_size).
    """
    B, H, W, C = x.shape
    # reshape each spatial axis to get which window and where in the window:
    # (B, H/ws, ws, W/ws, ws, C)
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    # bring the window axes together, then the within-window axes: (B, H/ws, W/ws, ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5)
    # one window per batch row with its ws*ws tokens flattened: (B*num_windows, ws*ws, C)
    x = x.reshape(B * (H*W) // window_size**2, window_size**2, C)
    return x


def window_reverse(windows, window_size, H, W):
    """
    Reassemble windows from ``window_partition`` into a grid of tokens.

    Inverse of ``window_partition``: windows must be in the order it
    produces. The batch size is inferred from the number of windows.

    Args:
        windows (torch.Tensor): Windows of shape
            (B * num_windows, window_size ** 2, C).
        window_size (int): Height and width of each window.
        H (int): Height of the token grid.
        W (int): Width of the token grid.

    Returns:
        torch.Tensor: Tokens of shape (B, H, W, C).
    """
    # batch size is total windows / windows per image
    B = windows.shape[0] // (H * W // window_size ** 2)
    # channels are the last axis of each window's tokens
    C = windows.shape[2]
    # unflatten windows and their tokens: (B, H/ws, W/ws, ws, ws, C)
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, C)
    # undo the partition's permute: (B, H/ws, ws, W/ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5)
    # merge each spatial pair back into full axis: (B, H, W, C)
    x = x.reshape(B, H, W, C)
    return x


def shift_mask(grid, window_size, shift):
    """
    Build the attention mask for shifted windows.

    After the grid is cyclically shifted up and left by ``shift``, the last
    window along each axis holds tokens wrapped from the opposite edge of
    the image. Each axis is split into three bands (untouched windows,
    original edge tokens, wrapped tokens), giving nine labeled regions;
    tokens may attend only to tokens with the same label. Windows are in
    ``window_partition`` order, matching the attention input.

    Args:
        grid (int): Height and width of the token grid.
        window_size (int): Height and width of each window.
        shift (int): Shift size, between 1 and window_size - 1.

    Returns:
        torch.Tensor: Mask of shape (num_windows, N, N), N = window_size ** 2:
        0 where attention is allowed and -100 where it is blocked.
    """
    labels = torch.zeros(1, grid, grid, 1)
    # a roll only affects the bottom row and right column of windows
    bands = (slice(0, -window_size), slice(-window_size, -shift), slice(-shift, None))
    label = 0
    # assigns label in mask for each band section
    for rows in bands:
        for cols in bands:
            labels[:, rows, cols, :] = label
            label += 1
    # cut the label grid into windows where the tokens will be cut so order matches
    windows = window_partition(labels, window_size).squeeze(-1)
    # compare every pair of labels within each window; same label = 0, different label != 0
    diff = windows[:, None, :] - windows[:, :, None]
    # converts that into -100 (blocked) or 0 (allowed)
    return torch.where(diff != 0, -100.0, 0.0)


class WindowAttention(nn.Module):
    """
    Multi-head self-attention within windows, with a relative position bias.

    Used by Swin blocks. Each input row is one window of window_size x
    window_size tokens (from ``window_partition``), and attention runs only
    among the tokens of the same window. Instead of an absolute position
    embedding, a learned bias is added to the attention scores for each
    relative offset between two tokens, separately per head; pairs with the
    same offset share one bias. For shifted windows, an optional mask blocks
    attention between tokens that were wrapped together by the shift.

    Args:
        d_model (int): Embedding dimension of each token.
        n_heads (int): Number of attention heads. Must divide ``d_model``.
        window_size (int): Height and width of each window.
        dropout (float): Dropout probability applied to the attention weights.

    Raises:
        ValueError: If ``d_model`` is not divisible by ``n_heads``.
    """
    def __init__(self, d_model, n_heads, window_size, dropout=0.0):
        super().__init__()

        # standard multi-head attention, run with the window bias added to its scores
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        # table of learned biases for each relative offset within a window per head:
        # ((2*ws-1)^2 offsets, n_heads)
        self.rel_pos_bias = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, n_heads))
        # small truncated-normal initialization, similar to pos_embed
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)
        # maps each token pair to its offset row in the bias table;
        # buffer: moves to the GPU with the model, but isn't a trainable parameter
        self.register_buffer("rel_pos_index", self._rel_pos_idx(window_size))

    @staticmethod
    def _rel_pos_idx(window_size):
        """
        Build the map from each token pair in a window to its bias-table row.

        Tokens are numbered row-major within the window, matching
        ``window_partition``. Entry (i, j) encodes the row and column offset
        from token j to token i, shifted to be non-negative, as one integer
        in [0, (2 * window_size - 1) ** 2).

        Args:
            window_size (int): Height and width of each window.

        Returns:
            torch.Tensor: Long tensor of shape (N, N), N = window_size ** 2.
        """
        # gives each positions row and column
        rows, cols = torch.meshgrid(torch.arange(window_size),
                                     torch.arange(window_size), indexing="ij")
        # stack and flatten into a (2, N) table of coordinates
        coords = torch.stack([rows, cols])
        coords = coords.flatten(1)
        # calculate offsets for every pair, (2, N, N)
        rel = coords[:, :, None] - coords[:, None, :]
        # offsets range over -(ws-1) and  ws-1, shift so offsets start at 0
        rel = rel + (window_size -1)
        # encode the row and column offset pairs as one integer
        index = rel[0] * (2 * window_size - 1) + rel[1]
        return index

    def forward(self, x, mask=None):
        """
        Apply attention within each window.

        Args:
            x (torch.Tensor): Windows of shape (B * num_windows, N, d_model),
                N = window_size ** 2, from ``window_partition``.
            mask (torch.Tensor, optional): Shift mask of shape
                (num_windows, N, N), 0 where attention is allowed and a large
                negative value where it is blocked. Shared by all images and
                heads. None for unshifted windows.

        Returns:
            torch.Tensor: Windows of shape (B * num_windows, N, d_model).
        """
        # look up learned bias for each token pair: (N, N, n_heads)
        bias = self.rel_pos_bias[self.rel_pos_index]
        # heads first to match the attention scores: (n_heads, N, N)
        bias = bias.permute(2, 0, 1)
        # add the mask for shifted blocks, copied once per image in the batch
        if mask is not None:
            # number of images = total windows/ windows per image
            B = x.shape[0] // mask.shape[0]
            # same mask for every head and same bias for every window
            bias = bias + mask.repeat(B, 1, 1).unsqueeze(1)
        # run normal attention
        return self.attn(x, bias)


@dataclass
class SwinConfig:
    """
    Hyperparameters for the Swin Transformer.

    Defaults are a small three-stage Swin for 32x32 inputs: 2x2 patches give
    a 16x16 token grid of ``embed_dim`` channels; each later stage halves the
    grid (patch merging) and doubles the channels, giving 16x16x64,
    8x8x128, and 4x4x256. Attention runs within window_size x window_size
    windows, alternating regular and shifted windows. Sized to be within 10%
    of the hybrid's parameter count.

    Attributes:
        img_size (int): Height and width of the square input images.
        patch_size (int): Height and width of each square patch. Must divide
            ``img_size``.
        n_channels (int): Number of input image channels.
        num_classes (int): Number of output classes.
        embed_dim (int): Token channels in the first stage; doubled at each
            later stage.
        depth (tuple): Number of Swin blocks in each stage.
        n_heads (tuple): Attention heads in each stage. Each must divide
            that stage's channels.
        window_size (int): Height and width of each attention window. Stages
            whose grid is no larger than the window use one unshifted window.
        mlp_ratio (int): Hidden size of each MLP as a multiple of the stage's
            channels.
        dropout (float): Dropout probability for attention weights and MLPs.
        drop_path_rate (float): Stochastic-depth rate of the last block; rates
            rise linearly from 0 across all blocks of all stages.
    """
    # pylint: disable=too-many-instance-attributes
    img_size: int = 32
    patch_size: int = 2
    n_channels: int = 3
    num_classes: int = 100
    embed_dim: int = 64
    depth: tuple = (2, 4, 4)
    n_heads: tuple = (2, 4, 8)
    window_size: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.0
    drop_path_rate: float = 0.1


class SwinBlock(nn.Module):
    """
    Pre-norm Swin Transformer block with regular or shifted windows.

    Like ``TransformerBlock``, but on a (B, H, W, C) token grid, with
    self-attention restricted to non-overlapping windows. In shifted blocks
    the grid is cyclically shifted by half a window before partitioning and
    shifted back afterward, so information flows across the previous
    block's window borders; a mask blocks attention between tokens that
    were only grouped by the wrap-around. When the stage's grid is no larger
    than the window, the whole grid is one window and no shift is applied.
    Each residual branch passes through stochastic depth.

    Args:
        config (SwinConfig): Model hyperparameters.
        stage (int): Stage index (0, 1, 2), which sets the channels
            (``embed_dim * 2 ** stage``), heads, and grid size.
        shifted (bool): Use shifted windows.
        drop_path (float): Stochastic-depth rate for both residual branches.
    """
    def __init__(self, config, stage, shifted, drop_path):
        super().__init__()

        # channels double each stage
        dim = config.embed_dim * 2**stage
        heads = config.n_heads[stage]
        # grid dimensions shrink as patches merge
        grid = config.img_size // config.patch_size // 2**stage
        self.window_size = config.window_size
        if grid <= self.window_size:
            self.window_size = grid
            self.shift = 0
        else:
            self.shift = self.window_size // 2 if shifted else 0
        if self.shift != 0:
            self.register_buffer("attn_mask", shift_mask(grid, self.window_size, self.shift))
        else:
            self.register_buffer("attn_mask", None)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads, self.window_size, config.dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, config.mlp_ratio, config.dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        """
        Apply one Swin block.

        Args:
            x (torch.Tensor): Token grid of shape (B, H, W, C).

        Returns:
            torch.Tensor: Token grid of shape (B, H, W, C).
        """
        H, W = x.shape[1], x.shape[2]
        shortcut = x
        x = self.norm1(x)
        if self.shift:
            # slides every token up-left by shift amount with wrapping
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
        # splits the grid into windows of ws*ws: (B*nW, ws*ws, C)
        windows = window_partition(x, self.window_size)
        # apply multi-head self attention for each window
        windows = self.attn(windows, self.attn_mask)
        # reassemble windows into original grid of tokens
        x = window_reverse(windows, self.window_size, H, W)
        if self.shift:
            # if shifted slide back to original positions
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))
        # attention branch residual with stochastic depth
        x = shortcut + self.drop_path(x)
        # MLP branch with its own residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """
    Downsample a token grid 2x between Swin stages, doubling the channels.

    Each 2x2 group of neighboring tokens is stacked into one vector of 4C
    values, normalized with LayerNorm, and projected to 2C by a linear layer
    without bias (the LayerNorm already provides a learnable shift). Halving
    the resolution while doubling the channels keeps the cost of each stage
    roughly balanced, as in a CNN.

    Args:
        dim (int): Channels C of the input tokens.
    """
    def __init__(self, dim):
        super().__init__()

        # normalize over stacked token vectors
        self.norm = nn.LayerNorm(4 * dim)
        # reduce to 2C to keep cost of each stage balanced
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        """
        Merge each 2x2 group of tokens into one.

        Args:
            x (torch.Tensor): Token grid of shape (B, H, W, C); H and W must
                be even.

        Returns:
            torch.Tensor: Token grid of shape (B, H / 2, W / 2, 2C).
        """
        B, H, W, C = x.shape
        # split each axis into which 2x2 group and where in group: (B, H/2, 2, W/2, 2, C)
        x = x.reshape(B, H // 2, 2, W // 2, 2, C)
        # group axes together and within-group axes together: (B, H/2, W/2, 2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5)
        # stack each group's 4 tokens into one vector: (B, H/2, W/2, 4C)
        x = x.reshape(B, H // 2, W // 2, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class Swin(nn.Module):
    """
    Swin Transformer for image classification.

    Embeds 2x2 patches as tokens on a 16x16 grid, then applies three stages
    of Swin blocks with alternating regular and shifted windows, with patch
    merging between stages (16x16x64 -> 8x8x128 -> 4x4x256). Stochastic
    depth rises linearly across all blocks. A final layer norm, mean pooling
    over the grid, and a linear head give class logits. Positions are
    encoded only by each block's relative position bias, so there is no
    absolute position embedding.

    Args:
        config (SwinConfig): Model hyperparameters.
    """
    def __init__(self, config):
        super().__init__()

        # 2x2 patches, each linearly projected to an embed_dim token (same class as ViT)
        self.patch_embedding = PatchEmbedding(config.embed_dim, config.img_size,
                                        config.patch_size, config.n_channels)
        # drop-path rate rises linearly over all blocks of all stages
        rates = torch.linspace(0, config.drop_path_rate, sum(config.depth)).tolist()
        layers = []
        # build each stage's blocks, alternating regular and shifted windows
        for i, n_blocks in enumerate(config.depth):
            for j in range(n_blocks):
                # odd blocks shift windows so information crosses window border
                shifted = j % 2 == 1
                # blocks in earlier stages + position for this stage
                rate = rates[sum(config.depth[:i]) + j]
                layers.append(SwinBlock(config, i, shifted, rate))
            # merge patches for all stages but last
            if i < len(config.depth) - 1:
                layers.append(PatchMerging(config.embed_dim * 2**i))
        self.stages = nn.Sequential(*layers)

        # dimensionality in final stage
        final_dim = config.embed_dim * 2 ** (len(config.depth) - 1)
        # final LayerNorm because pre-norm leaves the residual stream unnormalized
        self.norm = nn.LayerNorm(final_dim)
        # classifier on the mean-pooled tokens
        self.head = nn.Linear(final_dim, config.num_classes)
        # token grid size (16), used for reshaping tokens into a grid
        self.grid = config.img_size // config.patch_size
        self.apply(init_weights)

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
        x = self.patch_embedding(x)
        # flat tokens to a grid: (B, 16, 16, embed_dim)
        x = x.view(x.shape[0], self.grid, self.grid, -1)
        x = self.stages(x)
        x = self.norm(x)
        x = x.mean(dim=(1, 2))
        return self.head(x)
