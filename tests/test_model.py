"""
Tests for the ViT, hybrid, and Swin models.

Verifies output and token shapes, exact parameter counts and that the plain
ViT and the Swin are each within 10% of the hybrid's size, that invalid
inputs and configurations raise ``ValueError``, and the Swin's windowing:
the window partition round trip and the pattern of shifted blocks.
"""

import pytest
import torch
from src.models import (ViT, ViTConfig, PatchEmbedding, ResNetTokenizer,
                        MultiHeadSelfAttention, Swin, SwinConfig, SwinBlock,
                        window_partition, window_reverse)


def _count_params(model):
    """
    Return the number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("tokenizer", ["patch", "resnet"])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_output_shape(tokenizer, batch_size):
    """
    Check that both models map a batch of images to (batch_size, 100) logits.

    Batch size 1 catches reshapes that drop or assume the batch dimension.
    A shallow model (depth 2) is used, since depth does not affect the shape.
    """
    model = ViT(ViTConfig(tokenizer=tokenizer, depth=2))
    model.eval()
    x = torch.randn(batch_size, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    shape_message = f"Expected shape {(batch_size, 100)} got {out.shape}"
    assert out.shape == (batch_size, 100), shape_message


@pytest.mark.parametrize("tokenizer", ["patch", "resnet"])
def test_token_shape(tokenizer):
    """
    Check that both tokenizers produce the same 8x8 grid of d_model tokens.

    The patch embedding and the ResNet tokenizer must both output
    (batch_size, 64, d_model), so the two models differ in how tokens are
    formed, not in the sequence the encoder receives.
    """
    batch_size = 2
    config = ViTConfig(tokenizer=tokenizer, depth=1)
    model = ViT(config)
    tok = model.tokenizer
    x = torch.randn(batch_size, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        tokens = tok(x)
        assert tok.num_patches == 64
        assert tokens.shape == (batch_size, tok.num_patches, config.d_model)


@pytest.mark.parametrize("tokenizer, depth, expected",
                         [("patch", 9, 4_799_076), ("resnet", 7, 4_604_084)])
def test_parameter_count(tokenizer, depth, expected):
    """
    Check the exact trainable parameter count of each model as trained.

    Each transformer block has 527,104 parameters. The plain ViT (depth 9)
    adds the patch embedding (12,544), position embedding (16,384), final
    norm (512), and head (25,700). The hybrid (depth 7) replaces the patch
    embedding with the ResNet-56 tokenizer (871,760: backbone 855,120 plus a
    16,640-parameter projection). Catches sizing mistakes that shape tests
    cannot, such as a wrong MLP ratio or a missing bias.
    """
    model = ViT(ViTConfig(tokenizer=tokenizer, depth=depth))
    count = _count_params(model)
    count_message = f"Expected {expected} for {tokenizer} got {count}"
    assert count == expected, count_message


@pytest.mark.parametrize("make_model", [
    pytest.param(lambda: ViT(ViTConfig(tokenizer="patch", depth=9)), id="vit"),
    pytest.param(lambda: Swin(SwinConfig()), id="swin")
])
def test_models_similar_size(make_model):
    """
    Check that a comparison model is within 10% of the hybrid's parameter count.

    The assignment requires each comparison model's trainable parameter
    count to be within 10% of the primary (hybrid) model's, so accuracy
    differences reflect architecture rather than model capacity. Uses the
    configurations the models are trained with (hybrid depth 7, plain ViT
    depth 9, Swin defaults).
    """
    hybrid_count = _count_params(ViT(ViTConfig(tokenizer="resnet", depth=7)))
    count = _count_params(make_model())
    diff = abs(hybrid_count - count) / hybrid_count
    similar_message = f"Model sizes not close, differ by {diff:.1%}"
    assert diff < 0.10, similar_message


@pytest.mark.parametrize("make_invalid", [
    pytest.param(lambda: PatchEmbedding(192, 32, 5, 3), id="patch_size"),
    pytest.param(lambda: MultiHeadSelfAttention(192, 5), id="n_heads"),
    pytest.param(lambda: ViT(ViTConfig(tokenizer="diffusion")), id="tokenizer"),
    pytest.param(lambda: ResNetTokenizer(192, 30, 3), id="resnet_img_size"),
    pytest.param(lambda: ResNetTokenizer(192, 32, 3, 0), id="blocks_per_stage")
])
def test_invalid_inputs(make_invalid):
    """
    Check that invalid inputs and configurations raise ValueError.

    Covers a patch size that does not divide the image size, a head count
    that does not divide d_model, an unknown tokenizer name, an image size
    the ResNet tokenizer cannot downsample evenly, and a ResNet tokenizer
    with fewer than one block per stage.
    """
    with pytest.raises(ValueError):
        make_invalid()


@pytest.mark.parametrize("batch_size", [1, 4])
def test_swin_shape(batch_size):
    """
    Check that the Swin maps a batch of images to (batch_size, 100) logits.

    Batch size 1 matters especially for Swin: ``window_reverse`` and the
    shift mask's per-image copy both infer the batch size from the number
    of windows.
    """
    model = Swin(SwinConfig())
    model.eval()
    x = torch.randn(batch_size, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    shape_message = f"Expected shape {(batch_size, 100)} got {out.shape}"
    assert out.shape == (batch_size, 100), shape_message


def test_swin_param_count():
    """
    Check the exact trainable parameter count of the Swin as trained.

    With the default config (depth (2, 4, 4)): patch embedding (832), two
    stage-1 blocks (50,082 each), merge to 128 channels (33,280), four
    stage-2 blocks (198,468 each), merge to 256 channels (132,096), four
    stage-3 blocks (790,152 each), final norm (512), and head (25,700).
    Each block includes its relative position bias table.
    """
    count = _count_params(Swin(SwinConfig()))
    assert count == 4_247_064, f"Expected 4,247,064, got {count}"


def test_shift_pattern():
    """
    Check which Swin blocks use shifted windows, and by how much.

    Stages 1 and 2 alternate regular (0) and shifted (window_size // 2 = 2)
    windows. Stage 3's 4x4 grid is a single window, so none of its blocks
    shift.
    """
    model = Swin(SwinConfig())
    truth = [0, 2, 0, 2, 0, 2, 0, 0, 0, 0]
    shifts = [b.shift for b in model.stages if isinstance(b, SwinBlock)]
    assert shifts == truth, f"Expected shifts {truth} got {shifts}"


def test_window_round_trip():
    """
    Check that window_reverse exactly undoes window_partition.

    Uses ``torch.equal`` rather than a tolerance, since partitioning only
    moves values. The shift mask and the window attention depend on this
    window order.
    """
    x = torch.randn(2, 16, 16, 8)
    out = window_reverse(window_partition(x, 4), 4, 16, 16)
    assert torch.equal(out, x), "window_reverse did not restore the input"
