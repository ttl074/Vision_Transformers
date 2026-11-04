"""
Tests for the ViT and hybrid models.

Verifies output and token shapes for both tokenizers, exact parameter counts
and that the two models are matched in size, and that invalid inputs and
configurations raise ``ValueError``.
"""

import pytest
import torch
from src.models import ViT, ViTConfig, PatchEmbedding, ResNetTokenizer, MultiHeadSelfAttention


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


def test_models_similar_size():
    """
    Check that the plain ViT is within 10% of the hybrid's parameter count.

    The assignment requires the baseline's trainable parameter count to be
    within 10% of the primary (hybrid) model's, so accuracy differences
    reflect architecture rather than model capacity. Uses the depths the two
    models are trained with (ViT 9, hybrid 7).
    """
    model_vit = ViT(ViTConfig(tokenizer="patch", depth=9))
    model_hybrid = ViT(ViTConfig(tokenizer="resnet", depth=7))
    vit_count = _count_params(model_vit)
    hybrid_count = _count_params(model_hybrid)
    diff = abs(hybrid_count - vit_count) / hybrid_count
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
