"""
Build the plain ViT, hybrid, or Swin model from a run's settings.
"""
from src.models import ViT, ViTConfig
from src.swin import Swin, SwinConfig


def build_model(settings):
    """
    Build the model described by a run's settings.

    Shared by training (``vars(args)``) and evaluation (a checkpoint's
    ``run_info``), so a checkpoint is always rebuilt with the architecture it
    was trained with. Settings without an ``"arch"`` key, from runs saved
    before Swin was added, build a ViT.

    Args:
        settings (dict): Run settings. Uses ``arch`` (``"vit"`` or ``"swin"``,
            default ``"vit"``) and ``drop_path_rate``; for the ViT, also
            ``tokenizer`` and ``depth``.

    Returns:
        nn.Module: A ``ViT`` (plain or hybrid, by ``tokenizer``) or a ``Swin``
        with default ``SwinConfig`` settings, on the CPU.
    """
    if settings.get("arch", "vit") == "swin":
        return Swin(SwinConfig(drop_path_rate=settings["drop_path_rate"]))
    return ViT(ViTConfig(tokenizer=settings["tokenizer"], depth=settings["depth"],
                         drop_path_rate=settings["drop_path_rate"]))