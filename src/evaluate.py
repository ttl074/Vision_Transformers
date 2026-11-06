def load_model(ckpt_path, device):
    """
    Rebuild a trained model from a best.pt checkpoint, ready for inference.

    The architecture (plain ViT, hybrid, or Swin) comes from the run settings
    saved in the checkpoint, so the rebuilt model matches the one that was
    trained. Checkpoints saved before Swin was added rebuild as ViTs.

    Args:
        ckpt_path (str): Path to best.pt.
        device (torch.device): Device to load the model onto.

    Returns:
        tuple: (model, epoch, val_loss):
            - model (nn.Module): The trained model, in eval mode.
            - epoch (int): Epoch at which the checkpoint was saved.
            - val_loss (float): Validation loss at that epoch.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt["run_info"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["epoch"], ckpt["val_loss"]
