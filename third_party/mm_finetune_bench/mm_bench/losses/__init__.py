from .infonce import InfoNCELoss
from .siglip_loss import SigLIPLoss

__all__ = ["InfoNCELoss", "SigLIPLoss", "build_loss"]


def build_loss(loss_type: str, **kwargs):
    if loss_type == "infonce":
        return InfoNCELoss(**kwargs)
    if loss_type == "siglip":
        return SigLIPLoss(**kwargs)
    raise ValueError(f"Unknown loss type: {loss_type}")
