import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedBCELoss(nn.Module):
    """
    Custom Loss for U-Net Cloud Masking.
    Computes Binary Cross Entropy strictly for the center pixel (16, 16)
    using the point-based SYNOP label, ignoring the rest of the patch.
    """
    def __init__(self, ignore_index: int = -1, weight: torch.Tensor = None):
        super(MaskedBCELoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: [Batch, 2, H, W] (raw logits from U-Net)
            targets: [Batch, H, W] (Spatial labels where only (16,16) is valid, rest is -1)
        """
        # Flatten spatial dimensions
        preds_flat = preds.view(preds.size(0), preds.size(1), -1) # [B, 2, 33*33]
        targets_flat = targets.view(targets.size(0), -1)          # [B, 33*33]
        
        # Create mask for valid pixels (not ignore_index)
        valid_mask = (targets_flat != self.ignore_index)
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=preds.device, requires_grad=True)
            
        # Select valid predictions and targets
        preds_valid = preds_flat.transpose(1, 2)[valid_mask] # [N_valid, 2]
        targets_valid = targets_flat[valid_mask]             # [N_valid]
        
        # CrossEntropyLoss for 2 classes is equivalent to BCELoss on softmaxed logits.
        return F.cross_entropy(preds_valid, targets_valid, weight=self.weight)
