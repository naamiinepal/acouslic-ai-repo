import torch
from torch import nn

from typing import Sequence
import segmentation_models_pytorch as smp


class CustomUnet(smp.Unet):
    def __init__(
        self,
        classification_bias: torch.Tensor | None = None,
        segmentation_bias: torch.Tensor | None = None,
        **kwargs
    ):
        num_classes = 3 * kwargs["in_channels"]

        aux_params = kwargs.get("aux_params")
        if aux_params is not None:
            aux_params["classes"] = num_classes
        else:
            kwargs["aux_params"] = {"classes": num_classes}

        super().__init__(**kwargs)
        with torch.no_grad():
          if classification_bias is not None:
              self.classification_head[-2].bias = nn.Parameter(classification_bias)
            
          if segmentation_bias is not None:
              self.segmentation_head[0].bias = nn.Parameter(segmentation_bias)

    def get_pred_seg_num(self, pred_seg_logits: torch.Tensor):
        B = pred_seg_logits.size(0)

        # shape: (B, 140, 3)
        pred_seg_proc = pred_seg_logits.view(B, -1, 3)

        pred_seg_proc = torch.argmax(pred_seg_proc, dim=-1)

        curr_idx = 1
        seg_num = pred_seg_proc == curr_idx

        if seg_num.any():
            return torch.argmax(pred_seg_proc[..., curr_idx], -1, keepdim=True)

        curr_idx = 2
        seg_num = pred_seg_proc == curr_idx

        if seg_num.any():
            return torch.argmax(pred_seg_proc[..., curr_idx], -1, keepdim=True)

        curr_idx = 0
        return torch.argmin(pred_seg_logits[..., curr_idx], -1, keepdim=True)
