from torch.nn.modules.loss import _Loss
from torch.nn import CrossEntropyLoss
from monai.losses import DiceCELoss
import torch


class SegmentationFrameSelectionLoss(_Loss):
    def __init__(
        self,
        lambda_ce: float = 0.25,
        sigmoid: bool = True,
        cls_weights: torch.Tensor = torch.tensor([0.2, 0.45, 0.35]),
        seg_lambda: float = 1,
        cls_lambda: float = 1,
    ):
        super().__init__()
        self.seg_lambda = seg_lambda
        self.cls_lambda = cls_lambda
        self.classification_loss = CrossEntropyLoss(weight=cls_weights)
        self.segmentation_loss = DiceCELoss(sigmoid=sigmoid, lambda_ce=lambda_ce)

    def forward(self, pred_mask: torch.Tensor, pred_frame_logits: torch.Tensor, gt_mask: torch.Tensor, gt_frame_types: torch.Tensor):
        B, C, H, W = pred_mask.shape

        segmentation_loss = self.segmentation_loss(
            pred_mask.view(-1, 1, H, W), gt_mask.view(-1, 1, H, W)
        )

        frame_num_loss = self.classification_loss(
            pred_frame_logits.view(-1, 3), gt_frame_types.view(-1).long()
        )

        total_loss = (
            self.seg_lambda * segmentation_loss
            + self.cls_lambda * frame_num_loss
        )

        return {
            "segmentation_loss": segmentation_loss,
            "frame_num_loss": frame_num_loss,
            "total_loss": total_loss,
        }
        