import torch
from torch import nn
from torch.nn import functional as F

from typing import Sequence
import segmentation_models_pytorch as smp

class CustomUnet(smp.Unet):
    def __init__(self, decoder_upsampling_mode: str = "nearest-exact", **kwargs):
        aux_params = kwargs.get("aux_params")

        num_classes = 3 * kwargs["in_channels"]

        if aux_params is not None:
          aux_params["classes"] = num_classes
        else:
          kwargs["aux_params"] =  {"classes": num_classes}

        super().__init__(**kwargs)

        self.seg_num_normalizer = 2 / (kwargs["in_channels"] - 1)

        self.seg_num_projector = nn.Linear(1, kwargs["decoder_channels"][0], bias=False)

        self.decoder_upsampling_mode = decoder_upsampling_mode

    def decoder_block_forward(self, x: torch.Tensor, block_idx: int, skips: Sequence[torch.Tensor], seg_proj: torch.Tensor | None = None) -> torch.Tensor:
        skip = skips[block_idx] if block_idx < len(skips) else None

        decoder_block = self.decoder.blocks[block_idx]

        x = F.interpolate(x, scale_factor=2, mode=self.decoder_upsampling_mode)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = decoder_block.attention1(x)

        x = decoder_block.conv1(x)

        if seg_proj is not None:
            x += seg_proj[..., None, None]

        x = decoder_block.conv2(x)
        x = decoder_block.attention2(x)
        return x

    def decoder_forward(self, features, seg_num: torch.Tensor) -> torch.Tensor:
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.decoder.center(head)

        # normalize seg num in [-1, 1] range for stability
        norm_seg_num = seg_num * self.seg_num_normalizer - 1

        seg_proj = self.seg_num_projector(norm_seg_num)

        x = self.decoder_block_forward(x, block_idx=0, skips=skips, seg_proj=seg_proj)

        for i in range(1, len(self.decoder.blocks)):
            x = self.decoder_block_forward(x, block_idx=i, skips=skips)

        return x

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

    @torch.no_grad()
    def get_new_features(self, frame_type: torch.Tensor, features: Sequence[torch.Tensor]):
        new_features = [[] for _ in range(len(features))]
        meta_dict = {}
        for batch_idx, batch_frame in enumerate(frame_type.bool()):
            if batch_frame.any():
                batch_frame_count = batch_frame.sum()

                meta_dict[batch_idx] = batch_frame_count

                for i, feat in enumerate(features):
                    curr_feat = feat[batch_idx]

                    new_features[i].append(curr_feat.expand(batch_frame_count, *curr_feat.shape))

        return tuple(map(torch.concat, new_features)), meta_dict

    def forward(self, x, frame_type: torch.Tensor | None = None):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)

        # shape: (B, 3 * in_channels)
        pred_frame_logits = self.classification_head(features[-1])

        if frame_type is None:
            seg_num = self.get_pred_seg_num(pred_frame_logits)
            meta_dict = None
        else:
            seg_num = torch.nonzero(frame_type)[:, 1].view(-1, 1)
            features, meta_dict = self.get_new_features(frame_type, features)

        decoder_output = self.decoder_forward(features, seg_num=seg_num)

        masks = self.segmentation_head(decoder_output)

        return masks, pred_frame_logits, meta_dict