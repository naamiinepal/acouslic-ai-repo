import re
from typing import Any, Callable, Mapping, Optional

import torch
from monai.networks import one_hot
from monai.metrics.meandice import compute_dice
from monai.metrics.meaniou import compute_iou
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import nn
from torch.nn import functional as F    


_mapping_str_any = Mapping[str, Any]


class BaseModule(LightningModule):
    """Base LightningModule for Binary segmentation.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: nn.Module,
        segmentation_criterion: Callable[
            [torch.Tensor, torch.Tensor, bool], torch.Tensor
        ],
        classification_criterion: Callable[
            [torch.Tensor, torch.Tensor, bool], torch.Tensor
        ],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        scheduler_monitor: str = "val/loss",
        threshold: float = 0.5,
        multi_class: bool = True,
        log_output_masks: bool = True,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=["net", "segmentation_criterion", "classification_criterion"],
        )

        self.net = net

        # loss function
        self.segmentation_criterion = segmentation_criterion
        self.classification_criterion = classification_criterion

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def losses(self, pred_mask, pred_frame_logits, gt_mask, gt_frame_types):
        segmentation_loss = self.segmentation_criterion(pred_mask, gt_mask)

        frame_num_loss = self.classification_criterion(
            pred_frame_logits.view(-1, 3), gt_frame_types.view(-1)
        )

        total_loss = segmentation_loss + frame_num_loss

        return {
            "segmentation_loss": segmentation_loss,
            "frame_num_loss": frame_num_loss,
            "total_loss": total_loss,
        }

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        # self.val_acc_best.reset()
        pass

    def step(self, batch: _mapping_str_any) -> _mapping_str_any:
        image, mask, frame_type = batch["image"], batch["mask"], batch["frame_type"]

        pred_masks, pred_frame_logits, _ = self(
            x=image, frame_type=frame_type
        )
        
        losses = self.losses(
            pred_mask=pred_masks,
            pred_frame_logits=pred_frame_logits,
            gt_mask=mask,
            gt_frame_types=frame_type,
        )

        out = dict(
            **losses,
            pred_masks=pred_masks,
        )

        return out  

    def training_step(self, batch: _mapping_str_any, batch_idx: int):
        out = self.step(batch)
        images = batch['image']
        
        losses = dict(
            seg_loss=out['segmentation_loss'],
            frame_num_loss=out["frame_num_loss"],
            total_loss=out['total_loss']
        )

        self.log(
            "train/seg_loss",
            losses['seg_loss'],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.shape[0],
        )
        self.log(
            "train/frame_num_loss",
            losses['frame_num_loss'],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.shape[0],
        )
        self.log(
            "train/total_loss",
            losses['total_loss'],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.shape[0],
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return losses

    def on_train_epoch_end(self):
        pass

    def compile(self):
        assert re.match(
            r"2.", torch.__version__
        ), "Pytorch version >= 2.X is required to use compile() method."
        return torch.compile(self)

    def validation_step(self, batch: _mapping_str_any, batch_idx: int):
        out = self.step(batch)

        images, targets = batch['image'], batch['mask']

        losses = dict(
            seg_loss=out['segmentation_loss'],
            frame_num_loss=out["frame_num_loss"],
            total_loss=out['total_loss']
        )

        # Log images at the start of validation step
        if (
            batch_idx == 0
            and isinstance(self.logger, WandbLogger)
            and self.hparams.log_output_masks
        ):
            # Only Log 16 images at max
            max_images_logs = 16
            if len(targets) < max_images_logs:
                max_images_logs = len(targets)

            self.logger.log_image(
                key="val/image", images=list(images.float())[:max_images_logs]
            )
            self.logger.log_image(
                key="val/target_mask", images=list(targets.float())[:max_images_logs]
            )
            self.logger.log_image(
                key="val/pred_mask",
                images=list(((out['pred_masks'] > self.hparams.threshold) * 1).float())[
                    :max_images_logs
                ],
            )

        self.log(
            "val/seg_loss",
            losses['seg_loss'],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.shape[0],
        )
        self.log(
            "val/frame_num_loss",
            losses['frame_num_loss'],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.shape[0],
        )
        self.log(
            "val/total_loss",
            losses['total_loss'],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.shape[0],
        )

        return None

    def on_validation_epoch_end(self):
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # # otherwise metric would be reset by lightning after each epoch
        # self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True, batch_size=images.shape[0])
        pass

    def test_step(self, batch: _mapping_str_any, batch_idx: int):
        pass

    def on_test_epoch_end(self):
        pass

    def predict_step(self, batch: _mapping_str_any, batch_idx: int) -> Any:
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.scheduler_monitor,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
