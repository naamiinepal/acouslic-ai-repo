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
        criterion: Callable[
            [torch.Tensor, torch.Tensor, bool], torch.Tensor
        ],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        scheduler_monitor: str = "val/loss",
        threshold: float = 0.5,
        multi_class: bool = True,
        log_output_masks: bool = True,
        segmentation_lambda: float = 1,
        classification_lambda: float = 1,
        lr_scheduler_config:  _mapping_str_any | None = None,
        weight_decay: float = 0.001
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=["net", "criterion"],
        )

        self.net = net

        # loss function
        self.criterion = criterion

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)


    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        # self.val_acc_best.reset()
        pass

    def step(self, batch: _mapping_str_any) -> _mapping_str_any:
        image, mask, frame_type = batch["image"], batch["mask"], batch["frame_type"]

        pred_masks, pred_frame_logits = self(image)

        losses = self.criterion(
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
        images = batch["image"]

        losses = dict(
            seg_loss=out["segmentation_loss"],
            frame_num_loss=out["frame_num_loss"],
            total_loss=out["total_loss"],
        )

        self.log(
            "train/seg_loss",
            losses["seg_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.shape[0],
        )
        self.log(
            "train/frame_num_loss",
            losses["frame_num_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.shape[0],
        )
        self.log(
            "train/total_loss",
            losses["total_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.shape[0],
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return losses['total_loss']

    def on_train_epoch_end(self):
        pass

    def compile(self):
        assert re.match(
            r"2.", torch.__version__
        ), "Pytorch version >= 2.X is required to use compile() method."
        return torch.compile(self)

    def validation_step(self, batch: _mapping_str_any, batch_idx: int):
        out = self.step(batch)

        images, targets = batch["image"], batch["mask"]

        losses = dict(
            seg_loss=out["segmentation_loss"],
            frame_num_loss=out["frame_num_loss"],
            total_loss=out["total_loss"],
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
                images=list(((out["pred_masks"] > self.hparams.threshold) * 1).float())[
                    :max_images_logs
                ],
            )

        self.log(
            "val/seg_loss",
            losses["seg_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.shape[0],
        )
        self.log(
            "val/frame_num_loss",
            losses["frame_num_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.shape[0],
        )
        self.log(
            "val/total_loss",
            losses["total_loss"],
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

    def get_optim_groups(self):
        if self.hparams.weight_decay <= 0:  # type: ignore
            return self.parameters()

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.modules.conv._ConvNd)
        blacklist_weight_modules = (
            nn.Embedding,
            nn.GroupNorm,
            nn.LayerNorm,
            nn.modules.batchnorm._NormBase,  # Base class for batchnorm and instance norm
        )
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name
                if pn.endswith("proj_weight"):
                    # Add project weights to decay set
                    decay.add(fpn)
                elif pn.endswith("weight"):
                    # random note: because named_modules and named_parameters are recursive
                    # we will see the same tensors p many many times. but doing it this way
                    # allows us to know which parent module any tensor p belongs to...
                    if isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
                else:
                    # all paramters except weight will not be decayed
                    no_decay.add(fpn)

        inter_params = decay & no_decay
        if len(inter_params) != 0:
            msg = f"parameters {inter_params} made it into both decay/no_decay sets!"
            raise ValueError(msg)

        # validate that we considered every parameter
        param_dict = dict(self.named_parameters())

        extra_params = param_dict.keys() - (decay | no_decay)
        if len(extra_params) != 0:
            msg = f"parameters {extra_params} were not separated into either decay/no_decay set!"
            raise ValueError(msg)

        # create the pytorch optimizer parameters
        return [
            {
                "params": [param_dict[pn] for pn in decay],
                "weight_decay": self.hparams.weight_decay,  # type: ignore
            },
            {
                "params": [param_dict[pn] for pn in no_decay],
                "weight_decay": 0.0,
            },
        ]

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.

        Examples
        --------
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.

        """
        optim_groups = self.get_optim_groups()
        optimizer = self.hparams.optimizer(optim_groups)  # type: ignore

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.scheduler_monitor,
                    "interval": "epoch",
                    "frequency": 1,
                    **(self.hparams.lr_scheduler_config or {}),  # type: ignore
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
