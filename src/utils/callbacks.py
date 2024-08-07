import csv
from pathlib import Path
from typing import Any, Literal, Sequence
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from monai.data.image_writer import NibabelWriter
import numpy as np

class CSVLogger(BasePredictionWriter):
    """write into csv"""
    def __init__(self, filename, write_interval: Literal['batch'] | Literal['epoch'] | Literal['batch_and_epoch'] = "batch") -> None:
        super().__init__(write_interval)
        self.filestream_writer = csv.writer(open(Path(filename),"w"))
        header = ['filename_or_obj']
        self.filestream_writer.writerow(header)
    
    def get_filename(self, prediction:Any):
        "this will be column name identifier for each row"
        pass

    def write_on_batch_end(self, trainer: Trainer, pl_module: LightningModule, prediction: Any, batch_indices: Sequence[int] | None, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        filenames = batch['image'].meta['filename_or_obj']
        print(filenames)
        for fn in filenames:
            self.filestream_writer.writerow([fn,])

class NiftiPredictionWriter(BasePredictionWriter):
    """Save model prediction as nift
    Inherits from lightning callback for writing model prediction"""

    def __init__(self,
                 output_dir,
                 save_pred=True,
                 save_gt=True,
                 save_input=True,
                 write_interval: Literal['batch'] | Literal['epoch'] | Literal['batch_and_epoch'] = "batch") -> None:
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.save_pred = save_pred
        self.save_gt = save_gt
        self.save_input = save_input
        self.nifti_writer = NibabelWriter(output_dtype=np.int32)
    
    def write_on_batch_end(self, trainer: Trainer, pl_module: LightningModule, prediction: Any, batch_indices: Sequence[int] | None, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        batch_size = ...
        