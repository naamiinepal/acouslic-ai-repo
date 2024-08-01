from torch.utils.data import Dataset
from monai.transforms import LoadImaged, Compose, ToTensord, Resized
from monai.config.type_definitions import PathLike
from typing import Any, Sequence, Callable
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.nn import functional as TF


class AcouslicDataset(Dataset):
    def __init__(
        self,
        root_dir: PathLike,
        csv_file: PathLike,
        transform: Sequence[Callable] | Callable | None = None,
    ):
        self.root_dir = Path(root_dir)

        self.image_mask_arr = pd.read_csv(csv_file, usecols=[1, 2]).to_numpy()

        if transform is None:
            self.transform = Compose(
                [LoadImaged(keys=["image", "mask"]), ToTensord(keys=["image", "mask"])]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_mask_arr)

    def __getitem__(self, idx: int | slice | Sequence[int]):
        image_mask_dict = {
            "image": self.image_mask_arr[idx, 0],
            "mask": self.image_mask_arr[idx, 1],
        }

        image_mask_dict = self.transform(image_mask_dict)
        mask = image_mask_dict["mask"]

        image_mask_dict["frame_type"] = torch.amax(mask, (1, 2))
        
        return image_mask_dict
