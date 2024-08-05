from monai.data import PersistentDataset
from monai.transforms import LoadImaged, Compose, ToTensord, Resized
from monai.config.type_definitions import PathLike
from typing import Any, Sequence, Callable
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.nn import functional as TF
import shutil


class AcouslicPersistantDataset(PersistentDataset):
    def __init__(
        self,
        csv_file: PathLike,
        cache_dir: Path | str,
        transform: Sequence[Callable] | Callable | None = None,
        delete_cache: bool = False,
        **kwargs
    ):
        image_mask_df = pd.read_csv(csv_file, usecols=['image_path', 'mask_path'])

        data = tuple({"image": image, "mask": mask} for image, mask in zip(image_mask_df['image_path'], image_mask_df['mask_path'], strict=True))

        if transform is None:
            transform = Compose(
                [LoadImaged(keys=["image", "mask"]), ToTensord(keys=["image", "mask"])]
            )

        if delete_cache:
            print(f"Removing pre-existing cache at {cache_dir}")
            shutil.rmtree(cache_dir, ignore_errors=True)

        super().__init__(data=data, transform=transform, cache_dir=cache_dir, **kwargs)

    def __getitem__(self, idx: int | slice | Sequence[int]):
        image_mask_dict = super().__getitem__(idx)

        image_mask_dict["frame_type"] = torch.amax(image_mask_dict["mask"], (1, 2))

        return image_mask_dict
