from collections.abc import Iterable
from typing import Any

from torch.utils.data import default_collate


def selective_data_collator_fn(features: list[dict[str, Any]]):

    selected_features = [feat for feat in features if feat['mask'] is not None]

    return default_collate(selected_features)