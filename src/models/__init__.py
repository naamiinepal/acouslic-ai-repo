import torch

torch.set_float32_matmul_precision("medium")

from .base_module import BaseModule