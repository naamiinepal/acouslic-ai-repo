import numpy as np
import pytest
import torch

import pyrootutils
import omegaconf
import hydra
import pytest_check as check

class TestDataModule:
    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "data" / "acouslic_sample.yaml")
    acouslic_datamodule = hydra.utils.instantiate(cfg)

    def test_metadata(self):
        sample_data = self.acouslic_datamodule.data_val[0]
        assert False, sample_data['image'].meta.keys()
        assert False, sample_batch

    def test_datatype(self):
        # test trainloader
        trainloader = self.acouslic_datamodule.train_dataloader()
        sample_batch = next(iter(trainloader))
        
        
        # existence test
        assert "image" in sample_batch.keys()
        assert "mask" in sample_batch.keys()
        assert "frame_type" in sample_batch.keys()

        # data type test
        images = sample_batch['image']
        masks = sample_batch['mask']
        frame_types = sample_batch['frame_type']
        check.equal(images.dtype, torch.float32)
        check.equal(masks.dtype,torch.int64)
        
        # size test
        B,C,H,W = images.shape
        check.equal((C,H,W),(140,544,736))

        B,C,H,W = masks.shape
        check.equal((C,H,W),(140,544,736))

        B,C = frame_types.shape
        check.equal(C,140)

        check.equal(torch.unique(masks),torch.from_numpy(np.array([0,1,2])))

        check.is_greater_than(torch.min(images[0]),-3.0)
        check.is_less_than(torch.max(images[0]),3.0)
