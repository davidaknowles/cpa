from typing import Optional

from scvi import settings
from scvi.data import AnnDataManager
from scvi.dataloaders import DataSplitter, AnnDataLoader
import torch


def accelerator_device_from_use_gpu(use_gpu):
    if use_gpu is False:
        return "cpu", 1, torch.device("cpu")
    if isinstance(use_gpu, int) and not isinstance(use_gpu, bool):
        return "gpu", [use_gpu], torch.device(f"cuda:{use_gpu}")
    if isinstance(use_gpu, str):
        return "gpu", use_gpu, torch.device("cuda:0")
    if use_gpu is True:
        return "gpu", 1, torch.device("cuda:0")
    if torch.cuda.is_available():
        return "gpu", 1, torch.device("cuda:0")
    return "cpu", 1, torch.device("cpu")


class AnnDataSplitter(DataSplitter):
    def __init__(
            self,
            adata_manager: AnnDataManager,
            train_indices,
            valid_indices,
            test_indices,
            use_gpu: bool = False,
            **kwargs,
    ):
        super().__init__(adata_manager)
        self.data_loader_kwargs = kwargs
        self.use_gpu = use_gpu
        self.train_idx = train_indices
        self.val_idx = valid_indices
        self.test_idx = test_indices

    def setup(self, stage: Optional[str] = None):
        accelerator, _, self.device = accelerator_device_from_use_gpu(self.use_gpu)
        pin_memory_gpu_training = getattr(settings, "dl_pin_memory_gpu_training", True)
        self.pin_memory = (
            True
            if (pin_memory_gpu_training and accelerator == "gpu")
            else False
        )

    def train_dataloader(self):
        if len(self.train_idx) > 0:
            return AnnDataLoader(
                self.adata_manager,
                indices=self.train_idx,
                shuffle=True,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def val_dataloader(self):
        if len(self.val_idx) > 0:
            data_loader_kwargs = self.data_loader_kwargs.copy()
            # if len(self.valid_indices < 4096):
            #     data_loader_kwargs.update({'batch_size': len(self.valid_indices)})
            # else:
            #     data_loader_kwargs.update({'batch_size': 2048})
            return AnnDataLoader(
                self.adata_manager,
                indices=self.val_idx,
                shuffle=True,
                pin_memory=self.pin_memory,
                **data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        if len(self.test_idx) > 0:
            return AnnDataLoader(
                self.adata_manager,
                indices=self.test_idx,
                shuffle=True,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass
