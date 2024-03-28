import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset
from lightning.pytorch import LightningDataModule


class BoringDataModule(LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        self.random_full = RandomDataset(32, 64 * 4)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.random_train = Subset(self.random_full, indices=range(64))

        if stage in ("fit", "validate"):
            self.random_val = Subset(self.random_full, indices=range(64, 64 * 2))

        if stage == "test":
            self.random_test = Subset(self.random_full, indices=range(64 * 2, 64 * 3))

        if stage == "predict":
            self.random_predict = Subset(self.random_full, indices=range(64 * 3, 64 * 4))


class RandomDataset(Dataset):
    def __init__(self, size: int, length: int):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index: int) -> Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return self.len
