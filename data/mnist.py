from lightning.pytorch import LightningDataModule
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split


class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, val_size):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        MNIST(self.hparams.data_dir, train=True, download=True)
        MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: str):
        self.dataset = MNIST(self.data_dir,
                             train=True,
                             download=False,
                             transform=transforms.ToTensor())

        train_length = len(self.dataset)
        self.train_dataset, self.val_dataset = \
            random_split(self.dataset,
                         [train_length - self.hparams.val_size, self.hparams.val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=True)
