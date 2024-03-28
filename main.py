from lightning.pytorch.cli import LightningCLI
from model.classifier import Classifier
from data.mnist import MNISTDataModule
import torch

torch.set_float32_matmul_precision('high')


def cli_main():
    cli = LightningCLI(Classifier, MNISTDataModule)


if __name__ == "__main__":
    cli_main()
