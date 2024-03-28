from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule
from data.mnist import MNISTDataModule
from model.classifier import Classifier
import torch

torch.set_float32_matmul_precision('high')


def cli_main():
    cli = LightningCLI(DemoModel, BoringDataModule)


if __name__ == "__main__":
    cli_main()
