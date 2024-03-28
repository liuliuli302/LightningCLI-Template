from typing import Any
import torch
from torch import Tensor
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT


class DemoModel(LightningModule):
    def __init__(self, out_dim: int = 10, learning_rate: float = 0.02):
        super().__init__()
        self.l1 = torch.nn.Linear(32, out_dim)
        self.learning_rate = learning_rate

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch: Any, batch_nb: int) -> STEP_OUTPUT:
        x = batch
        x = self(x)
        return x.sum()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
