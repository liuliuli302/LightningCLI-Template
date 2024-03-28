import torch
from torch import nn
from lightning.pytorch import LightningModule
from model.layers.basic_linear_layer import BasicLinearModel


class Classifier(LightningModule):
    def __init__(self, in_features, out_features, hidden_dim, lr, T_max):
        super().__init__()
        self.save_hyperparameters()
        self.model = BasicLinearModel(in_features, out_features, hidden_dim)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = self.criterion(output, target.view(-1))
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = self.criterion(output, target.view(-1))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.T_max)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return optim_dict