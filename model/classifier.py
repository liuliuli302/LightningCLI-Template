import torch
from lightning.pytorch import LightningModule
from model.layers.basic_linear_layer import BasicLinearModel
from torch.nn import functional as F


class Classifier(LightningModule):
    def __init__(self, in_features, out_features, hidden_dim, lr, T_max, eta_min):
        super().__init__()
        self.save_hyperparameters()
        self.model = BasicLinearModel(in_features, out_features, hidden_dim)
        self.validation_step_outputs = []
        self.validation_step_acc = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = F.cross_entropy(output, target.view(-1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = F.cross_entropy(output, target.view(-1))
        acc = torch.sum(torch.eq(torch.argmax(output, -1), target).to(torch.float32)) / len(target)

        self.validation_step_outputs.append(loss)
        self.validation_step_acc.append(acc)

        self.log('val/step/loss', loss)
        self.log('val/step/acc', acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.T_max,
                                                               eta_min=self.hparams.eta_min)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return optim_dict

    def on_validation_epoch_end(self):
        mean_loss = torch.stack(self.validation_step_outputs).mean()
        mean_acc = torch.stack(self.validation_step_acc).mean()

        self.log('val/epoch/loss', mean_loss, prog_bar=True)
        self.log('val/epoch/acc', mean_acc, prog_bar=True)

        self.validation_step_outputs.clear()
        self.validation_step_acc.clear()
