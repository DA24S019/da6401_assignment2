# cnn_model.py

import torch
import torch.nn as nn
import pytorch_lightning as pl

class CNN(pl.LightningModule):
    def __init__(self,
                 input_channels,
                 conv_filters,
                 kernel_sizes,
                 activation,
                 dense_neurons,
                 num_classes,
                 lr,
                 batch_norm=False,
                 dropout=0.0):
        super().__init__()
        self.save_hyperparameters()

        self.activation_fn = self._get_activation_fn(activation)

        # Convolutional layers
        layers = []
        in_channels = input_channels
        for out_channels, ksize in zip(conv_filters, kernel_sizes):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=ksize // 2))
            
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))

            layers.append(self._get_activation_fn(activation))
            layers.append(nn.MaxPool2d(2, 2))

            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))

            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*layers)

        # Estimate flatten dim with dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 128, 128)
            dummy_output = self.conv_blocks(dummy_input)
            flatten_dim = dummy_output.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flatten_dim, dense_neurons),
            self.activation_fn,
            nn.Linear(dense_neurons, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

    def _get_activation_fn(self, name):
        name = name.lower()
        if name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        elif name == 'silu':
            return nn.SiLU()
        elif name == 'mish':
            return nn.Mish()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
