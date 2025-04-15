import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl



class CNN(pl.LightningModule):
    def __init__(self,
                 input_channels=3,
                 conv_filters=[32, 64, 128, 256, 512],
                 kernel_sizes=[3, 3, 3, 3, 3],
                 activation='relu',
                 dense_neurons=512,
                 num_classes=10,
                 lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Dynamically set activation
        self.activation_fn = self._get_activation_fn(activation)

        # Build convolutional layers
        layers = []
        in_channels = input_channels
        for out_channels, kernel_size in zip(conv_filters, kernel_sizes):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(self.activation_fn)
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*layers)

        # Infer flattened size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 512, 512)  # assuming 224x224 image
            dummy_output = self.conv_blocks(dummy_input)
            flatten_dim = dummy_output.view(1, -1).shape[1]

        # Dense classifier
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
        x = x.view(x.size(0), -1)  # flatten
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
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
