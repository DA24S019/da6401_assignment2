import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

class ResNetFinetune(pl.LightningModule):
    def __init__(self, num_classes, lr, optimizer='adam', weight_decay=0.0, momentum=0.9,
                 scheduler=False, freeze_type='none', freeze_upto_layer=0, resnet_variant='resnet18'):
        super().__init__()
        self.save_hyperparameters()

        # Load specified ResNet variant
        self.backbone = getattr(models, resnet_variant)(pretrained=True)

        # Freeze strategy
        if freeze_type == 'all':
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif freeze_type == 'upto':
            child_counter = 0
            for child in self.backbone.children():
                child_counter += 1
                for param in child.parameters():
                    param.requires_grad = child_counter > freeze_upto_layer

        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_acc_metric = MulticlassAccuracy(num_classes=num_classes)
        self.train_prec_metric = MulticlassPrecision(num_classes=num_classes, average='macro')
        self.train_rec_metric = MulticlassRecall(num_classes=num_classes, average='macro')
        self.train_f1_metric = MulticlassF1Score(num_classes=num_classes, average='macro')

        self.val_acc_metric = MulticlassAccuracy(num_classes=num_classes)
        self.val_prec_metric = MulticlassPrecision(num_classes=num_classes, average='macro')
        self.val_rec_metric = MulticlassRecall(num_classes=num_classes, average='macro')
        self.val_f1_metric = MulticlassF1Score(num_classes=num_classes, average='macro')

        self.test_acc_metric = MulticlassAccuracy(num_classes=num_classes)
        self.test_prec_metric = MulticlassPrecision(num_classes=num_classes, average='macro')
        self.test_rec_metric = MulticlassRecall(num_classes=num_classes, average='macro')
        self.test_f1_metric = MulticlassF1Score(num_classes=num_classes, average='macro')

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)

        loss = self.loss_fn(logits, y)
        acc = self.train_acc_metric(preds, y)
        precision = self.train_prec_metric(preds, y)
        recall = self.train_rec_metric(preds, y)
        f1 = self.train_f1_metric(preds, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        self.log("train_precision", precision, on_epoch=True)
        self.log("train_recall", recall, on_epoch=True)
        self.log("train_f1", f1, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)

        loss = self.loss_fn(logits, y)
        acc = self.val_acc_metric(preds, y)
        precision = self.val_prec_metric(preds, y)
        recall = self.val_rec_metric(preds, y)
        f1 = self.val_f1_metric(preds, y)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        self.log("val_precision", precision, on_epoch=True)
        self.log("val_recall", recall, on_epoch=True)
        self.log("val_f1", f1, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)

        loss = self.loss_fn(logits, y)
        acc = self.test_acc_metric(preds, y)
        precision = self.test_prec_metric(preds, y)
        recall = self.test_rec_metric(preds, y)
        f1 = self.test_f1_metric(preds, y)

        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_epoch=True)
        self.log("test_precision", precision, on_epoch=True)
        self.log("test_recall", recall, on_epoch=True)
        self.log("test_f1", f1, on_epoch=True)

    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'sgd':
            opt = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay
            )

        if self.hparams.scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "val_loss"
                }
            }

        return opt
