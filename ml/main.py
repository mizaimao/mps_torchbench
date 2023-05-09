""""""
import torch
import torchvision
from torchvision import datasets, transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from resnet50 import ResNet50Model

BATCH_SIZE: int = 128
N_WORKERS: int = 0
LR: float = 3e-4
EPOCHS: int = 10


train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=N_WORKERS,
)

#val_size: int = int(len(train_val_loader) * 0.2)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        "../data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=N_WORKERS,
)


model = ResNet50Model(
    weights=False, in_channels=3, num_classes=10, lr=LR, freeze=False
)

trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=EPOCHS,
    callbacks=[
        EarlyStopping(
            monitor="train_loss",
            mode="min",
            patience=20,
        )
    ],
)
model.hparams.lr = LR


trainer.fit(model, train_loader)
metrics = trainer.logged_metrics
trainer.test(model, test_loader)

fold = 0
logs = {}

logs[f"fold{fold}"] = {
    "train_loss": metrics["train_loss_epoch"].item(),
    "val_loss": metrics["val_loss"].item(),
    "train_acc": metrics["train_acc_epoch"].item(),
    "val_acc": metrics["val_acc"].item(),
}

print(
    f"Train Loss: {logs[f'fold{fold}']['train_loss']} | Train Accuracy: {logs[f'fold{fold}']['train_acc']}"
)
print(
    f"Val Loss: {logs[f'fold{fold}']['val_loss']} | Val Accuracy: {logs[f'fold{fold}']['val_acc']}"
)
