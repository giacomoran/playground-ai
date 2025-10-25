import torch
from datetime import datetime
import wandb
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

project_root = Path(__file__).resolve().parent.parent

#: configs

base_batch_size = 64
base_epochs = 20
base_learning_rate = 4e-3

batch_size = 512
epochs = base_epochs
learning_rate = base_learning_rate * (batch_size / base_batch_size)

device = "mps"

log_interval = 100
wandb_project = project_root.name + "/000"
wandb_run_name = datetime.now().isoformat()

# Print scaling info
print(f"Batch size: {batch_size}")
print(f"Epochs: {epochs} (base: {base_epochs} at batch size {base_batch_size})")
print(
    f"Scaled learning rate: {learning_rate:.6f} (base: {base_learning_rate} at batch size {base_batch_size})"
)
print()

#: dataset

data_dir = project_root / "data"
data_dir.mkdir(parents=True, exist_ok=True)

training_data = datasets.FashionMNIST(
    root=str(data_dir),
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root=str(data_dir),
    train=False,
    download=True,
    transform=ToTensor(),
)

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

#: model


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

#: train

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

wandb_dir = project_root / "wandb"
wandb_dir.mkdir(parents=True, exist_ok=True)

wandb_run = wandb.init(
    project=project_root.name,
    name=wandb_run_name,
    config={
        "architecture": "NeuralNetwork_2x512",
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "optimizer": "SGD",
        "device": str(device),
    },
    dir=str(wandb_dir),
)

if wandb_run is not None:
    wandb.watch(model, log="all", log_freq=log_interval)


def train(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    running_loss = 0.0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        loss_value = loss.item()
        running_loss += loss_value

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % log_interval == 0:
            current = min((batch + 1) * dataloader.batch_size, size)
            mps_memory = (
                torch.mps.current_allocated_memory() / 1024 / 1024
            )  # Convert to MB
            print(
                f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]  MPS Memory: {mps_memory:>7.2f} MB"
            )
            if wandb.run is not None:
                wandb.log(
                    {
                        "train/batch_loss": loss_value,
                        "train/mps_memory_mb": mps_memory,
                        "epoch": epoch + 1,
                    },
                    step=epoch * num_batches + batch,
                )

    return running_loss / num_batches


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    return test_loss, correct


try:
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        avg_train_loss = train(train_dataloader, model, loss_fn, optimizer, epoch=t)
        avg_test_loss, test_accuracy = test(test_dataloader, model, loss_fn)

        if wandb.run is not None:
            wandb.log(
                {
                    "epoch": t + 1,
                    "train/epoch_loss": avg_train_loss,
                    "eval/loss": avg_test_loss,
                    "eval/accuracy": test_accuracy,
                    "train/mps_memory_mb": torch.mps.current_allocated_memory()
                    / 1024
                    / 1024,
                },
                step=(t + 1) * len(train_dataloader),
            )
finally:
    if wandb.run is not None:
        wandb.finish()

print("Done!")
