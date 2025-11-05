import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import wandb

#:

config = dict(
    batch_size=256,
    num_epochs=100,
    temperature=0.5,
    learning_rate=1e-4,
    weight_decay=1e-6,
    dataset="CIFAR-10",
    base_encoder="ResNet50",
    optimizer="AdamW",
)

#: Utils


def get_random_color_distortion(s=1.0):  # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


#: Model


# REFS: https://gist.github.com/ShairozS/a945c43bb81457b94e6c16cefbc0a858
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5, device="cuda"):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer(
            "negatives_mask",
            (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float(),
        )
        self.device = device

    def forward(self, z1, z2):
        B, C = z1.shape
        assert B == self.batch_size

        # According to Claude normalizing before cosine_similarity is more efficient
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # This differs from the pseudocode in the paper, they intersperse the
        # two tensors so that positive pairs are close to each other.
        z = torch.cat([z1, z2], dim=0)  # 2B, C

        S = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # 2B, 2B

        positives_upper = torch.diag(S, diagonal=B)  # B
        positives_lower = torch.diag(S, diagonal=-B)  # B
        positives = torch.cat([positives_upper, positives_lower], dim=0)  # 2B

        # Compute per-row denominator
        den = torch.sum(
            self.negatives_mask * torch.exp(S / self.temperature), dim=1
        )  # 2B

        loss = -torch.mean(positives / self.temperature - torch.log(den))

        return loss


class SimCLR(nn.Module):
    def __init__(self, batch_size, temperature=0.5, device="cuda"):
        super().__init__()

        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        # ResNet with modification for smaller images (CIFAR-10)
        resnet = torchvision.models.resnet50()
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        resnet.fc = nn.Identity()

        self.base_encoder = resnet

        linear1 = torch.nn.Linear(2048, 2048, bias=False)
        relu = torch.nn.ReLU()
        linear2 = torch.nn.Linear(2048, 128, bias=False)

        self.projection_head = torch.nn.Sequential(linear1, relu, linear2)

        self.loss = NTXentLoss(
            batch_size=self.batch_size, temperature=self.temperature, device=self.device
        )

    def forward(self, x1, x2):
        B, C, H, W = x1.shape

        h1 = self.base_encoder(x1)
        z1 = self.projection_head(h1)

        h2 = self.base_encoder(x2)
        z2 = self.projection_head(h2)

        loss = self.loss(z1, z2)

        return loss


#: Datasets


class SimCLRTrainDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.base_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True
        )
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (32, 32), scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                get_random_color_distortion(0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __getitem__(self, index):
        image, label = self.base_dataset[index]
        x1 = self.transform(image)
        x2 = self.transform(image)
        return x1, x2, label

    def __len__(self):
        return len(self.base_dataset)


class SimCLRTestDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.base_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __getitem__(self, index):
        image, label = self.base_dataset[index]
        x = self.transform(image)
        return x, label

    def __len__(self):
        return len(self.base_dataset)


#: Evaluation


def evaluate_linear_classifier(
    model, trainloader, testloader, device, num_classes=10, epochs=2
):
    """
    Train a linear classifier on top of the base_encoder features and evaluate.
    Returns top-1 test accuracy and test loss.
    """
    model.eval()

    # Get feature dimension from first batch
    with torch.no_grad():
        for batch in trainloader:
            x1, x2, labels = batch
            x1 = x1.to(device)
            features = model.base_encoder(x1)
            feature_dim = features.shape[1]
            break

    # Create linear classifier
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-6)

    # Train classifier using batches from trainloader
    classifier.train()
    for _ in range(epochs):
        for batch in trainloader:
            x1, x2, labels = batch
            x1 = x1.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                features = model.base_encoder(x1)

            optimizer.zero_grad()
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    # Evaluate on test set using testloader
    classifier.eval()
    test_losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in testloader:
            x, labels = batch
            x = x.to(device)
            labels = labels.to(device)

            features = model.base_encoder(x)
            logits = classifier(features)
            loss = criterion(logits, labels)
            test_losses.append(loss.item())

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    test_loss = np.mean(test_losses)
    test_accuracy = correct / total

    return test_accuracy, test_loss


#: Main


if __name__ == "__main__":
    print("START")

    C = 3
    H = 32
    W = 32

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project="2025-10-simclr", config=config)

    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    wandb.config.update({"device": str(device)})
    if torch.cuda.is_available():
        wandb.config.update({"cuda_device": torch.cuda.get_device_name(0)})

    trainset = SimCLRTrainDataset()
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=wandb.config.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    testset = SimCLRTestDataset()
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=wandb.config.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=True,
    )

    print("Creating model")

    model = SimCLR(
        batch_size=wandb.config.batch_size,
        temperature=wandb.config.temperature,
        device=device,
    )
    model = model.to(device)

    wandb.watch(model, log="all", log_freq=100)

    print("Running...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=wandb.config.learning_rate,
        weight_decay=wandb.config.weight_decay,
    )

    # Plot augmented pairs of images for the batch
    # print(" ".join(f"{classes[labels[j]]:5s}" for j in range(batch_size)))
    # grid = torch.cat([x1, x2], dim=0)
    # imshow(torchvision.utils.make_grid(grid, nrow=batch_size))

    model.train()
    for epoch in range(wandb.config.num_epochs):
        epoch_losses = []
        epoch_batch_times = []

        for ix, batch in enumerate(trainloader):
            batch_start_time = time.time()

            x1, x2, labels = batch

            optimizer.zero_grad()

            x1, x2 = x1.to(device), x2.to(device)
            loss = model(x1, x2)
            loss.backward()

            optimizer.step()

            batch_time = time.time() - batch_start_time
            epoch_losses.append(loss.item())
            epoch_batch_times.append(batch_time)

            print(
                f"Epoch={epoch} Batch={ix}/{len(trainloader)} Loss={loss.item():.4f} Time={batch_time:.3f}s"
            )

        # Log epoch-level metrics (standard practice for ML experiments)
        epoch_avg_loss = np.mean(epoch_losses)
        epoch_avg_time = np.mean(epoch_batch_times)

        log_dict = {
            "loss": epoch_avg_loss,
            "batch_time": epoch_avg_time,
            "samples_per_sec": wandb.config.batch_size / epoch_avg_time,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch": epoch,
        }

        # Evaluate linear classifier periodically
        if epoch % 2 == 0 or epoch == wandb.config.num_epochs - 1:
            test_accuracy, test_loss = evaluate_linear_classifier(
                model, trainloader, testloader, device, num_classes=10, epochs=100
            )
            log_dict["linear_test_accuracy"] = test_accuracy
            log_dict["linear_test_loss"] = test_loss

        wandb.log(log_dict, step=epoch)

    model.eval()

    # Finalize wandb run
    wandb.finish()

    print("END")
