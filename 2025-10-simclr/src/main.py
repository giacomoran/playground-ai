import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

#: Constants

B = 4
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


class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape

        # TODO:
        return x1, x2


#: Datasets


class SimCLRTrainDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.base_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomResizedCrop(
                    (H, W), scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                get_random_color_distortion(0.5),
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


#: Main


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    torch.manual_seed(0)
    np.random.seed(0)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(0)

    trainset = SimCLRTrainDataset()
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=B, shuffle=True, num_workers=8
    )

    testset = SimCLRTestDataset()
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=B, shuffle=False, num_workers=8
    )

    dataiter = iter(trainloader)

    model = SimCLR()

    for i in range(3):
        x1, x2, labels = next(dataiter)
        print(" ".join(f"{classes[labels[j]]:5s}" for j in range(B)))

        grid = torch.cat([x1, x2], dim=0)
        imshow(torchvision.utils.make_grid(grid, nrow=B))
