# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import torch
from model import SimpleNetwork
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset


class StripeDataset(Dataset):
    """Small deterministic image dataset with a learnable label pattern."""

    def __init__(self, size, seed, noise=0.03):
        generator = torch.Generator().manual_seed(seed)
        labels = torch.arange(size, dtype=torch.long) % 10
        self.labels = labels[torch.randperm(size, generator=generator)]
        self.images = torch.randn(size, 3, 32, 32, generator=generator) * noise
        for index, label in enumerate(self.labels.tolist()):
            row = 2 + label * 3
            self.images[index, 0, row : row + 2, :] = 1.0
            self.images[index, 1, :, row : row + 2] = 0.5
            self.images[index, 2, row : row + 2, row : row + 2] = 1.5

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def build_loaders(batch_size, train_size, test_size, seed):
    train_dataset = StripeDataset(size=train_size, seed=seed)
    test_dataset = StripeDataset(size=test_size, seed=seed + 10_000)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / max(len(train_loader), 1)


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            predicted = model(images).argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model = SimpleNetwork().to(device)
    train_loader, test_loader = build_loaders(args.batch_size, args.train_size, args.test_size, args.seed)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        accuracy = evaluate(model, test_loader, device)
        print(f"epoch={epoch + 1} loss={loss:.4f} accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
