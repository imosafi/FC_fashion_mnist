import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import torch.backends.cudnn as cudnn
import datetime
import os
import matplotlib.pyplot as plt
import torchvision
import sys
from fc_model import FCModel
import torch.utils

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(epoch_num, model, train_loader, optimizer):
    print('Epoch {}:'.format(epoch_num))
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()


def validate(model, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0.0
    for data, target in test_loader:
        data = data.reshape(-1, 28*28).to(device)
        target = target.to(device)

        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.sampler)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.sampler),
        float(100.0 * correct) / len(test_loader.sampler)))


def get_train_valid_loaders(data_dir, batch_size):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            # normalize,
    ])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        # normalize,
    ])

    train_dataset = datasets.FashionMNIST(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.FashionMNIST(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.2 * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=1, pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=1, pin_memory=True,
    )

    return (train_loader, valid_loader)


def main():
    print(sys.version)
    print(torch.__version__)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('GPU is available')
    else:
        print('GPU is not available')


    # train_loader = torchvision.datasets.FashionMNIST('./data', train=True, download=True)
    # test_loader = torchvision.datasets.FashionMNIST('./data', train=False, download=True)

    train_loader, val_loader = get_train_valid_loaders('./data', batch_size=30)

    # l1_labels_count = [0] * 10
    # l2_labels_count = [0] * 10
    # # for i in l1.datase
    #
    # for batch_idx, (data, labels) in enumerate(l1):
    #     l1_labels_count[int(labels)] += 1
    #
    # for batch_idx, (data, labels) in enumerate(l2):
    #     l2_labels_count[int(labels)] += 1
    #
    # print('train')
    # for i in range(10):
    #     print('label {}: {}'.format(i, l1_labels_count[i]))
    #
    # print('test')
    # for i in range(10):
    #     print('label {}: {}'.format(i, l2_labels_count[i]))

    model = FCModel(784, 100, 10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)


    for epoch in range(70):
        train(epoch + 1, model, train_loader, optimizer)
        validate(model, val_loader)


if __name__ == '__main__':
    main()
