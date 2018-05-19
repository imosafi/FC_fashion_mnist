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
from utils import save_test_val_acc_loss_plots

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(epoch_num, model, train_loader, optimizer):
    print('Epoch {}:'.format(epoch_num))
    model.train()
    train_loss = 0.0
    correct = 0.0
    for data, labels in train_loader:
        data = data.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)

        train_loss += F.nll_loss(output, labels, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(labels.data.view_as(pred)).sum()

        loss.backward()
        optimizer.step()
    train_acc = float(100.0 * correct) / len(train_loader.sampler)
    return train_acc, float(train_loss / len(train_loader.sampler))


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
    test_acc = float(100.0 * correct) / len(test_loader.sampler)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.sampler), test_acc))
    return test_acc, float(test_loss)


def get_train_valid_loaders(data_dir, batch_size):
    train_dataset = datasets.FashionMNIST(
        root=data_dir, train=True,
        download=True, transform=transforms.ToTensor())

    valid_dataset = datasets.FashionMNIST(
        root=data_dir, train=True,
        download=True, transform=transforms.ToTensor())

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.2 * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=1, pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=1, pin_memory=True)

    return (train_loader, valid_loader)


def save_test_predictions(model, path):
    return None


def main():
    print(sys.version)
    print(torch.__version__)
    print('GPU is available' if torch.cuda.is_available() else 'GPU is not available')

    train_loader, val_loader = get_train_valid_loaders('./data', batch_size=15)

    model = FCModel(784, 100, 10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    train_acc_list, train_loss_list, val_acc_list, val_loss_list = [], [], [], []
    for epoch in range(70):
        train_acc, train_loss = train(epoch + 1, model, train_loader, optimizer)
        val_acc, val_loss = validate(model, val_loader)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)
    save_test_val_acc_loss_plots(train_acc_list, val_acc_list, train_loss_list, val_loss_list)
    save_test_predictions(model, 'data/test_x')


if __name__ == '__main__':
    main()
