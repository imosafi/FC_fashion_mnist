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
from fc_model import FCModel, ConvNet
import torch.utils

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from utils import save_test_val_acc_loss_plots

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IS_CNN = False


def train(epoch_num, model, train_loader, optimizer):
    print('Epoch {}:'.format(epoch_num))
    model.train()
    train_loss = 0.0
    correct = 0.0
    for data, labels in train_loader:
        if IS_CNN:
            data = data.to(device)
        else:
            data = data.reshape(-1, 28 * 28).to(device)
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
    train_loss = float(train_loss / len(train_loader.sampler))
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        train_loss, correct, len(train_loader.sampler), train_acc))
    return train_acc, train_loss


def validate(model, test_loader):
    model.eval()
    val_loss = 0.0
    correct = 0.0
    for data, target in test_loader:
        if IS_CNN:
            data = data.to(device)
        else:
            data = data.reshape(-1, 28 * 28).to(device)
        target = target.to(device)

        output = model(data)
        val_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).sum()
    val_loss /= len(test_loader.sampler)
    val_acc = float(100.0 * correct) / len(test_loader.sampler)
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        val_loss, correct, len(test_loader.sampler), val_acc))
    return val_acc, val_loss


def get_train_validation_test_loaders(data_dir, batch_size):
    train_dataset = datasets.FashionMNIST(
        root=data_dir, train=True,
        download=True, transform=transforms.ToTensor())

    valid_dataset = datasets.FashionMNIST(
        root=data_dir, train=True,
        download=True, transform=transforms.ToTensor())

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=False)

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

    return (train_loader, valid_loader, test_loader)


def save_test_predictions(model, path):
    test_x = torch.Tensor(np.loadtxt(path) / 255).to(device)
    if IS_CNN:
        test_x = test_x.view(5000, 1, 28, 28)

    output = model(test_x)
    preds = output.data.max(1, keepdim=True)[1]

    with open('output/test.pred', 'w+') as f:
        f.writelines(map(lambda x: str(int(x)) + '\n', preds))


def create_confusion_matrix(model, test_loader, categories_num=10):
    confusion_martix = []
    for i in range(categories_num):
        confusion_martix.append([0] * categories_num)
    for data, labels in test_loader:
        if IS_CNN:
            data = data.to(device)
        else:
            data = data.reshape(-1, 28 * 28).to(device)

        labels = labels.to(device)

        output = model(data)
        preds = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        # correct += pred.eq(labels.data.view_as(pred)).sum()
        for pred, label in zip(preds, labels):
            confusion_martix[label][pred] += 1

    print('\nConfustion matrix:\n')
    for l in range(categories_num):
        print(confusion_martix[l])


def print_test_acc_loss(model, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0.0
    for data, target in test_loader:
        if IS_CNN:
            data = data.to(device)
        else:
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
    return test_acc, test_loss


def main():
    print(sys.version)
    print(torch.__version__)
    print('GPU is available' if torch.cuda.is_available() else 'GPU is not available')

    train_loader, val_loader, test_loader = get_train_validation_test_loaders('./data', batch_size=15)

    if IS_CNN:
        model = ConvNet().to(device)
    else:
        model = FCModel(784, 100, 50, 10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    train_acc_list, train_loss_list, val_acc_list, val_loss_list = [], [], [], []
    for epoch in range(10):
        train_acc, train_loss = train(epoch + 1, model, train_loader, optimizer)
        val_acc, val_loss = validate(model, val_loader)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

    print_test_acc_loss(model, test_loader)

    save_test_val_acc_loss_plots(train_acc_list, val_acc_list, train_loss_list, val_loss_list)
    create_confusion_matrix(model, test_loader)
    save_test_predictions(model, 'data/test_x')


if __name__ == '__main__':
    main()
