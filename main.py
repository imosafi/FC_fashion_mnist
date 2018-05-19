import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from fc_model import FCModel

EPOCHS = 10

def train(model):
    model.train()

def test(model):
    model.eval()


def main():
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('gpu is available')

    model = FCModel(784, 100, 10)



    for epoch in range(EPOCHS):
        train()
        test()


if __name__ == '__main__':
    main()