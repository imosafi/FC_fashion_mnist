import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FCModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(FCModel, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, input):
        out = F.tanh(self.linear1(input))
        out = self.linear2(out)
        return F.log_softmax(out)
