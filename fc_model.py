import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FCModel(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, out_dim):
        super(FCModel, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, out_dim)

    def forward(self, input):
        out = F.relu(self.linear1(input))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        return F.log_softmax(out)
