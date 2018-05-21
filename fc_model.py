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
        self.batch_norm1 = nn.BatchNorm1d(100)
        self.batch_norm2 = nn.BatchNorm1d(50)

    #with batchnorm
    def forward(self, input):
        out = F.relu(self.batch_norm1(self.linear1(input)))
        out = F.relu(self.batch_norm2(self.linear2(out)))
        out = self.linear3(out)
        return F.log_softmax(out)


    # with dropout
    # def forward(self, input):
    #     out = F.relu(self.linear1(input))
    #     out = F.dropout(out, p=0.2, training=self.training)
    #     out = F.relu(self.linear2(out))
    #     out = F.dropout(out, p=0.2, training=self.training)
    #     out = self.linear3(out)
    #     return F.log_softmax(out)


    ##regular
    # def forward(self, input):
    #     out = F.relu(self.linear1(input))
    #     out = F.relu(self.linear2(out))
    #     out = self.linear3(out)
    #     return F.log_softmax(out)