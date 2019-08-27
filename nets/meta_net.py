from __future__ import print_function
import torch.nn as nn
from torch.nn import functional as F

class meta_Net(nn.Module):
    def __init__(self, if_relu): # it only gets paramters from other network's parameters
        super(meta_Net, self).__init__()
        self.vars = nn.ParameterList()
        self.softmax = nn.LogSoftmax(dim=1)
        if if_relu:
            self.activ = nn.ReLU()
        else:
            self.activ = nn.Tanh()
    def forward(self, x, var):
        idx = 0
        while idx < len(var):
            if idx > 0:
                x = self.activ(x)
            if idx == 0:
                w1, b1 = var[idx], var[idx + 1] # weight and bias
                x = F.linear(x, w1, b1)
                idx += 2
            elif idx == 2:
                w2, b2 = var[idx], var[idx + 1]  # weight and bias
                x = F.linear(x, w2, b2)
                idx += 2
            elif idx == 4:
                w3, b3 = var[idx], var[idx + 1]  # weight and bias
                x = F.linear(x, w3, b3)
                idx += 2
            elif idx == 6:
                w4, b4 = var[idx], var[idx + 1]  # weight and bias
                x = F.linear(x, w4, b4)
                idx += 2
            elif idx == 8:
                w5, b5 = var[idx], var[idx + 1]  # weight and bias
                x = F.linear(x, w5, b5)
                idx += 2
        x = self.softmax(x)
        return x

def meta_net(**kwargs):
    net = meta_Net(**kwargs)
    return net
