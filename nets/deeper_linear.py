from __future__ import print_function
import torch.nn as nn

class deeper_Net(nn.Module):
    def __init__(self, args, m_ary, num_neurons_first, num_neurons_second, num_neurons_third, if_bias, if_relu):
        if m_ary == 5:
            m_ary = 4
        super(deeper_Net, self).__init__()
        self.layer_list = []
        if args.if_cavia:
            self.fc1 = nn.Linear(2 + args.num_context_para, num_neurons_first, bias=if_bias)  # 10
        else:
            self.fc1 = nn.Linear(2, num_neurons_first, bias=if_bias) #10
        self.layer_list.append(self.fc1)
        if num_neurons_second is not None:
            self.fc2 = nn.Linear(num_neurons_first, num_neurons_second, bias=if_bias)
            self.layer_list.append(self.fc2)
        if num_neurons_third is not None:
            self.fc3 = nn.Linear(num_neurons_second, num_neurons_third, bias=if_bias)
            self.layer_list.append(self.fc3)
        if num_neurons_second == None:
            num_neurons_final = num_neurons_first
        elif num_neurons_third == None:
            num_neurons_final = num_neurons_second
        else:
            num_neurons_final = num_neurons_third

        self.fc_last = nn.Linear(num_neurons_final, m_ary, bias=if_bias) # two inputs (real and im), two neurons
        self.softmax = nn.LogSoftmax(dim=1)
        if if_relu:
            self.activ = nn.ReLU()
        else:
            self.activ = nn.Tanh()
    def forward(self, x):
        for j in range(len(self.layer_list)):
            x = self.layer_list[j](x)
            x = self.activ(x)
        x = self.fc_last(x)
        x = self.softmax(x)
        return x

def deeper_linear_net(**kwargs):
    net = deeper_Net(**kwargs)
    return net

def deeper_linear_net_prime(**kwargs):
    net_prime = deeper_Net(**kwargs)
    return net_prime

def deeper_linear_net_prime_prime(**kwargs):
    net_prime_prime = deeper_Net(**kwargs)
    return net_prime_prime