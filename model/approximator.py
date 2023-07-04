
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons, make_circles
import numpy 
import torch.nn.functional as F





class V(nn.Module):

    def __init__(self, dim_in, dim_hid, dim_out):
        super(V, self).__init__()
        self.layer1 = nn.Linear(dim_in, dim_hid)
        #self.res=nn.Linear(dim_hid, dim_hid)
        self.activation=nn.ReLU()
        self.layer2 = nn.Linear(dim_hid, dim_hid)
        self.layer3 = nn.Linear(dim_hid, dim_hid)
        self.layer4 = nn.Linear(dim_hid, dim_out)
        self.finalLayer=nn.Linear(dim_out,2)
    def forward(self, x):
        x=self.layer1(x)
        x=self.activation(x)
        x=self.layer2(x)
        x=self.activation(x)
        x=self.layer3(x)
        x=self.activation(x)
        x=self.layer4(x)

        logits=self.finalLayer(x)
        probas = F.softmax(logits, dim=1)

        return logits, probas