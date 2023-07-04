
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
import geotorch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons, make_circles
import numpy 




def train_epoch(model, Y, X, opt, criterion, batch_size=50):
    relu=nn.ReLU()
    model.train()
    
    #Y=f(X)
    losses = []
    for beg_i in range(0, X.size(0), batch_size):
        x_batch = Variable(X[beg_i:beg_i + batch_size, :])
        y_batch = Y[beg_i:beg_i + batch_size, :]
        #x_batch = Variable(x_batch)
        #y_batch = Variable(y_batch)

        opt.zero_grad()
        # (1) Forward
        y_hat = v(x_batch)
        # (2) Compute diff
        loss = criterion(x_batch,y_batch)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()        
        losses.append(loss.data.numpy())
    return losses








def train(v,f,num_epochs=100):




    opt = optim.Adam(v.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()


    e_losses = []

    for e in range(num_epochs):
        e_losses += train_epoch(v, f, opt, criterion)
    plt.plot(e_losses)












