


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




# this function performs the optimization for one batch of the data



def train_epoch(model, opt, X, Y, d, batch_size=50):
    
    
    model.train()
    criterion = nn.CrossEntropyLoss()

    for beg_i in range(0, X.size(0), batch_size):
        opt.zero_grad()
        x_batch = Variable(X[beg_i:beg_i + batch_size, :])
        x_batch=x_batch.view(-1,d)
        y_batch = Y[beg_i:beg_i + batch_size, :]


        

        

            #losses,
        logits, probas = model(x_batch)
        loss=criterion(logits, y_batch.view(logits.size()[0]).long())
        opt.zero_grad()
        loss.backward()

        opt.step()
        #except RuntimeError:
            #print(logits, y_batch)
            #print(y_batch.size())
            



        #loss = bound_loss(lyap_pred,ubound,lbound) + gweight * grad_loss_eq(grad_Lyap, x_batch, vf_batch)

        #loss2=bound_loss(y_pred,ubound,lbound) + gweight * grad_loss_eq(gradx, x_batch, vf_batch)
        # (3) Compute gradients




def train(model,X,Y,d,num_epochs=50):


    #z=(torch.rand(10000, 2)-0.5)*10


    #opt = optim.Adam(set(a.parameters()).union(model.parameters()), lr=0.05, betas=(0.9, 0.999))
    opt = optim.Adam(set(model.parameters()), lr=0.05, betas=(0.9, 0.999))
    #criterion = nn.BCELoss()
    

    
    

    for e in range(num_epochs):
        train_epoch(model, opt, X, Y, d, batch_size=50)
        print('epoch= %', e)
      
    #plt.plot(e_losses)





