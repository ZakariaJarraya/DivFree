import torch.nn.functional as F
from data import data as dt
from model import lyapunov
from plot import plot_lyapunov as pl
from plot import plot_field as p
from model import resnet as rs
from model import approximator as app
from trainer import approximator_trainer as app_trainer
from trainer import resnet_trainer
#from plot import plot_animation as anime
from matplotlib.animation import FuncAnimation
import numpy as np
from plot import animate_lyapunov as anime 
import matplotlib.animation as animation

import torch
import matplotlib.pyplot as plt
import umap.umap_ as umap
import seaborn as sns
import torch.nn as nn
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

CUDA_LAUNCH_BLOCKING=1

DEVICE = "cuda:0"
GRAYSCALE = True

import gc

gc.collect()

torch.cuda.empty_cache()


device = torch.device(DEVICE)







#fig = plt.figure()
#ax = plt.axes(projection='3d')
"""
mean1=torch.tensor([0.,0.])
mean2=torch.tensor([3.,3.])
covariance1=torch.tensor([[1.,0.],[0.,1.]])
covariance2=torch.tensor([[1.,0.],[0.,1.]])







data_choice=int(input('choose between 1(two_circles), 2(two_moons) and 3(two_gaussians): '))
class_size=int(input('choose class size: '))


data=dt.DATA(data_choice,class_size,mean1,mean2,covariance1,covariance2)



data.load()


X,Y,X0,X1= data.prepare()











a=lyapunov.lyapunov(c1=mean1,c2=mean2)

residual_block=lyapunov.grad(a)


#residual_block=app.V(2,2,2)

#residual_block=lyapunov.dynamics_init()
print(a.c1)
print(a.c2)
pl.plot_lyapunov(X,X0,X1,a,numpoints=40)
reseau=rs.resnet(2,2,2,residual_block)
embedder=app.V(2,2,2)



grad=resnet_trainer.train(embedder,reseau,a,X,Y,num_epochs=100)
print(a.c1)
print(a.c2)
for i in range(len(grad)):
    if i % 2 == 0:
        grad[i]=grad[i][0]
grad=np.array(grad)

#p.plot_vector_field(X,X0,X1,reseau,20)

print(torch.sum(torch.tensor(torch.round(reseau(X,30))==Y))/len(Y))


#anime.plot(reseau,grad)

#anime.plot(grad, X, X0, X1, a)









fig, ax = plt.subplots(figsize=(4,4))
color= ['red' if l == 0. else 'green' for l in Y]
i=0
data_temp=grad[0:2]
def update(f):
            global data_temp, i
            
            
            
            ax.clear()
            ax.set_xlim(1.5*np.min(grad[:,0]),1.5*np.max(grad[:,0]))
            ax.set_ylim(1.5*np.min(grad[:,1]),1.5*np.max(grad[:,1]))
            

            



            #ax.set_xlim(1.5*xmin,1.5*xmax)
            #ax.set_ylim(1.5*ymin,1.5*ymax)
            #ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.Greens_r)
            #newline([1.5,1.5],[0.,3.])
            #ax.quiver(g,h,k,l,EE,cmap='gist_gray')
            #ax.set_axis_off()
            ax.scatter(data_temp[:,0], data_temp[:,1])
            ax.scatter(X[:,0], X[:,1], color=color)
            


            i=i+2
            i=i%148
            data_temp=grad[i:i+2]
        



        #call the animation and save
p = FuncAnimation(fig, update, interval=150, frames=range(150))
#p.save('8-5.gif', writer='pillow')











   

"""
d=2
fig = plt.figure()
ax = plt.axes(projection='3d')

mean1=0*torch.ones(1, d)
mean2=3*torch.ones(1, d)
covariance1=torch.eye(d)
covariance2=torch.eye(d)





data_choice=int(input('choose between 1(two_circles), 2(two_moons) and 3(two_gaussians): '))
class_size=int(input('choose class size: '))


data=dt.DATA(data_choice,class_size,mean1,mean2,covariance1,covariance2)



data.load()


X,Y,X0,X1= data.prepare()

#for i in range(len(Y)):
    #X[i]=X[i].to(device)
    #Y[i]=Y[i].to(device)

a=lyapunov.lyapunov(mean1,mean2, d)

residual_block=lyapunov.grad(a, d)
residual_block=lyapunov.dynamics_init(mean1,mean2, d)

#residual_block=app.V(50,50,50)

#residual_block=lyapunov.dynamics_init()
print(a.c1)
print(a.c2)
#pl.plot_lyapunov(X,X0,X1,a,numpoints=40)
reseau=rs.resnet(d,ResidualBlock=residual_block)

#reseau.to(DEVICE)

X=X.view(-1,d)


grad=resnet_trainer.train(reseau,X,Y,d,num_epochs=100)
print(a.c1)
print(a.c2)
#for i in range(len(grad)):
    #if i % 2 == 0:
        #grad[i]=grad[i][0]
#grad=np.array(grad)

#p.plot_vector_field(X,X0,X1,reseau,20)

#print(torch.sum(torch.tensor(torch.round(reseau(X.view(-1,d),30))==Y))/len(Y))
 

embedding = umap.UMAP().fit_transform(X.view(-1,d).cpu().detach().numpy())


col0=[]
for i in range(len(embedding[:,0])):
    col0.append(embedding[:,0][i])



col1=[]
for i in range(len(embedding[:,1])):
    col1.append(embedding[:,1][i])   









def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, -5, 5)
    # Return the perturbed image
    return perturbed_image









def test( model, X, Y, epsilon):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for i in range(len(Y)):



        # Set requires_grad attribute of tensor. Important for Attack
        s=X[i].clone().detach().requires_grad_(True)
        #s=s.view(-1,d)

        #s=s.view(-1,d)
        # Forward pass the data through the model
        output = model(s.view(-1,d))[1]
        #init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        #output=output.view(1)

        criterion = nn.CrossEntropyLoss()
        # Calculate the loss
        loss = criterion(output, Y[i].long())

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward(retain_graph=True)

        # Collect datagrad
        data_grad = s.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(s, epsilon, data_grad)

        # Re-classify the perturbed image
        adv_examples.append(perturbed_data)

    return adv_examples    






activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


network=app.V(d,70,d)
#network.to(DEVICE)
grad=resnet_trainer.train(network,X,Y,d,num_epochs=100)


def acc(network, liste, Y):
     acc=0
     indice=[]
     for i in range(len(Y)):
            if torch.argmax(network(liste[i].view(-1,d))[1])==Y[i]:
                     acc+=1
            else:
                pass
                indice.append(i)         
     return acc/len(Y), indice







def plot_decision_boundary(network,X,y):
    fig = plt.figure()
    
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx,yy=np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = torch.argmax(network(torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float())[1],dim=1)   
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    #plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.binary)     
    plt.show()




