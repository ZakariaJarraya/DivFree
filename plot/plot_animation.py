import numpy as np
import torch
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(4,4))




def f(x,i):
    i=i%58
    return x[i:i+2]
def limit(model,X,limit=30):


    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook



    i=0

    xmin=[]
    xmax=[]
    ymax=[]
    ymin=[]
    while i<=limit:
        model.Identity.register_forward_hook(get_activation('Identity'))
        output = model(X,t=i)
        xmin.append(np.min(activation['Identity'][:,0].numpy()))
        xmax.append(np.max(activation['Identity'][:,0].numpy()))
        ymin.append(np.min(activation['Identity'][:,1].numpy()))
        ymax.append(np.max(activation['Identity'][:,1].numpy()))
        i+=1

    return np.min(xmin),np.min(ymin),np.max(xmax),np.max(ymax)


def plot(model,X):
        #xmin,ymin,xmax,ymax=limit(model,X,limit=30)
        i=0
        data_temp=X[0:2]
        def update(f):
            global data_temp, i
            
            
            
            ax.clear()
            ax.set_xlim(1.5*np.min(X[:,0]),1.5*np.max(X[:,0]))
            ax.set_ylim(1.5*np.min(X[:,1]),1.5*np.max(X[:,1]))





            #ax.set_xlim(1.5*xmin,1.5*xmax)
            #ax.set_ylim(1.5*ymin,1.5*ymax)
            #ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.Greens_r)
            #newline([1.5,1.5],[0.,3.])
            #ax.quiver(g,h,k,l,EE,cmap='gist_gray')
            ax.set_axis_off()
            ax.scatter(data_temp.detach().numpy()[:,0], data_temp.detach().numpy()[:,1])
            
            i=i+2
            i=i%60
            data_temp=X[i:i+2]
        



        #call the animation and save
        p = FuncAnimation(fig, update, interval=100, frames=range(100))
        p.save(filename="b.mp4", dpi =80, fps=10)
        #p.save('b.gif', writer='pillow')








