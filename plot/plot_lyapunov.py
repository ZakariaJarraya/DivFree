import numpy as np
import torch
import matplotlib.pyplot as plt

def limit_points(X,reseau=torch.nn.Identity(2)):
    

    i=0

    xmin=[]
    xmax=[]
    ymax=[]
    ymin=[]
    while i<=100:
        
        try:
            reseau.Identity.register_forward_hook(get_activation('Identity'))
            output = reseau(X,t=i)
            xmin.append(np.min(activation['Identity'][:,0].numpy()))
            xmax.append(np.max(activation['Identity'][:,0].numpy()))
            ymin.append(np.min(activation['Identity'][:,1].numpy()))
            ymax.append(np.max(activation['Identity'][:,1].numpy()))
            i+=1
        except AttributeError:
            output = reseau(X)    
            xmin.append(reseau(X)[:,1].numpy())
            xmax.append(reseau(X)[:,0].numpy())
            ymin.append(reseau(X)[:,1].numpy())
            ymax.append(reseau(X)[:,1].numpy())
            i+=1
    return np.min(xmin),np.max(xmax),np.min(ymin),np.max(ymax)





def plot_lyapunov(X,X0,X1,lyapunov,numpoints,reseau=torch.nn.Identity(2)):
    # define plotting range and mesh
    

    xmin,xmax,ymin,ymax = limit_points(X,reseau)

    _x = np.linspace(xmin, xmax, numpoints)
    _y = np.linspace(ymin, ymax, numpoints)

    _X, _Y = np.meshgrid(_x, _y)

    s= _X.shape


    Ze = np.zeros(s)
    Zp = np.zeros(s)
    DT = np.zeros((numpoints**2,2))

    # convert mesh into point vector for which the model can be evaluated
    c = 0
    for i in range(s[0]):
        for j in range(s[1]):
            DT[c,0] = _X[i,j]
            DT[c,1] = _Y[i,j]
            c = c+1;


    Ep=torch.stack([lyapunov(i) for i in torch.tensor(DT).float()])        

    c = 0
    for i in range(s[0]):
        for j in range(s[1]):
            #Ze[i,j] = Ee[c]
            Zp[i,j] = Ep[c]
            c = c+1;

    # define figure
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');

    # plot values V
    #ax.plot_surface(_X, _Y, Zp, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    ax.plot_wireframe(_X, _Y, Zp, rstride=7, cstride=7)


    # plot orbital derivative DVf
    #ax.plot_wireframe(X, Y, Ze, rstride=1, cstride=1)
    ax.scatter3D(X1[:, 0], X1[:, 1], s=2, color='green')
    ax.scatter3D(X0[:, 0], X0[:, 1], s=2, color='blue')
    #ax.scatter3D(X.numpy()[1000:2000,0], X.numpy()[1000:2000,1], z1[:1000], s=2, color='blue')
    plt.show()
    