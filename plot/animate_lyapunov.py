import numpy as np
print('numpy: '+np.version.full_version)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.animation as animation
import matplotlib
import torch
print('matplotlib: '+matplotlib.__version__)


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


def structure_data(X,X0,X1,lyapunov,numpoints,reseau=torch.nn.Identity(2)):
    # define plotting range and mesh
    

    xmin,xmax,ymin,ymax = limit_points(X,reseau)

    _x = np.linspace(xmin, xmax, numpoints)
    _y = np.linspace(ymin, ymax, numpoints)

    _X, _Y = np.meshgrid(_x, _y)

    s= _X.shape


    Ze = np.zeros(s)
    Zp = np.zeros(s)
    DT = np.zeros((numpoints**2,2))



    Ep=torch.stack([lyapunov(i) for i in torch.tensor(DT).float()])        

    c = 0
    for i in range(s[0]):
        for j in range(s[1]):
            #Ze[i,j] = Ee[c]
            Zp[i,j] = Ep[c]
            c = c+1;

    # define figure
    return Zp, _X, _Y 



   




def plot(centres, X, X0, X1, lyapunov, numpoints, reseau=torch.nn.Identity(2), fps=10):
    
    def update(frame_number,centres, X, X0, X1,lyapunov, numpoints, reseau=torch.nn.Identity(2)):
        
        ax.clear()
        frame_number=frame_number % 20
        lyapunov.c1=centres[2*frame_number]
        lyapunov.c2=centres[2*frame_number+1]
        Zp, _X, _Y=structure_data(X,X0,X1,lyapunov,40)


        ax.plot_wireframe(_X, _Y, Zp, rstride=7, cstride=7)
        ax.scatter3D(X1[:, 0], X1[:, 1], s=2, color='green')
        ax.scatter3D(X0[:, 0], X0[:, 1], s=2, color='blue')
        frame_number+=2        





    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ani = animation.FuncAnimation(fig, update, 50, fargs=(centres, X, X0, X1, lyapunov), interval=1000/fps)

    fn = 'plot_surface_animation_funcanimation'
    #ani.save(fn+'.mp4',writer='ffmpeg',fps=fps)
    ani.save(fn+'.gif',writer='imagemagick',fps=fps)







