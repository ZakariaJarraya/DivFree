import numpy as np
import torch
import matplotlib.pyplot as plt


def limit_points(X,reseau):
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
    while i<=10:
        reseau.Identity.register_forward_hook(get_activation('Identity'))
        output = reseau(X,t=i)
        xmin.append(np.min(activation['Identity'][:,0].detach().cpu().numpy()))
        xmax.append(np.max(activation['Identity'][:,0].detach().cpu().numpy()))
        ymin.append(np.min(activation['Identity'][:,1].detach().cpu().numpy()))
        ymax.append(np.max(activation['Identity'][:,1].detach().cpu().numpy()))
        i+=1
    return np.min(xmin),np.max(xmax),np.min(ymin),np.max(ymax)


def plot_vector_field(X,reseau,numpoints):
    xmin,xmax,ymin,ymax = limit_points(X,reseau)
    x,y = np.meshgrid(np.linspace(-20,20,25),np.linspace(-2,2,25))
    u=[]
    p=[]
    for j in torch.flatten(torch.from_numpy(np.transpose(y)[0])):
        for i in torch.flatten(torch.from_numpy(x[0])):
                u.append(float(reseau.g(torch.tensor([[i,j]]).float().to('cuda'))[0][0]))
                p.append(float(reseau.g(torch.tensor([[i,j]]).float().to('cuda'))[0][1]))
    u=np.array(u)
    p=np.array(p)
    u.resize(25,25)
    p.resize(25,25)
    V_norm=np.sqrt(u**2 + p**2)
    plt.contourf(x,y,V_norm, cmap="YlGn")
    plt.quiver(x,y,u,p)
    plt.show()





def quiver_points(X,reseau,numpoints,methode,donnee):
    xmin,xmax,ymin,ymax = limit_points(X,reseau)
    x,y = np.meshgrid(np.linspace(-20,20,25),np.linspace(-2,2,25))
    u=[]
    p=[]
    for j in torch.flatten(torch.from_numpy(np.transpose(y)[0])):
        for i in torch.flatten(torch.from_numpy(x[0])):
                u.append(float(reseau.g(torch.tensor([[i,j]]).float().to('cuda'))[0][0]))
                p.append(float(reseau.g(torch.tensor([[i,j]]).float().to('cuda'))[0][1]))
    u=np.array(u)
    p=np.array(p)
    u.resize(25,25)
    p.resize(25,25)
    V_norm=np.sqrt(u**2 + p**2)
    plt.contourf(x,y,V_norm, cmap="YlGn")
    plt.quiver(x,y,u,p)
    plt.axis('off')
    plt.savefig('field' + '-' +methodes[int(methode)]+'-'+donnees[int(donnee)]+'.png')

    












def animation(model,x,y,methode,donne):
    from matplotlib.animation import FuncAnimation
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    fig, ax = plt.subplots(figsize=(4,4))
    color= ['Blue' if l == 0. else 'red' for l in y]
    def animate_reseau(i):
            ax.clear()
            xmin,xmax,ymin,ymax = limit_points(x,model)
            x_,y_ = np.meshgrid(np.linspace(xmin,xmax,25),np.linspace(ymin,ymax,25))
            u=[]
            p=[]
            for j in torch.flatten(torch.from_numpy(np.transpose(y_)[0])):
                for q in torch.flatten(torch.from_numpy(x_[0])):
                        s=torch.tensor([[q,j]]).float().to('cuda')
                        s.requires_grad=True
                        u.append(float(model.g(s)[0][0]))
                        p.append(float(model.g(s)[0][1]))
            u=np.array(u)
            p=np.array(p)
            u.resize(25,25)
            p.resize(25,25)
            V_norm=np.sqrt(u**2 + p**2)
            plt.contourf(x_,y_,V_norm, cmap="YlGn")
            plt.quiver(x_,y_,u,p)
            model.Identity.register_forward_hook(get_activation('Identity'))
            output = model(x,t=i)
            ax.scatter(activation['Identity'][:,0].cpu().numpy(),activation['Identity'][:,1].cpu().numpy(),color=color)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_title('accuracy ='+str(score*100)+'%')
            plt.show()
    p = FuncAnimation(fig, animate_reseau, interval=200, frames=range(10))
    p.save('Penalization'+'-'+donnees[int(donnee)]+'.gif', writer='pillow')




def animation(model,x,y,methode,donnee,f):
    from matplotlib.animation import FuncAnimation
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    fig, ax = plt.subplots(figsize=(4,4))
    color= ['Blue' if l == 0. else 'red' for l in y]
    score=acc(model,x,y)
    energ=energy(model,x)
    def animate_reseau(i):
        ax.clear()
        #xmin,xmax,ymin,ymax = limit_points(x,model)
        xmin,xmax,ymin,ymax = -10,10,-10,10
        x_,y_ = np.meshgrid(np.linspace(xmin,xmax,25),np.linspace(ymin,ymax,25))
        u=[]
        p=[]
        for j in torch.flatten(torch.from_numpy(np.transpose(y_)[0])):
            for q in torch.flatten(torch.from_numpy(x_[0])):
                s=torch.tensor([[q,j]]).float().to('cuda')
                s.requires_grad=True
                u.append(float(model.g(s)[0][0]))
                p.append(float(model.g(s)[0][1]))
        u=np.array(u)
        p=np.array(p)
        u.resize(25,25)
        p.resize(25,25)
        V_norm=np.sqrt(u**2 + p**2)
        plt.contourf(x_,y_,V_norm, cmap="YlGn")
        plt.quiver(x_,y_,u,p)
        model.Identity.register_forward_hook(get_activation('Identity'))
        output = model(x,t=i)
        ax.scatter(activation['Identity'][:,0].cpu().numpy(),activation['Identity'][:,1].cpu().numpy(),color=color)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title('Pénalisation '+str(int(score*10000)/100)+'%'+'--'+str(int(energy(model,x_test)*100)/100))
        plt.show()
    p = FuncAnimation(fig, animate_reseau, interval=200, frames=range(10))
    p.save('P'+'-'+donnees[int(donnee)]+'-'+str(f)+'.gif', writer='pillow')






def metric_plot(liste1,liste2,name,mean,std):
    import matplotlib.pyplot as plt
    plt.clf()
    x = [pow(10, i) for i in range(-5,10)]
    plt.xscale('log')
    plt.plot(x,liste1,label='Penalization method')
    plt.yticks(np.arange(min(liste1)-5, max(liste1)+5, 0.3))
    plt.fill_between(x,[liste1[i]-liste2[i] for i in range(len(liste1))],[liste1[i]+liste2[i] for i in range(len(liste1))],alpha=.1)
    plt.axhline(y = mean, label='Divergence-free by construction', color = 'r', linestyle = '-')
    plt.fill_between(x,mean-std,mean+std,alpha=.1)
    plt.title(name)
    l = plt.legend(loc ='upper right')
    l.set_zorder(2.)
    plt.savefig(name+'.png')

           from matplotlib.animation import FuncAnimation
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    fig, ax = plt.subplots(figsize=(4,4))
    color= ['Blue' if l == 0. else 'red' for l in y]
    def animate_reseau(i):
            ax.clear()
            x_min, x_max = x[:, 0].min() - 12.5, x[:, 0].max() + 12.5
            y_min, y_max = x[:, 1].min() - 12.5, x[:, 1].max() + 12.5
            h = 0.1
            # Generate a grid of points with distance h between them
            plt.xlim([x_min, x_max])
            plt.ylim([y_min, y_max])
            # Plot the contour and training examples
            model.Identity.register_forward_hook(get_activation('Identity'))
            output = model(x_test,t=i)
            ax.scatter(activation['Identity'][:,0].cpu().numpy(),activation['Identity'][:,1].cpu().numpy(),color=color)
            plt.show()
    p = FuncAnimation(fig, animate_reseau, interval=200, frames=range(10))
    p.save('spiral_div.gif', writer='pillow')


