import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import sklearn.datasets
from math import pi
import numpy as np
from model_ import *
from dynamic import *
from metric import *
from sklearn import preprocessing
import argparse



random.seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



torch.manual_seed(0)

methodes=['Baseline','DivFree','Lyapunov','Penalizaion']

donnees=['Moons','Spirals','Gaussian']




##saisir les choix

print('Choisir le numéro du jeu de donnée:')

for i in range(len(donnees)):
    print(str(i)+')-'+ ' '+ donnees[i])

donnee=input('')    


print("Choisir le numéro de l'algorithme:")

for i in range(len(methodes)):
    print(str(i)+')-'+ ' '+ methodes[i])

methode=input('')   

donnee=int(donnee)
methode=int(methode)


penalization=(methode==3)
alpha=0
if penalization:
    alpha=int(input('saisir le coefficient de pénalisation'))





##creation des donnees



def create_data(donnee,taille):
    if donnee==0:
        x, y = sklearn.datasets.make_moons(n_samples=taille, shuffle=True, noise=None, random_state=1)
        scaler = preprocessing.StandardScaler().fit(x)
        x = scaler.transform(x)
        x, y = torch.tensor(x), torch.tensor(y)
        x.requires_grad=True
        x, y = x.to(device).float(), y.to(device)
        return x, y
    elif donnee==1:
        N = int(taille/2)
        theta = 1.*np.sqrt(np.random.rand(N))*2*pi # np.linspace(0,2*pi,100)
        r_a = 2*theta + pi
        data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
        x_a = data_a + np.random.randn(N,2)
        x_a=torch.tensor(x_a)
        r_b = -2*theta - pi
        data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
        x_b = data_b + np.random.randn(N,2)
        x_b=torch.tensor(x_b)
        x=torch.cat((x_a,x_b),0)
        scaler = preprocessing.StandardScaler().fit(x)
        x = scaler.transform(x)
        x=torch.tensor(x)
        x.requires_grad=True
        l1=torch.tensor([0. for i in range(N)])
        l2=torch.tensor([1. for i in range(N)])
        y=torch.cat((l1,l2),dim=0).view(-1,1)
        x, y = x.to(device).float(), y.to(device)
        return x, y    




x, y = create_data(donnee,1000)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


def g(methode):
    if int(methode)==0 or int(methode)==3:
        return unconstrained_dynamic()
    elif int(methode)==1:
        return divfree_dynamic()
    elif int(methode)==2:
        return grad_potential_dynamic()







def create_spiral(taille,longueur):
            random.seed(1)
            np.random.seed(seed=1)
            N = int(taille/2)
            theta = longueur*np.sqrt(np.random.rand(N))*2*pi # np.linspace(0,2*pi,100)
            r_a = 2*theta + pi
            data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
            x_a = data_a + np.random.randn(N,2)
            x_a=torch.tensor(x_a)
            r_b = -2*theta - pi
            data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
            x_b = data_b + np.random.randn(N,2)
            x_b=torch.tensor(x_b)
            x=torch.cat((x_a,x_b),0)
            scaler = preprocessing.StandardScaler().fit(x)
            x = scaler.transform(x)
            x=torch.tensor(x)
            x.requires_grad=True
            l1=torch.tensor([0. for i in range(N)])
            l2=torch.tensor([1. for i in range(N)])
            y=torch.cat((l1,l2),dim=0).view(-1,1)
            x, y = x.to(device).float(), y.to(device)
            return x,y







def animate_reseau(i):
        x , y = create_spiral(1000*(2-(i/10)),2-(i/10))
        a, x, b, y=  train_test_split(x, y, test_size=0.25, random_state=42)
        network=torch.load('/home/zakaria/Bureau/projet-lyapunov/spiral-unroll/'+methodes[int(methode)]+'-'+donnees[int(donnee)]+'-'+str(2-(i/10))+'-'+str(0))
        x_min, x_max = -3, 3
        y_min, y_max = -3, 3
        h = 0.1
        xx,yy=np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole gid
        f=torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float().to(device)
        f.requires_grad=True
        Z = network(f)[1][:,0]   
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        Z=Z.detach().cpu().numpy()
        x=x.detach().cpu().numpy()
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(x[:, 0], x[:, 1], c=y.detach().cpu().numpy(), cmap=plt.cm.binary)  
        plt.show()







Score=[]
V_inter=[]
V_intra=[]
Distortion=[]
Adv=[]
J_norm=[]
Energy=[]
for i in range(20):
        print(i)
        print(alpha)
        torch.manual_seed(i)
        model=ResNet(g(methode), 10, 0.1)
        model=model.to('cuda')
        train(model,penalization,x_train,y_train,alpha,100)
        Score.append(acc(model,x_test,y_test))
        V_inter.append(variance_inter(last_layer(model,x_test),y_test).item())
        V_intra.append(variance_intra(last_layer(model,x_test),y_test).item())
        Distortion.append(lq_distortion(model,x_test).item())
        Adv.append(acc(model,test(model, x_test, y_test, 0.1),y_test))
        Energy.append(energy(model,x_test))
        #torch.save(model, '/home/zakaria/Bureau/projet-lyapunov/lambda-models/'+'-'+methodes[int(methode)]+'-'+donnees[int(donnee)]+'-'+str(i)+'-'+str(alpha))


print('Accuracy : ' + str(np.mean(Score)) + ' +/- ' + str(np.std(Score)))
print('Inter-class variance : ' + str(np.mean(V_inter)) + ' +/- ' + str(np.std(V_inter)))
print('Intra-class variance : ' + str(np.mean(V_intra)) + ' +/- ' + str(np.std(V_intra)))
print('Distortion : ' + str(np.mean(Distortion)) + ' +/- ' + str(np.std(Distortion)))
print('Accuracy under adversarial attack: ' + str(np.mean(Adv)) + ' +/- ' + str(np.std(Adv)))
print('Energy : ' + str(np.mean(Energy)) + ' +/- ' + str(np.std(Energy)))






"""      
def create_spiral(taille,longueur):
            random.seed(1)
            N = int(taille/2)
            theta = longueur*np.sqrt(np.random.rand(N))*2*pi # np.linspace(0,2*pi,100)
            r_a = 2*theta + pi
            data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
            x_a = data_a + np.random.randn(N,2)
            x_a=torch.tensor(x_a)
            r_b = -2*theta - pi
            data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
            x_b = data_b + np.random.randn(N,2)
            x_b=torch.tensor(x_b)
            x=torch.cat((x_a,x_b),0)
            scaler = preprocessing.StandardScaler().fit(x)
            x = scaler.transform(x)
            x=torch.tensor(x)
            x.requires_grad=True
            l1=torch.tensor([0. for i in range(N)])
            l2=torch.tensor([1. for i in range(N)])
            y=torch.cat((l1,l2),dim=0).view(-1,1)
            x, y = x.to(device).float(), y.to(device)
            return x,y



            
Score=[]
V_inter=[]
V_intra=[]
Distortion=[]
Adv=[]
J_norm=[]
Energy=[]
Free=[]


Score1=[]
V_inter1=[]
V_intra1=[]
Distortion1=[]
Adv1=[]
J_norm1=[]
Energy1=[]
Free1=[]





for i in range(1):
    #alpha=pow(10,i)
    print(i)
    Score_=[]
    V_inter_=[]
    V_intra_=[]
    Distortion_=[]
    Adv_=[]
    J_norm_=[]
    Energy_=[]
    Free_=[]
    for j in range(20):
        print(j)
        torch.manual_seed(j)
        model=ResNet(g(methode), 10, 0.1)
        model=model.to('cuda')
        train(model,x_train,y_train)
        #model=torch.load('/home/zakaria/Bureau/projet-lyapunov/lambda-models/'+'-'+methodes[int(methode)]+'-'+donnees[int(donnee)]+'-'+str(j)+'-'+str(alpha))
        #Free_.append(div(model.g,x_test).mean().item())
        Score_.append(acc(model,x_test,y_test))
        V_inter_.append(variance_inter(last_layer(model,x_test),y_test).item())
        V_intra_.append(variance_intra(last_layer(model,x_test),y_test).item())
        Distortion_.append(lq_distortion(model,x_test).item())
        Adv_.append(acc(model,test(model, x_test, y_test, 0.1),y_test))
        Energy_.append(energy(model,x_test))
    Score.append(np.mean(Score_))
    Score1.append(np.std(Score_))    
    V_inter.append(np.mean(V_inter_))
    V_inter1.append(np.std(V_inter_)) 
    V_intra.append(np.mean(V_intra_))
    V_intra1.append(np.std(V_intra_)) 
    Distortion.append(np.mean(Distortion_))
    Distortion1.append(np.std(Distortion_)) 
    Adv.append(np.mean(Adv_))
    Adv1.append(np.std(Adv_)) 
    Free.append(np.mean(Free_))
    Free1.append(np.std(Free_)) 
    Energy.append(np.mean(Energy_))
    Energy1.append(np.std(Energy_)) 
        


for i in range(10):
    print(i)
    x , y = create_spiral(1000*(2-(i/10)),2-(i/10))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    model=torch.load('/home/zakaria/Bureau/projet-lyapunov/spiral-unroll/'+methodes[int(methode)]+'-'+donnees[int(donnee)]+'-'+str(2-(i/10))+'-'+str(j))
    Score_=[]
    V_inter_=[]
    V_intra_=[]
    Distortion_=[]
    Adv_=[]
    J_norm_=[]
    Energy_=[]
    for j in range(20):
        print(j)
        torch.manual_seed(j)
        torch.load(model, '/home/zakaria/Bureau/projet-lyapunov//'+methodes[int(methode)]+'-'+donnees[int(donnee)]+'-'+str(2-(i/10))+'-'+str(j))



        








def animation(model,x,y,methode,donnee,f):
    from matplotlib.animation import FuncAnimation
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    fig, ax = plt.subplots(figsize=(4,4))
    color= ['Blue' if l == 0. else 'red' for l in y]
    def animate_reseau(i):
            x , y = create_spiral(1000*(2-(i/10)),2-(i/10))
            a, x, b, y= train_test_split(x, y, test_size=0.25, random_state=42)
            network=torch.load('/home/zakaria/Bureau/projet-lyapunov/spiral-unroll/'+methodes[int(methode)]+'-'+donnees[int(donnee)]+'-'+str(2-(i/10))+'-'+str(0))
            x_min, x_max = -3, 3
            y_min, y_max = -3, 3
            h = 0.1
            xx,yy=np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            # Predict the function value for the whole gid
            f=torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float().to(device)
            f.requires_grad=True
            Z = network(f)[1][:,0]   
            Z = Z.reshape(xx.shape)
            # Plot the contour and training examples
            Z=Z.detach().cpu().numpy()
            x=x.detach().cpu().numpy()
            plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
            plt.scatter(x[:, 0], x[:, 1], c=y.detach().cpu().numpy(), cmap=plt.cm.binary)  
            plt.show()
    p = FuncAnimation(fig, animate_reseau, interval=200, frames=range(10))
    p.save('decision'+'-'+donnees[int(donnee)]+'-'+str(0)+'.gif', writer='pillow')      



    
model=ResNet(g(methode), 10, 0.1)
model=model.to('cuda')
train(model,x_train,y_train)







def plot_decision_boundary(network,x,y,i):
    fig = plt.figure()
    import matplotlib as mpl
    plt.clf()
    x_min, x_max = -3, 3
    y_min, y_max = -3, 3
    color= ['Blue' if l == 0. else 'red' for l in y]
    h = 0.1
    xx,yy=np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    f=torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float().to(device)
    f.requires_grad=True
    Z = network(f)[1][:,0]   
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    Z=Z.detach().cpu().numpy()
    x=x.detach().cpu().numpy()
    cmap = mpl.cm.RdBu
    bounds = [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))    
    plt.scatter(x[:, 0], x[:, 1], cmap=plt.cm.binary)     
    plt.savefig('/home/zakaria/Bureau/projet-lyapunov/spiral-unroll/'+str(i)+'.png')     



"""



















