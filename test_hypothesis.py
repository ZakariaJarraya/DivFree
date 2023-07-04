from prompt_toolkit import prompt
from sklearn.model_selection import train_test_split
from numpy import pi
import numpy as np
from metric import *
from model_ import *
from dynamic import *
import sklearn
from nice.models import NICEModel
from sklearn.preprocessing import StandardScaler








torch.manual_seed(0)


methodes=["Baseline","DivFree","Neural Conservation Law","Nice"]

for i in range(len(methodes)):
    print(str(i) + ' - ' + methodes[i])

statement=True
while statement:
    methode = prompt('choisir le numéro de la méthode: ')
    statement = methode in range(len(methode))


donnees=["moons","spiral","gaussian"]

for i in range(len(donnees)):
    print(str(i) + ' - ' + donnees[i])




statement=True
while statement:
    donnee = prompt('choisir le numéro du jeu: ')
    statement = donnee in range(len(donnees))    



def prepare_data(i,size=300):
    if i==0:
        X, Y = sklearn.datasets.make_moons(n_samples=size, noise=0.1)
        X = X.astype("float32")
        X = X * 2 + np.array([-1, -0.2])
        scaler=StandardScaler()
        scaler.fit(X)
        X = X.astype("float32")
        #X=scaler.transform(X)
        X=torch.tensor(X).view(-1,2)
        X=X.to('cuda')
        X.requires_grad=True
        Y=torch.tensor(Y).view(-1,1)
        Y=Y.to('cuda')
        return X,Y
    elif i==1:
        theta = np.sqrt(np.random.rand(size))*2*pi # np.linspace(0,2*pi,100)
        r_a = 2*theta + pi
        data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
        x_a = data_a + np.random.randn(size,2)
        r_b = -2*theta - pi
        data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
        x_b = data_b + np.random.randn(size,2)
        res_a = np.append(x_a, np.zeros((size,1)), axis=1)
        res_b = np.append(x_b, np.ones((size,1)), axis=1)
        res = np.append(res_a, res_b, axis=0)
        np.random.shuffle(res)
        X,Y=res[:,:2],res[:,2]
        scaler=StandardScaler()
        scaler.fit(X)
        #X=scaler.transform(X)
        X = X.astype("float32")
        X=torch.tensor(X).view(-1,2)
        Y=torch.tensor(Y).view(-1,1)
        X=X.to('cuda')
        Y=Y.to('cuda')
        X.requires_grad=True
        return X,Y
    else:
        mean1=torch.tensor([[3.,0.]])
        mean3=torch.tensor([[0.,0.]])
        covariance1=torch.tensor([[[1.,0.],[0.,1.]]])
        covariance3=torch.tensor([[[1.,0.],[0.,1.]]])
        dist1=torch.distributions.multivariate_normal.MultivariateNormal(mean1, covariance1)
        dist3=torch.distributions.multivariate_normal.MultivariateNormal(mean3, covariance3)
        gauss1=[dist1.sample() for i in range(size)]
        gauss1=torch.stack(gauss1)
        gauss1=gauss1.view(-1,2)
        gauss3=[dist3.sample() for i in range(size)]
        gauss3=torch.stack(gauss3)
        gauss3=gauss3.view(-1,2)
        X=torch.cat((gauss1,gauss3), dim=0)
        Y=[torch.tensor([1.]) for i in range(size)]+[torch.tensor([0.]) for i in range(size)]
        Y=torch.stack(Y)
        Y=Y.view(-1,1)
        r=torch.randperm(len(X))
        X=X[r]
        Y=Y[r]
        #X=X.to(torch.device("cuda:0"))
        X.requires_grad=True
        Y=Y.to(torch.device("cuda:0"))
        return   X,Y








x,y = prepare_data(int(donnee),10)
#x=x.detach().cpu()
#y=y.detach().cpu()

x_train, x_test, y_train, y_test  =  train_test_split(x,y ,random_state=104, test_size=0.25, shuffle=True)



accuracy=[]
var_inter=[]
var_intra=[]
distortion=[]
jacob=[]
adv=[]
energ=[]



def model_init(methode):
    if methode==0:
        return ResNet(unconstrained_dynamic(),1,0.1)
    elif methode == 1:
        return ResNet(divfree_dynamic(),1,0.1)
    elif methode== 2:
        return ResNet(neurips_dynamic(),1,0.1)
    else:
        return NICEModel(2,3,3)



def image(x):
    return last_layer(model,x)

def jac(model,x):
        output=last_layer(model,x)
        total=0
        for i in range(len(output)):
            total+=torch.norm(torch.autograd.functional.jacobian(image,output[i]))
        return total/len(output)    


model=model_init(int(methode))
model=model.to('cuda')
train(model,x_train,y_train,100)

   
    



"""


for i in range(50):
    torch.manual_seed(i) 
    torch.cuda.manual_seed(i)
    model=model_init(int(methode))
    model=model.to('cuda')
    train(model,x_train,y_train,100)
    accuracy.append(acc(model,x_test,y_test))
    var_inter.append(variance_inter(last_layer(model,x_test),y_test).item())
    var_intra.append(variance_intra(last_layer(model,x_test),y_test).item())
    distortion.append(lq_distortion(model,x_test).item())
    jacob.append(jac(model,x_test).item())
    adv.append(acc(model,test(model,x_test,y_test,0.1),y_test))
    #total=0
    #for j in range(10):
        #t=torch.norm(model.g(rk4_solver(model.g, x_test, 0, j*0.1, 0.1),rk4_solver(model.g, x_test, 0, j*0.1, 0.1)),dim=1)
        #t=torch.pow(t,2)
        #t=torch.sum(t,0)
    energ.append(energy(model,x))    
    torch.cuda.empty_cache()


"""
