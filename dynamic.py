from divergence_free import *
import torch
import torch.nn as nn
from functorch import *
import functorch
import torch.nn.functional as FA



if torch.cuda.is_available(): 
 dev = "cuda:0" 
else: 
 dev = "cpu" 
device = torch.device(dev)



upper=torch.triu(torch.ones(2, 2))-torch.diag(torch.ones(2))
lower=-torch.transpose(upper,0,1)


matrix=upper+lower
matrix=matrix.to(torch.device("cuda:0"))


class scalar(nn.Module):
    def __init__(self,ini,mid,out):
        super(scalar, self).__init__()
        self.layer1=torch.nn.Linear(ini,mid)
        self.layer2=torch.nn.Linear(mid,mid)
        self.layer3=torch.nn.Linear(mid,out)
        self.act=torch.nn.Tanh()
    def forward(self, x):      
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        x = self.layer3(x)
        return x




class dynamic(nn.Module):
    def __init__(self,fonction):
        super(dynamic, self).__init__()
        self.fonction=fonction
    def forward(self, t, x):      
        x=self.fonction(x)
        return x       

class dynamic(nn.Module):
    def __init__(self,fonction):
        super(dynamic, self).__init__()
        self.fonction=fonction
    def forward(self, x):      
        x=self.fonction(x)
        return x       




class transformation(nn.Module):
    def __init__(self,fonction):
        super(transformation, self).__init__()
        self.fonction=fonction
    def forward(self, x): 
        #x=torch.sum(torch.stack([torch.autograd.grad(self.fonction(x[i]),x, create_graph=True)[0] for i in range(len(x))]),dim=0)
        x=torch.autograd.grad(self.fonction(x),x,grad_outputs=torch.ones_like(self.fonction(x)), create_graph=True)[0]
        #x=x.view(-1,2)
        x=x.view(-1,2,1)
        new_matrix=torch.cat([matrix]*x.size()[0])
        new_matrix=new_matrix.view(x.size()[0],2,2)     
        return torch.bmm(new_matrix,x).view(-1,2)
        







def unconstrained_dynamic():

    fonction=scalar(2,128,2)
    
    fonction=dynamic(fonction)

    fonction=fonction.to(torch.device("cuda:0"))

    return fonction



def grad_potential_dynamic():

    fonction=ConvexPotentialLayerLinear(2,2)
    
    fonction=dynamic(fonction)

    fonction=fonction.to(torch.device("cuda:0"))

    return fonction





def divfree_dynamic():
    upper=torch.triu(torch.ones(2, 2))-torch.diag(torch.ones(2))
    lower=-torch.transpose(upper,0,1)
    matrix=upper+lower
    matrix=matrix.to('cuda')
    fonction=scalar(2,128,1)
    fonction=fonction.to(torch.device("cuda:0"))
    F=transformation(fonction)
    F=F.to(torch.device("cuda:0"))
    return F



"""
ndim = 2
module = nn.Sequential(
            nn.Linear(ndim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, ndim),
        )

def build_divfree_vector_field(module,x):
    def transpose(x):
        J=torch.autograd.functional.jacobian(module,x)
        a=(J-J.T)
        return a.reshape(-1)
    liste=[]
    for i in x:
        D = i.nelement()
        torch.autograd.functional.jacobian(transpose,x).reshape(D, D, D).diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)
        liste.append(torch.trace(torch.autograd.functional.jacobian(transpose,i.view(-1,2)).reshape(D, D, D)))
        
    return torch.tensor(liste).view(-1,2)
"""        


def build_divfree_vector_field(module):
    F_fn, params = make_functional(module)
    J_fn = jacrev(F_fn, argnums=1)
    def A_fn(params, x):
        J = J_fn(params, x)
        A = J - J.T
        return A
    def A_flat_fn(params, x):
        A = A_fn(params, x)
        A_flat = A.reshape(-1)
        return A_flat
    def ddF(params, x):
        D = x.nelement()
        dA_flat = jacrev(A_flat_fn, argnums=1)(params, x)
        Jac_all = dA_flat.reshape(D, D, D)
        ddF = vmap(torch.trace)(Jac_all)
        return ddF
    return ddF, params, A_fn




def neurips_dynamic():
    bsz = 10
    ndim = 2
    module = nn.Sequential(
            nn.Linear(ndim, 3),
            nn.Tanh(),
            nn.Linear(3, 3),
            nn.Tanh(),
            nn.Linear(3, ndim),
        )
    u_fn, params, A_fn = build_divfree_vector_field(module)
    f = lambda t,x: u_fn(params, x)
    f=functorch.vmap(f)
    return f, params



def NICE():

    return None




class ConvexPotentialLayerLinear(nn.Module):
    def __init__(self, cin, cout, epsilon=1e-4):
        super(ConvexPotentialLayerLinear, self).__init__()
        self.activation = nn.ReLU(inplace=False)
        self.mat=torch.nn.Linear(2, 2, bias=True)
        #self.weights = torch.zeros(cout, cin)
        #self.bias = torch.zeros(cout)
        #self.weights = nn.Parameter(self.weights)
        #self.bias = nn.Parameter(self.bias)
    def forward(self, x):
        res = self.mat(x)
        res = self.activation(res)
        res = FA.linear(res, self.mat.weight.t())
        return res