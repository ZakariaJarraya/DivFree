import torch
import random
import math
import torch.nn as nn
import model_

def acc(network, liste, Y):
    network.eval()
    acc=0
    indice=[]
    for i in range(len(Y)):
        if torch.argmax(network(liste[i].view(-1,2))[1])==Y[i]:
                acc+=1       
        else:
            pass
            indice.append(i)        
    return  (acc/len(Y))   




def variance_inter(X,Y):
    cond1=Y==1
    cond0=Y==0
    X0=X[cond0.nonzero(as_tuple=True)[0]]
    X1=X[cond1.nonzero(as_tuple=True)[0]]
    X1=X1.view(-1,2)
    X0=X0.view(-1,2)
    X=X.view(-1,2)
    X_mean=torch.mean(X,dim=0)
    X0_mean=torch.mean(X0,dim=0)
    X1_mean=torch.mean(X1,dim=0)
    return torch.cdist(X0_mean.view(-1,2), X_mean.view(-1,2), p=2.0)*(len(X0)/len(X))+torch.cdist(X1_mean.view(-1,2), X_mean.view(-1,2), p=2.0)*(len(X1)/len(X))




def variance_intra(X,Y):
    cond1=Y==1
    cond0=Y==0
    X0=X[cond0.nonzero(as_tuple=True)[0]]
    X1=X[cond1.nonzero(as_tuple=True)[0]]
    X1=X1.view(-1,2)
    X0=X0.view(-1,2)
    X=X.view(-1,2)
    return torch.var(X0)*(len(X0)/len(X))+torch.var(X1)*(len(X1)/len(X))





activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def last_layer(model,x):
    if isinstance(model,model_.ResNet):
        return model_.rk4_solver(model.g, x.view(-1,2), 0, 1, 0.1)
    else:
        model.eval()
        y=model.layer1(x.view(-1,2))
        y=model.layer2(y)
        y=model.layer3(y)
        y=model.layer4(y)   
        y=torch.matmul(y, torch.diag(torch.exp(model.scaling_diag))) 
        return y




def ratio(x,y,model):
    data=torch.cat((x,y),0)
    data=data.view(-1,2)
    output=last_layer(model,data)
    numerator=torch.cdist(output[0].view(-1,2),output[1].view(-1,2))
    denumerator=torch.cdist(data[0].view(-1,2),data[1].view(-1,2))
    return numerator/denumerator





def lq_distortion(model,x):
    mean=0
    length=len(x)
    for i in range(length):
        indice1=random.randint(0,length-1)
        indice2=random.randint(0,length-1)
        a=x[indice1]
        b=x[indice2]
        f=torch.pow(ratio(a,b,model),2)
        if math.isnan(f):
            break
        else:
            mean+=f 
    mean=mean/length
    mean=torch.pow(mean,0.5)
    return mean




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
    model.eval()
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for i in range(len(Y)):



        # Set requires_grad attribute of tensor. Important for Attack
        s=X[i].clone().detach().requires_grad_(True)
        #s=s.view(-1,d)

        #s=s.view(-1,d)
        # Forward pass the data through the model
        output = model(s.view(-1,2))[1]
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

    return torch.stack(adv_examples).view(-1,2)



     

def energy(model,x):
    total=0
    if isinstance(model,model_.ResNet):
        for i in range(10):
            #t=torch.norm(model.g(model_.rk4_solver(model.g, x, 0, i*0.1, 0.1),model_.rk4_solver(model.g, x, 0, i*0.1, 0.1)),dim=1)
            t=torch.norm(model.g(model_.rk4_solver(model.g, x, 0, i*0.1, 0.1)),dim=1)
            t=torch.pow(t,2)
            t=torch.sum(t,0)
            total+=t.item()
        return total/(len(x)*10)   
    else:
        return 0

            
        
