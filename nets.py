from torch import nn
import torch
import numpy as np
import copy
from torch.autograd import Function

class Binarize (nn.Module):
    @staticmethod
    def forward(input):
        return Binarize_layer.apply(input)

class Binarize_layer(Function):
    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        return torch.sign(input)
    @staticmethod
    def backward(self,grad_output):    #straight-through
        input=self.saved_tensors[0]
        grad_output[input>1]=0
        grad_output[input<-1]=0
        return grad_output

class MLP(nn.Module):
    def __init__(self, sizes, activations='ReLU'):
        """
        builds a multi layer perceptron
        sizes : list of the size of the different layers
        act : activation function either "relu", "elu", or "soft" (softmax)
        """
        if len(sizes)< 2 :
            raise Exception("sizes argument is" +  sizes.__str__() + ' . At least two elements are needed to have the input and output sizes')
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        input_size = sizes[0]
        output_size = sizes[-1]
        self.linears = nn.ModuleList()
        self.linears_reels = []
        self.activations=[]
        for i in range(1, len(sizes)):
            linear = nn.Linear(sizes[i-1], sizes[i])
            if isinstance(activations,str):
                act=activations
            else:
                act=activations[i-1]

            activation = getattr(nn, act)()
            self.activations.append(activation)
            self.linears.append(linear)
            self.linears_reels.append([copy.deepcopy(linear.weight.data),copy.deepcopy(linear.bias.data)])


    def forward(self, x):
        x = self.flatten(x)
        for linear, activation in zip(self.linears, self.activations):
            x = linear(x)
            x=activation(x)
        return x

    def to(self, device):
        model =super(MLP, self).to(device)
        for linear_reel in model.linears_reels:
            linear_reel[0] = linear_reel[0].to(device)
            linear_reel[1] = linear_reel[1].to(device)
        return model



class BinaryConnect(MLP):
    def __init__(self, sizes, activations='ReLU'):
        """
        builds a multi layer perceptron
        sizes : list of the size of the different layers
        act : activation function either "relu", "elu", or "soft" (softmax)
        """
        if len(sizes)< 2 :
            raise Exception("sizes argument is" +  sizes.__str__() + ' . At least two elements are needed to have the input and output sizes')
        super(BinaryConnect, self).__init__(sizes, activations)
        for i,linear in enumerate(self.linears):
            if i!=len(self.linears)-1 or len(self.linears)==1:
                linear.weight.data = np.sign(linear.weight.data)
                linear.bias.data = np.sign(linear.bias.data)

class BinaryNetwork(MLP):
    def __init__(self, sizes, activations='ReLU'):
        """
        builds a multi layer perceptron
        sizes : list of the size of the different layers
        act : activation function either "relu", "elu", or "soft" (softmax)
        """
        if len(sizes)< 2 :
            raise Exception("sizes argument is" +  sizes.__str__() + ' . At least two elements are needed to have the input and output sizes')
        super(BinaryNetwork, self).__init__(sizes, activations)
        self.binarize=Binarize()
        for i,linear in enumerate(self.linears):
            if i!=len(self.linears)-1:
                linear.weight.data = np.sign(linear.weight.data)
                linear.bias.data = np.sign(linear.bias.data)

    def forward(self, x):
        x = self.flatten(x)
        for i, (linear, activation) in enumerate(zip(self.linears, self.activations)):
            x = linear(x)
            if i != len(self.linears)-1:
                x = self.binarize(x)
            else:
                x = activation(x) 
        return x


loss = nn.MSELoss()
def my_mse_loss(x,y):
    mse_loss = nn.MSELoss()
    y = y.reshape((y.shape[0],1))
    y_onehot = torch.FloatTensor(x.shape[0], x.shape[1]).to(y.device)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return mse_loss(x, y_onehot)

def evaluate(dataloader, model, loss_fn):
    device = next(model.parameters()).device
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    return test_loss, correct

def eval_(X, y, model, loss_fn):
    with torch.no_grad():
        pred = model(X)
        test_loss = loss_fn(pred, y).item()
        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
    return test_loss, correct
