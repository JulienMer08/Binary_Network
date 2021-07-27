from abc import ABC, abstractmethod
import torch
import numpy as np



def train_1_epoch_small_nei(dataloader, model, loss_fn, **kwargs):
    """
    either gradient or mcmc are used, depending on the arguments in kwargs
    using a small neighborood each time
    """
    if 'iter_mcmc' in kwargs:
        acceptance_ratio = 0.
    for batch, (X, y) in enumerate(dataloader):
        if 'iter_mcmc' in kwargs:
            iter_mcmc, lamb, proposal, prior= kwargs['iter_mcmc'], kwargs['lamb'], kwargs['proposal'], kwargs['prior']
            acceptance_ratio += mcmc_small_nei(X, y, model, loss_fn, proposal, prior=prior, lamb=lamb, iter_mcmc=iter_mcmc)
        else:
            lr = kwargs['lr']
            gradient(X, y, model, loss_fn, lr=lr)
    if 'iter_mcmc' in kwargs:
        return acceptance_ratio / (batch+1)
    else:
        return 0

class Optimizer(ABC):
    def __init__(self, data_points_max = 1000000000):
        """
        number of data points to used in the dataset
        """
        self.data_points_max = data_points_max

    def train_1_epoch(self, dataloader, model, loss_fn):
        num_items_read = 0
        # attempting to guess the device on the model.
        device = next(model.parameters()).device
        for batch, (X, y) in enumerate(dataloader):
            if self.data_points_max <= num_items_read:
                break
            X = X[:min(self.data_points_max - num_items_read, X.shape[0])]
            y = y[:min(self.data_points_max - num_items_read, X.shape[0])]
            num_items_read = min(self.data_points_max, num_items_read + X.shape[0])
            X = X.to(device)
            y = y.to(device)
            self.train_1_batch(X, y, model, loss_fn)

    @abstractmethod
    def train_1_batch(self, X, y, model):
        pass


class GradientOptimizer(Optimizer):
    def __init__(self, data_points_max = 1000000000, lr=0.001):
        super(GradientOptimizer, self).__init__(data_points_max = 1000000000)
        #self.lr = lr


    def train_1_batch(self, X, y, model, loss_fn):
        """
        SGD optimization
        inputs:
        lr : learning rate
        """
        pred = model(X)
        los = loss_fn(pred, y)
        gg = torch.autograd.grad(los, model.parameters(), retain_graph=True)
        for i,linear_reels in enumerate(model.linears_reels):
            linear_reels[0] -= gg[2*i]*self.lr
            linear_reels[1] -= gg[2*i+1]*self.lr


class LinearModel_GradientOptimizer(GradientOptimizer):
    def __init__(self, data_points_max = 1000000000, lr=0.001):
        super(LinearModel_GradientOptimizer, self).__init__(data_points_max = 1000000000, lr=0.001)
        self.lr=lr


    def train_1_batch(self, X, y, model, loss_fn):
        super(LinearModel_GradientOptimizer,self).train_1_batch(X,y,model,loss_fn)
        for i,linear_reels in enumerate(model.linears_reels):
            model.linears[i].weight.data = torch.sign(linear_reels[0])
            model.linears[i].bias.data = torch.sign(linear_reels[1])

            torch.clip(linear_reels[0],-1,1, out=linear_reels[0])
            torch.clip(linear_reels[1],-1,1, out=linear_reels[1])

class PerceptronModel_GradientOptimizer(GradientOptimizer):
    def __init__(self, data_points_max = 1000000000, lr=0.001):
        super(PerceptronModel_GradientOptimizer, self).__init__(data_points_max = 1000000000, lr=0.001)
        self.lr=lr

    def train_1_batch(self, X, y, model, loss_fn):
        super(PerceptronModel_GradientOptimizer,self).train_1_batch(X,y,model,loss_fn)
        for i,linear_reels in enumerate(model.linears_reels):
            if i!= len(model.linears_reels)-1:
                model.linears[i].weight.data = torch.sign(linear_reels[0])
                model.linears[i].bias.data = torch.sign(linear_reels[1])
            else:
                model.linears[i].weight.data = linear_reels[0]
                model.linears[i].bias.data = linear_reels[1]

            torch.clip(linear_reels[0],-1,1, out=linear_reels[0])
            torch.clip(linear_reels[1],-1,1, out=linear_reels[1])
