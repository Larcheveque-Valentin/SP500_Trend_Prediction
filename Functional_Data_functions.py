

# -*- coding: utf-8 -*-
"""
April - August 2023
Hyper Parameter Search for Functionnal Convolution 

@author:Valentin Larchevêque


"""

# Import modules
import inspect
import gc
import random
import torch
import torch.nn.init as init
import torch.nn.functional as F
from scipy.stats import norm
import torch.nn as nn
from skfda.representation.basis import VectorValued as MultiBasis
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
import pandas as pd
import numpy as np
from numpy import *
import seaborn as sns
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
# import skfda as fda
# from skfda import representation as representation
# from skfda.exploratory.visualization import FPCAPlot
# # from skfda.exploratory.visualization import FPCAPlot
# # from skfda.preprocessing.dim_reduction import FPCA
# # from skfda.representation.basis import BSpline, Fourier, Monomial
import scipy
from scipy.interpolate import BSpline
import os
import ignite
from tqdm import tqdm
import sklearn
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
import random
from random import seed
from scipy import stats
import statistics
from statistics import stdev
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

import skfda
from skfda import FDataGrid as fd
from skfda.representation.basis import BSpline as B


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, std=0.005)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, std=0.005)
        torch.nn.init.constant_(m.bias.data, 0.0)

def from_torch_to_Datagrid(x):
    if isinstance(x,torch.Tensor):
        x_grid=fd(data_matrix=x[:,0,:].cpu(),grid_points=np.arange(x.shape[2]+1)[1:])
    elif isinstance(x,skfda.representation.grid.FDataGrid):
        x_grid=x
    else:
        raise ValueError("the NN argument must be either torch.tensor or skfda.representation.grid.FDataGrid")
    
    return x_grid
    


def conv_block_out(kernel_size,stride,padding,dilation,n):
    return ((n+2*padding-dilation*(kernel_size-1)-1)//stride)+1




        


class Smoothing_method:
    def __init__(self,knots_or_basis="knots",Mode="Smooth",basis_type="Bspline",order=3,t0=1,T=12,period=2*pi,n_basis=13,n_knots=6):
        self.Mode=Mode
        self.basis_type=basis_type
        # self.interpolation_order=interpolation_order
        self.order=order
        self.knots_or_basis=knots_or_basis

        self.knots=np.linspace(t0,T,n_knots)
        self.n_knots=n_knots
        self.period=period
        self.n_basis=n_basis
            
    def smoothing(self):
        if 'inter' in self.Mode:
            # interpolation=skfda.representation.interpolation.SplineInterpolation(interpolation_order=self.interpolation_order)             
            smooth_meth= skfda.representation.interpolation.SplineInterpolation(interpolation_order=self.order)             
        else:
            if "knots" in self.knots_or_basis:  
                if ("spline" in self.basis_type) or ("Bsp" in self.basis_type) or ("spl" in self.basis_type):
                    smooth_meth= B(knots=self.knots,order=self.order)
                if self.basis_type=="Fourier":
                    smooth_meth=skfda.representation.basis.FourierBasis(domain_range=[min(self.knots),max(self.knots)],period=self.period)
            if "basis" in self.knots_or_basis:
                if ("spline" in self.basis_type) or ("Bsp" in self.basis_type) or ("spbasis" in self.basis_type):
                    smooth_meth= B(n_basis=self.n_basis,order=self.order)
                if ("fourier" in self.basis_type)or ("fourrier" in self.basis_type) or ("Fourier" in self.basis_type) or ("four" in self.basis_type):
                    smooth_meth=skfda.representation.basis.FourierBasis(n_basis=self.n_basis,period=self.period)
        return smooth_meth

def Granulator(x,granulation,channels=1,basis=B(knots=linspace(0,1,15),order=4),derivative=[0],mode="Bspline"):
        
        x_grid=from_torch_to_Datagrid(x=x)
        if "inter" not in mode:     
            Recons_train=torch.zeros([x_grid.shape[0],channels*len(derivative),granulation]).float().cuda()
            i=0
            for channel in range(channels):
                for deriv in derivative:
                    eval_points=linspace(1,x_grid.grid_points[0].shape[0],granulation)
                    coefs_torch=torch.from_numpy(x_grid.to_basis(basis).coefficients).float().cuda()
                    basis_eval=basis(eval_points=eval_points,derivative=deriv)
                    basis_fc = torch.from_numpy(basis_eval).float().cuda()
                    # coefs_torch=torch.from_numpy(coefs).float().cuda()        
                    Recons_train[:,i,:]=torch.matmul(coefs_torch,basis_fc[:,:,channel])
                    i+=1    
        
        else:
            x_grid.interpolation=skfda.representation.interpolation.SplineInterpolation(basis.order)
            eval_points=linspace(1,x_grid.grid_points[0].shape[0],granulation)
            Recons_train=x_grid.interpolation._evaluate(fdata=x_grid,eval_points=eval_points)[:,:,0]
            
            Recons_train=torch.tensor(Recons_train).reshape(Recons_train.shape[0],channels,Recons_train.shape[1])
        
        return Recons_train.float().cuda()


class TSConv1d(nn.Module):
    def __init__(self,kernel_size,out_channels,dilation,granulation,knots=linspace(0,1,15),order=4,channels=1,derivative=[0],mode="Bspline",padding=0,stride=1,hyperparameter=None):
            
        super(TSConv1d,self).__init__()
        
        self.basis=B(knots=knots,order=order)
        self.granulation
        self.channels=channels
        self.derivative=derivative
        self.mode=mode
        self.convlayer=nn.Sequential(nn.Conv1d(in_channels=channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilatation,bias=True,groups=1))
                    
    def forward(self,x):
        Granulated_x_train=Granulator(x,self.granulation,basis=self.basis,channels=self.channels,derivative=self.derivative,mode=self.mode)
        Conv_out=self.convlayer(Granulated_x_train)
        return Conv_out

class TSConv1d_hyperparam(nn.Module):
    def __init__(self,hyperparameter):
            
        super(TSConv1d_hyperparam,self).__init__()
        self.hyperparameter=hyperparameter
        self.basis=hyperparameter.basis
        self.convlayer=nn.Sequential(
            nn.Conv1d(in_channels=hyperparameter.n_channel*len(hyperparameter.derivative),out_channels=hyperparameter.n_conv_in,kernel_size=hyperparameter.kernel_size_1,stride=hyperparameter.stride_1,padding=hyperparameter.padding_1,dilation=hyperparameter.dilation_1,bias=True,groups=1),
            )



    def forward(self,x):
            Granulated_x_train=Granulator(x,self.hyperparameter.granulation,basis=self.hyperparameter.basis,channels=self.hyperparameter.n_channel,derivative=self.hyperparameter.derivative,mode=self.hyperparameter.Smoothing_mode)
            Conv_out=self.convlayer(Granulated_x_train)
            return Conv_out
                        





class HyperParameters:
    def __init__(self,basis=skfda.representation.basis.VectorValuedBasis([Smoothing_method().smoothing()
    ,
    Smoothing_method().smoothing(),]),Smoothing_mode="smooth",batch_size=30, n_epochs=25, granulation=2000,
                 n_conv_in=32, n_conv_in2=512, n_conv_in3=256,n_conv_out=64, n_Flat_out=256,
                 stride_1=1, stride_2=1, stride_3=1,
                 stride_pool_1=2, stride_pool_2=2, stride_pool_3=1,
                 kernel_size_1=7, kernel_size_2=4, kernel_size_3=3,
                 kernel_size_pool_1=3, kernel_size_pool_2=3, kernel_size_pool_3=2,
                 dilation_1=1, dilation_2=1, dilation_3=1,
                 dilation_pool_1=1, dilation_pool_2=1, dilation_pool_3=1,
                 padding_1=2, padding_2=2, padding_3=2,derivative=[0],
                 padding_pool_1=0, padding_pool_2=0, padding_pool_3=0,
                 opt="Adam", lr=0.00089, loss=nn.CrossEntropyLoss(),
                 norm_type=inf,
                 activation=nn.Identity(),
                 negative_slope=0.17,
                 n_channel=1,
                 bidirectional=False,
                 n_knots=6,
                 foreach=True,
                 decay=0.00006,fused=True,
                 dampening=0.0001,
                 nesterov=False,
                 momentum=0.000001,eps=1e-9,
                 rho=0.95,
                 lr_decay=0.0007,
                 dropout=0.15,betas=[0.85,.99],order=4):
        self.Smoothing_mode=Smoothing_mode
        self.order=order
        self.DropOut=dropout
        self.lr_decay=lr_decay
        self.betas=betas
        self.eps=eps
        self.rho=rho
        self.derivative=derivative
        self.momentum=momentum
        self.nesterov=nesterov
        self.foreach=foreach
        self.decay=decay
        self.fused=fused
        self.dampening=dampening
        self.bidirectional=bidirectional

        self.n_epochs = n_epochs
        self.batch_size=batch_size
        self.activation=activation
        self.n_conv_out=n_conv_out
        self.n_channel=n_channel
        self.n_knots=n_knots
        self.granulation = granulation
        self.n_conv_in = n_conv_in
        self.n_conv_in2 = n_conv_in2
        self.n_conv_in3 = n_conv_in3
        self.n_Flat_out = n_Flat_out
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.stride_3 = stride_3
        self.stride_pool_1 = stride_pool_1
        self.stride_pool_2 = stride_pool_2
        self.stride_pool_3 = stride_pool_3
        self.kernel_size_1 = kernel_size_1
        self.kernel_size_2 = kernel_size_2
        self.kernel_size_3 = kernel_size_3
        self.kernel_size_pool_1 = kernel_size_pool_1
        self.kernel_size_pool_2 = kernel_size_pool_2
        self.kernel_size_pool_3 = kernel_size_pool_3
        self.dilation_1 = dilation_1
        self.dilation_2 = dilation_2
        self.dilation_3 = dilation_3
        self.dilation_pool_1 = dilation_pool_1
        self.dilation_pool_2 = dilation_pool_2
        self.dilation_pool_3 = dilation_pool_3
        self.padding_1 = padding_1
        self.padding_2 = padding_2
        self.padding_3 = padding_3
        self.padding_pool_1 = padding_pool_1
        self.padding_pool_2 = padding_pool_2
        self.padding_pool_3 = padding_pool_3
        self.opt = opt
        self.lr = lr
        self.loss = loss
        self.negative_slope=negative_slope
        self.norm_type=norm_type
        if n_channel!=1:
            self.basis=basis
        else:
            self.basis=basis




# def conv_total_out(hyperparams=HyperParameters()):
#     n=hyperparams.granulation
#     stride_1 = [hyperparams.stride_1,
#     hyperparams.stride_2,
#     hyperparams.stride_3,]
#     stride_pool_1 = [hyperparams.stride_pool_1,
#     hyperparams.stride_pool_2,
#     hyperparams.stride_pool_3,]
#     kernel_size_1 = [hyperparams.kernel_size_1,
#     hyperparams.kernel_size_2,
#     hyperparams.kernel_size_3,]
#     kernel_size_pool_1 = [hyperparams.kernel_size_pool_1,
#     hyperparams.kernel_size_pool_2,
#     hyperparams.kernel_size_pool_3,]
#     dilation_1 = [hyperparams.dilation_1,
#     hyperparams.dilation_2,
#     hyperparams.dilation_3,]
#     dilation_pool_1 = [hyperparams.dilation_pool_1,
#     hyperparams.dilation_pool_2,
#     hyperparams.dilation_pool_3,]
#     # basis=hyperparams.basis
#     padding_1 = [hyperparams.padding_1,
#     hyperparams.padding_2,
#     hyperparams.padding_3,]
#     padding_pool_1 = [hyperparams.padding_pool_1,
#     hyperparams.padding_pool_2,
#     hyperparams.padding_pool_3,]
    
#     k=conv_block_out(n=n,
#                         kernel_size=kernel_size_1[0],
#                         stride=stride_1[0],
#                         dilation=dilation_1[0],
#                         padding=padding_1[0],
#                         )
    
#     v=conv_block_out(n=k,
#                         kernel_size=kernel_size_1[1],
#                         stride=stride_1[1],
#                         dilation=dilation_1[1],
#                         padding=padding_1[1],
#                         )
#     j=conv_block_out(n=v,
#                         kernel_size=kernel_size_pool_1[0],
#                         stride=stride_pool_1[0],
#                         dilation=dilation_pool_1[0],
#                         padding=padding_pool_1[0],
#                         )
#     n=j
#     k=conv_block_out(n=n,
#                         kernel_size=kernel_size_1[1],
#                         stride=stride_1[1],
#                         dilation=dilation_1[1],
#                         padding=padding_1[1],
#                         )
#     v=conv_block_out(n=k,
#                         kernel_size=kernel_size_1[1],
#                         stride=stride_1[1],
#                         dilation=dilation_1[1],
#                         padding=padding_1[1],
#                         )
#     j=conv_block_out(n=v,
#                         kernel_size=kernel_size_pool_1[1],
#                         stride=stride_pool_1[1],
#                         dilation=dilation_pool_1[1],
#                         padding=padding_pool_1[1],
#                         )
#     n=j
#     for i in range(3):        
#         n=conv_block_out(n=n,
#                             kernel_size=kernel_size_1[2],
#                             stride=stride_1[2],
#                             dilation=dilation_1[2],
#                             padding=padding_1[2],
#                             )
       
#     for i in range(3):        
#         n=conv_block_out(n=n,
            
#                         kernel_size=kernel_size_pool_1[2],
#                         stride=stride_pool_1[2],
#                         dilation=dilation_pool_1[2],
#                         padding=padding_pool_1[2]
#                         )


#     return n   
def conv_block_out_lp(kernel_size,stride,padding,dilation,n):
    return (n-kernel_size)//stride+1



def conv_total_out(hyperparams=HyperParameters(),lp=False,Maxpooling=True):
    n=hyperparams.granulation
    stride_1 = [hyperparams.stride_1,
    hyperparams.stride_2,
    hyperparams.stride_3,]
    stride_pool_1 = [hyperparams.stride_pool_1,
    hyperparams.stride_pool_2,
    hyperparams.stride_pool_3,]
    kernel_size_1 = [hyperparams.kernel_size_1,
    hyperparams.kernel_size_2,
    hyperparams.kernel_size_3,]
    kernel_size_pool_1 = [hyperparams.kernel_size_pool_1,
    hyperparams.kernel_size_pool_2,
    hyperparams.kernel_size_pool_3,]
    dilation_1 = [hyperparams.dilation_1,
    hyperparams.dilation_2,
    hyperparams.dilation_3,]
    dilation_pool_1 = [hyperparams.dilation_pool_1,
    hyperparams.dilation_pool_2,
    hyperparams.dilation_pool_3,]
    # basis=hyperparams.basis
    padding_1 = [hyperparams.padding_1,
    hyperparams.padding_2,
    hyperparams.padding_3,]
    padding_pool_1 = [hyperparams.padding_pool_1,
    hyperparams.padding_pool_2,
    hyperparams.padding_pool_3,]
    
    
    for i in range(3):
        k=conv_block_out(n=n,
                        kernel_size=kernel_size_1[i],
                        stride=stride_1[i],
                        dilation=dilation_1[i],
                        padding=padding_1[i],
                        )
        
        if Maxpooling:
            j=conv_block_out(n=k,
                            kernel_size=kernel_size_pool_1[i],
                            stride=stride_pool_1[i],
                            dilation=dilation_pool_1[i],
                            padding=padding_pool_1[i],
                            )
            n=j
        elif lp:
            j=conv_block_out_lp(n=k,
                            kernel_size=kernel_size_pool_1[i],
                            stride=stride_pool_1[i],
                            dilation=dilation_pool_1[i],
                            padding=padding_pool_1[i],
                            )
            n=j
        else:
            n=k
    

    return n

def conv_total_out_une_couche(hyperparams=HyperParameters(),lp=False):
    n=hyperparams.granulation
    stride_1 = [hyperparams.stride_1,
    hyperparams.stride_2,
    hyperparams.stride_3,]
    stride_pool_1 = [hyperparams.stride_pool_1,
    hyperparams.stride_pool_2,
    hyperparams.stride_pool_3,]
    kernel_size_1 = [hyperparams.kernel_size_1,
    hyperparams.kernel_size_2,
    hyperparams.kernel_size_3,]
    kernel_size_pool_1 = [hyperparams.kernel_size_pool_1,
    hyperparams.kernel_size_pool_2,
    hyperparams.kernel_size_pool_3,]
    dilation_1 = [hyperparams.dilation_1,
    hyperparams.dilation_2,
    hyperparams.dilation_3,]
    dilation_pool_1 = [hyperparams.dilation_pool_1,
    hyperparams.dilation_pool_2,
    hyperparams.dilation_pool_3,]
    # basis=hyperparams.basis
    padding_1 = [hyperparams.padding_1,
    hyperparams.padding_2,
    hyperparams.padding_3,]
    padding_pool_1 = [hyperparams.padding_pool_1,
    hyperparams.padding_pool_2,
    hyperparams.padding_pool_3,]
    
    
    k=conv_block_out(n=n,
                    kernel_size=kernel_size_1[0],
                    stride=stride_1[0],
                    dilation=dilation_1[0],
                    padding=padding_1[0],
                    )
    if not lp:
        j=conv_block_out(n=k,
                        kernel_size=kernel_size_pool_1[0],
                        stride=stride_pool_1[0],
                        dilation=dilation_pool_1[0],
                        padding=padding_pool_1[0],
                        )
        n=j
    else:
        j=conv_block_out_lp(n=k,
                        kernel_size=kernel_size_pool_1[i],
                        stride=stride_pool_1[i],
                        dilation=dilation_pool_1[i],
                        padding=padding_pool_1[i],
                        )
        n=j
    return n
    
def conv_total_out_no_pool(hyperparams=HyperParameters()):
    n=hyperparams.granulation
    stride_1 = [hyperparams.stride_1,
    hyperparams.stride_2,
    hyperparams.stride_3,]
    stride_pool_1 = [hyperparams.stride_pool_1,
    hyperparams.stride_pool_2,
    hyperparams.stride_pool_3,]
    kernel_size_1 = [hyperparams.kernel_size_1,
    hyperparams.kernel_size_2,
    hyperparams.kernel_size_3,]
    kernel_size_pool_1 = [hyperparams.kernel_size_pool_1,
    hyperparams.kernel_size_pool_2,
    hyperparams.kernel_size_pool_3,]
    dilation_1 = [hyperparams.dilation_1,
    hyperparams.dilation_2,
    hyperparams.dilation_3,]
    dilation_pool_1 = [hyperparams.dilation_pool_1,
    hyperparams.dilation_pool_2,
    hyperparams.dilation_pool_3,]
    # basis=hyperparams.basis
    padding_1 = [hyperparams.padding_1,
    hyperparams.padding_2,
    hyperparams.padding_3,]
    padding_pool_1 = [hyperparams.padding_pool_1,
    hyperparams.padding_pool_2,
    hyperparams.padding_pool_3,]
    
    for i in range(3):
        k=conv_block_out(n=n,
                    kernel_size=kernel_size_1[i],
                    stride=stride_1[i],
                    dilation=dilation_1[i],
                    padding=padding_1[i],
                    )
        n=k

    return n









class LayerNorm(nn.Module):

    def __init__(self, d, eps=1e-6):
        super().__init__()
        # d is the normalization dimension
        self.d = d
        self.eps = eps
        self.alpha = nn.Parameter(torch.randn(d))
        self.beta = nn.Parameter(torch.randn(d))

    def forward(self, x):
        # x is a torch.Tensor
        # avg is the mean value of a layer
        avg = x.mean(dim=-1, keepdim=True)
        # std is the standard deviation of a layer (eps is added to prevent dividing by zero)
        std = x.std(dim=-1, keepdim=True) + self.eps
        return (x - avg) / std * self.alpha + self.beta



def conv_total_out_without_pooling(hyperparams=HyperParameters()):
    n=hyperparams.granulation
    stride_1 = [hyperparams.stride_1,
    hyperparams.stride_2,
    hyperparams.stride_3,]
    stride_pool_1 = [hyperparams.stride_pool_1,
    hyperparams.stride_pool_2,
    hyperparams.stride_pool_3,]
    kernel_size_1 = [hyperparams.kernel_size_1,
    hyperparams.kernel_size_2,
    hyperparams.kernel_size_3,]
    kernel_size_pool_1 = [hyperparams.kernel_size_pool_1,
    hyperparams.kernel_size_pool_2,
    hyperparams.kernel_size_pool_3,]
    dilation_1 = [hyperparams.dilation_1,
    hyperparams.dilation_2,
    hyperparams.dilation_3,]
    dilation_pool_1 = [hyperparams.dilation_pool_1,
    hyperparams.dilation_pool_2,
    hyperparams.dilation_pool_3,]
    # basis=hyperparams.basis
    padding_1 = [hyperparams.padding_1,
    hyperparams.padding_2,
    hyperparams.padding_3,]
    padding_pool_1 = [hyperparams.padding_pool_1,
    hyperparams.padding_pool_2,
    hyperparams.padding_pool_3,]
    
    
    for i in range(3):
        k=conv_block_out(n=n,
                        kernel_size=kernel_size_1[i],
                        stride=stride_1[i],
                        dilation=dilation_1[i],
                        padding=padding_1[i],
                        )
       
        j=conv_block_out(n=k,
                        kernel_size=kernel_size_pool_1[i],
                        stride=stride_pool_1[i],
                        dilation=dilation_pool_1[i],
                        padding=padding_pool_1[i],
                        )
        n=k

    return n   
def _inner_product(f1, f2, h):
    """    
    f1 - (B, J) : B functions, observed at J time points,
    f2 - (B, J) : same as f1
    h  - (J-1,1): weights used in the trapezoidal rule
    pay attention to dimension
    <f1, f2> = sum (h/2) (f1(t{j}) + f2(t{j+1}))
    """
    prod = f1 * f2 # (B, J = len(h) + 1)
    return torch.matmul((prod[:, :-1] + prod[:, 1:]), h.unsqueeze(dim=-1))/2

def _l1(f, h):
    # f dimension : ( B bases, J )
    B, J = f.size()
    return _inner_product(torch.abs(f), torch.ones((B, J)), h)
def _l2(f, h):
    # f dimension : ( B bases, J )
    # output dimension - ( B bases, 1 )
    return torch.sqrt(_inner_product(f, f, h)) 

class FeedForward(nn.Module):

    def __init__(self, in_d=1, hidden=[4,4,4], dropout=0.1, activation=F.relu):
        # in_d      : input dimension, integer
        # hidden    : hidden layer dimension, array of integers
        # dropout   : dropout probability, a float between 0.0 and 1.0
        # activation: activation function at each layer
        super().__init__()
        self.sigma = activation
        dim = [in_d] + hidden + [1]
        self.layers = nn.ModuleList([nn.Linear(dim[i-1], dim[i]) for i in range(1, len(dim))])
        self.ln = nn.ModuleList([LayerNorm(k) for k in hidden])
        self.dp = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(hidden))])

    def forward(self, t):
        for i in range(len(self.layers)-1):
            t = self.layers[i](t)
            # skipping connection
            t = t + self.ln[i](t)
            t = self.sigma(t)
            # apply dropout
            t = self.dp[i](t)
        # linear activation at the last layer
        return self.layers[-1](t)

class TSCNN(nn.Module):
    def __init__(self, hyperparams,output_size):
        super(TSCNN, self).__init__()

        self.hyperparameters=hyperparams
        self.convlayer1=nn.Sequential(
            nn.Conv1d(self.hyperparameters.n_channel*len(hyperparams.derivative),hyperparams.n_conv_in,kernel_size=hyperparams.kernel_size_1,stride=hyperparams.stride_1,padding=hyperparams.padding_1,dilation=hyperparams.dilation_1),
            nn.BatchNorm1d(hyperparams.n_conv_in),
            nn.LeakyReLU(hyperparams.negative_slope),
            nn.Dropout(p=hyperparams.DropOut),
            hyperparams.activation,
            nn.MaxPool1d(kernel_size=hyperparams.kernel_size_pool_1,stride=hyperparams.stride_pool_1,padding=hyperparams.padding_pool_1,dilation=hyperparams.dilation_pool_1),
            nn.BatchNorm1d(hyperparams.n_conv_in),
            nn.LeakyReLU(hyperparams.negative_slope),
            nn.Dropout(p=hyperparams.DropOut),
        )
        
        self.convlayer2=nn.Sequential(
            nn.Conv1d(hyperparams.n_conv_in,hyperparams.n_conv_in2,kernel_size=hyperparams.kernel_size_2,stride=hyperparams.stride_2,padding=hyperparams.padding_2,dilation=hyperparams.dilation_2),
            # nn.Conv1d(hyperparams.n_conv_in2,hyperparams.n_conv_in2,kernel_size=hyperparams.kernel_size_2,stride=hyperparams.stride_2,padding=hyperparams.padding_2,dilation=hyperparams.dilation_2),
            nn.BatchNorm1d(hyperparams.n_conv_in2),
            nn.LeakyReLU(hyperparams.negative_slope),
            nn.Dropout(p=hyperparams.DropOut),
            
            hyperparams.activation,
            nn.MaxPool1d(kernel_size=hyperparams.kernel_size_pool_2,stride=hyperparams.stride_pool_2,padding=hyperparams.padding_pool_2,dilation=hyperparams.dilation_pool_2),
            nn.BatchNorm1d(hyperparams.n_conv_in2),
            nn.LeakyReLU(hyperparams.negative_slope),
            nn.Dropout(p=hyperparams.DropOut),
        )
        
        self.convlayer3=nn.Sequential(

            nn.Conv1d(hyperparams.n_conv_in2,hyperparams.n_conv_in3,kernel_size=hyperparams.kernel_size_3,stride=hyperparams.stride_3,padding=hyperparams.padding_3,dilation=hyperparams.dilation_3),
            # nn.Conv1d(hyperparams.n_conv_in3,hyperparams.n_conv_in3,kernel_size=hyperparams.kernel_size_3,stride=hyperparams.stride_3,padding=hyperparams.padding_3,dilation=hyperparams.dilation_3),
            # nn.Conv1d(hyperparams.n_conv_in3,hyperparams.n_conv_in3,kernel_size=hyperparams.kernel_size_3,stride=hyperparams.stride_3,padding=hyperparams.padding_3,dilation=hyperparams.dilation_3),
            nn.BatchNorm1d(hyperparams.n_conv_in3),
            nn.LeakyReLU(hyperparams.negative_slope),
            nn.Dropout(p=hyperparams.DropOut),
            hyperparams.activation,
            nn.MaxPool1d(kernel_size=hyperparams.kernel_size_pool_3,stride=hyperparams.stride_pool_3,padding=hyperparams.padding_pool_3,dilation=hyperparams.dilation_pool_3),
            nn.BatchNorm1d(hyperparams.n_conv_in3),
            hyperparams.activation,
            nn.LeakyReLU(hyperparams.negative_slope),
            nn.Dropout(p=hyperparams.DropOut),
            # nn.Conv1d(hyperparams.n_conv_in2,hyperparams.n_conv_in3,kernel_size=hyperparams.kernel_size_3,stride=hyperparams.stride_3,padding=hyperparams.padding_3,dilation=hyperparams.dilation_3),
            # nn.Conv1d(hyperparams.n_conv_in3,hyperparams.n_conv_in3,kernel_size=hyperparams.kernel_size_3,stride=hyperparams.stride_3,padding=hyperparams.padding_3,dilation=hyperparams.dilation_3),
            # nn.Conv1d(hyperparams.n_conv_in3,hyperparams.n_conv_in3,kernel_size=hyperparams.kernel_size_3,stride=hyperparams.stride_3,padding=hyperparams.padding_3,dilation=hyperparams.dilation_3),
        )

        self.fc_block1=nn.Sequential(
            nn.Flatten(),
            nn.Linear(hyperparams.n_conv_out*hyperparams.n_conv_in3,hyperparams.n_Flat_out),
            nn.BatchNorm1d(hyperparams.n_Flat_out),
            nn.LeakyReLU(hyperparams.negative_slope),
            hyperparams.activation,
            nn.Linear(hyperparams.n_Flat_out,output_size),
            )
        
        # self.fc_block2=nn.Sequential(
        #     nn.Linear(hyperparams.n_Flat_out,hyperparams.n_conv_in),
        #     nn.BatchNorm1d(hyperparams.n_conv_in),
        #     nn.LeakyReLU(hyperparams.negative_slope),
        #     hyperparams.activation,
        #     nn.Linear(hyperparams.n_conv_in,output_size),
            
        # )




    
    def Granulator(self,x):
        x_grid=from_torch_to_Datagrid(x=x)
        if "inter" not in self.hyperparameters.Smoothing_mode:     
            Recons_train=torch.zeros([x_grid.shape[0],self.hyperparameters.n_channel*len(self.hyperparameters.derivative),self.hyperparameters.granulation]).float().cuda()
            i=0
            for channel in range(self.hyperparameters.n_channel):
                for deriv in self.hyperparameters.derivative:
                    eval_points=linspace(1,x_grid.grid_points[0].shape[0],self.hyperparameters.granulation)
                    coefs_torch=torch.from_numpy(x_grid.to_basis(basis=self.hyperparameters.basis).coefficients).float().cuda()
                    basis_eval=self.hyperparameters.basis(eval_points=eval_points,derivative=deriv)
                    basis_fc = torch.from_numpy(basis_eval).float().cuda()
                    # coefs_torch=torch.from_numpy(coefs).float().cuda()        
                    Recons_train[:,i,:]=torch.matmul(coefs_torch,basis_fc[:,:,channel])
                    i+=1    
        
        else:
            x_grid.interpolation=Smoothing_method(n_knots=self.hyperparameters.n_knots,Mode=self.hyperparameters.Smoothing_mode,order=self.hyperparameters.order).smoothing()
            eval_points=linspace(1,x_grid.grid_points[0].shape[0],self.hyperparameters.granulation)
            Recons_train=x_grid.interpolation._evaluate(fdata=x_grid,eval_points=eval_points)[:,:,0]
            
            Recons_train=torch.tensor(Recons_train).reshape(Recons_train.shape[0],self.hyperparameters.n_channel,Recons_train.shape[1])
        
        return Recons_train.float().cuda()



    def forward(self,x):
        Granulated_x_train=self.Granulator(x)
        # tresh_out=torch.relu(Granulated_x_train)
        Conv_out=self.convlayer1(Granulated_x_train)
        Conv_out2=self.convlayer2(Conv_out)
        Conv_out3=self.convlayer3(Conv_out2)
        Lin_out=self.fc_block1(Conv_out3)
        return Lin_out.float().unsqueeze_(2).unsqueeze_(3)



class AdaFNN(nn.Module):

    def __init__(self, n_base=4, base_hidden=[64, 64, 64], grid=(0, 1),
                 sub_hidden=[128, 128, 128], dropout=0.1, lambda1=0.0, lambda2=0.0,
                 device=None):
        """
        n_base      : number of basis nodes, integer
        base_hidden : hidden layers used in each basis node, array of integers
        grid        : observation time grid, array of sorted floats including 0.0 and 1.0
        sub_hidden  : hidden layers in the subsequent network, array of integers
        dropout     : dropout probability
        lambda1     : penalty of L1 regularization, a positive real number
        lambda2     : penalty of L2 regularization, a positive real number
        device      : device for the training
        """
        super().__init__()
        self.n_base = n_base
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.device = device
        # grid should include both end points
        grid = np.array(grid)
        # send the time grid tensor to device
        self.t = torch.tensor(grid).to(device).float()
        self.h = torch.tensor(grid[1:] - grid[:-1]).to(device).float()
        # instantiate each basis node in the basis layer
        self.BL = nn.ModuleList([FeedForward(1, hidden=base_hidden, dropout=dropout, activation=F.selu)
                                 for _ in range(n_base)])
        # instantiate the subsequent network
        self.FF = FeedForward(n_base, sub_hidden, dropout)

    def forward(self, x):
        B, J = x.size()
        assert J == self.h.size()[0] + 1
        T = self.t.unsqueeze(dim=-1)
        # evaluate the current basis nodes at time grid
        self.bases = [basis(T).transpose(-1, -2) for basis in self.BL]
        """
        compute each basis node's L2 norm
        normalize basis nodes
        """
        l2_norm = _l2(torch.cat(self.bases, dim=0), self.h).detach()
        self.normalized_bases = [self.bases[i] / (l2_norm[i, 0] + 1e-6) for i in range(self.n_base)]
        # compute each score <basis_i, f> 
        score = torch.cat([_inner_product(b.repeat((B, 1)), x, self.h) # (B, 1)
                           for b in self.bases], dim=-1) # score dim = (B, n_base)
        # take the tensor of scores into the subsequent network
        out = self.FF(score)
        return out

    def R1(self, l1_k):
        """
        L1 regularization
        l1_k : number of basis nodes to regularize, integer        
        """
        if self.lambda1 == 0: return torch.zeros(1).to(self.device)
        # sample l1_k basis nodes to regularize
        selected = np.random.choice(self.n_base, min(l1_k, self.n_base), replace=False)
        selected_bases = torch.cat([self.normalized_bases[i] for i in selected], dim=0) # (k, J)
        return self.lambda1 * torch.mean(_l1(selected_bases, self.h))

    def R2(self, l2_pairs):
        """
        L2 regularization
        l2_pairs : number of pairs to regularize, integer  
        """
        if self.lambda2 == 0 or self.n_base == 1: return torch.zeros(1).to(self.device)
        k = min(l2_pairs, self.n_base * (self.n_base - 1) // 2)
        f1, f2 = [None] * k, [None] * k
        for i in range(k):
            a, b = np.random.choice(self.n_base, 2, replace=False)
            f1[i], f2[i] = self.normalized_bases[a], self.normalized_bases[b]
        return self.lambda2 * torch.mean(torch.abs(_inner_product(torch.cat(f1, dim=0),
                                                                  torch.cat(f2, dim=0),
                                                                  self.h)))

class FCNN(nn.Module):
    def __init__(self, hyperparams,output_size):
        super(FCNN, self).__init__()
        # Reste du code pour l'initialisation de la classe model
        self.hyperparameters=hyperparams
        self.Relu=nn.ReLU()

        self.convlayer1=nn.Sequential(
            nn.Conv1d(hyperparams.n_channel,hyperparams.n_conv_in,kernel_size=hyperparams.kernel_size_1,stride=hyperparams.stride_1,padding=hyperparams.padding_1,dilation=hyperparams.dilation_1),
            nn.BatchNorm1d(hyperparams.n_conv_in),
            nn.LeakyReLU(hyperparams.negative_slope),
            
            nn.MaxPool1d(kernel_size=hyperparams.kernel_size_pool_1,stride=hyperparams.stride_pool_1,padding=hyperparams.padding_pool_1,dilation=hyperparams.dilation_pool_1),
            nn.BatchNorm1d(hyperparams.n_conv_in),
            nn.LeakyReLU(hyperparams.negative_slope),
        )
        
        self.convlayer2=nn.Sequential(
            nn.Conv1d(hyperparams.n_conv_in,hyperparams.n_conv_in2,kernel_size=hyperparams.kernel_size_2,stride=hyperparams.stride_2,padding=hyperparams.padding_2,dilation=hyperparams.dilation_2),
            nn.BatchNorm1d(hyperparams.n_conv_in2),
            nn.LeakyReLU(hyperparams.negative_slope),
            
            nn.MaxPool1d(kernel_size=hyperparams.kernel_size_pool_2,stride=hyperparams.stride_pool_2,padding=hyperparams.padding_pool_2,dilation=hyperparams.dilation_pool_2),
            nn.BatchNorm1d(hyperparams.n_conv_in2),
            nn.LeakyReLU(hyperparams.negative_slope),
        )
        
        self.convlayer3=nn.Sequential(

            nn.Conv1d(hyperparams.n_conv_in2,hyperparams.n_conv_in3,kernel_size=hyperparams.kernel_size_3,stride=hyperparams.stride_3,padding=hyperparams.padding_3,dilation=hyperparams.dilation_3),
            nn.BatchNorm1d(hyperparams.n_conv_in3),
            nn.LeakyReLU(hyperparams.negative_slope),
            
            nn.MaxPool1d(kernel_size=hyperparams.kernel_size_pool_3,stride=hyperparams.stride_pool_3,padding=hyperparams.padding_pool_3,dilation=hyperparams.dilation_pool_3),
            nn.BatchNorm1d(hyperparams.n_conv_in3),
            nn.LeakyReLU(hyperparams.negative_slope),
        )

        self.fc_block=nn.Sequential(
            nn.Flatten(),
            nn.Linear(hyperparams.n_conv_out*hyperparams.n_conv_in3,hyperparams.n_Flat_out),
            nn.BatchNorm1d(hyperparams.n_Flat_out),
            nn.LeakyReLU(hyperparams.negative_slope),
            
            nn.Linear(hyperparams.n_Flat_out,output_size),
        )
    
        
    def forward(self,x):
       
        Conv_out=self.convlayer1(x)
        Conv_out2=self.convlayer2(Conv_out)
        Conv_out3=self.convlayer3(Conv_out2)
        Lin_out=self.fc_block(Conv_out3)
        return Lin_out.float().unsqueeze_(2).unsqueeze_(3)






class MLP(nn.Module):
    def __init__(self,hyperparams,input_size,output_size):
        super(MLP,self).__init__()
        self.input_size=input_size
        self.fc_block=nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size,hyperparams.n_conv_in),
            nn.BatchNorm1d(hyperparams.n_conv_in),
            nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
            nn.Dropout(p=hyperparams.DropOut),
            hyperparams.activation,
            nn.Linear(hyperparams.n_conv_in,hyperparams.n_conv_in2),
            nn.BatchNorm1d(hyperparams.n_conv_in2),
            nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
            nn.Dropout(p=hyperparams.DropOut),
            hyperparams.activation,
            nn.Linear(hyperparams.n_conv_in2,hyperparams.n_conv_in3),
            nn.BatchNorm1d(hyperparams.n_conv_in3),
            nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
            nn.Dropout(p=hyperparams.DropOut),
            hyperparams.activation,
            nn.Linear(hyperparams.n_conv_in3,output_size),
        )

    def forward(self,x):
        if isinstance(x,skfda.representation.grid.FDataGrid):
            data_matrix=torch.tensor(x.data_matrix).float().cuda()
            Lin_out=self.fc_block(data_matrix)
        elif isinstance(x,torch.Tensor):
            Lin_out=self.fc_block(x)
        else:
            raise ValueError("if isinstance(x,skfda.representation.grid.FDataGrid):")
        return Lin_out.float().unsqueeze(2).unsqueeze(3)

        

class Project_classifier(nn.Module):
    def __init__(self,hyperparams,output_size):
        super(Project_classifier,self).__init__()
        self.basis=hyperparams.Smoothing_method.smoothing()
        self.fc_block=nn.Sequential(
            nn.Linear(self.basis.n_basis,hyperparams.n_conv_in),
            nn.BatchNorm1d(hyperparams.n_conv_in),
            nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
            nn.Dropout(p=hyperparams.DropOut),
            hyperparams.activation,
            nn.Linear(hyperparams.n_conv_in,hyperparams.n_conv_in2),
            nn.BatchNorm1d(hyperparams.n_conv_in2),
            nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
            nn.Dropout(p=hyperparams.DropOut),
            hyperparams.activation,
            nn.Linear(hyperparams.n_conv_in2,hyperparams.n_conv_in3),
            nn.BatchNorm1d(hyperparams.n_conv_in3),
            nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
            nn.Dropout(p=hyperparams.DropOut),
            hyperparams.activation,
            nn.Linear(hyperparams.n_conv_in3,output_size),
        )

        
    def Project(self, x, basis_fc):
        # basis_fc: n_time X nbasis
        # x: n_subject X n_time
        # w = x.size(1)-1
        # W = torch.tensor([1/(2*w)]+[1/w]*(w-1)+[1/(2*w)])
        
        f = torch.matmul(torch.tensor(x.data_matrix[:,:,0]).float().cuda(), torch.t(basis_fc))
        return f
    def forward(self,x):
        if isinstance(x,skfda.representation.grid.FDataGrid):
            basis_fc=self.basis(x.grid_points[0],derivative=0)[:,:,0]
            basis_fc=torch.tensor(basis_fc).float().cuda()
            proj_out=self.Project(x,basis_fc=basis_fc)
            lin_out=self.fc_block(proj_out)
        if isinstance(x,torch.Tensor):
            grid=fd(x[:,0,:].cpu(),grid_points=np.arange(x.shape[2]+1)[1:])
            basis_fc=self.basis(grid.grid_points[0],derivative=0)[:,:,0]
            basis_fc=torch.tensor(basis_fc).float().cuda()
            proj_out=self.Project(grid,basis_fc=basis_fc)
            lin_out=self.fc_block(proj_out)
        return lin_out.float().unsqueeze(2).unsqueeze(3)
    
class MLP(nn.Module):
            def __init__(self,hyperparams,input_size,output_size):
                super(MLP,self).__init__()
                self.input_size=input_size
                self.hyperparameters=hyperparams
                self.fc_block=nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(input_size*hyperparams.n_channel,hyperparams.n_conv_in),
                    nn.BatchNorm1d(hyperparams.n_conv_in),
                    nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
                    nn.Dropout(p=hyperparams.DropOut),

                    hyperparams.activation,
                    nn.Linear(hyperparams.n_conv_in,hyperparams.n_conv_in2),
                    nn.BatchNorm1d(hyperparams.n_conv_in2),
                    nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
                    nn.Dropout(p=hyperparams.DropOut),
                    hyperparams.activation,
                    nn.Linear(hyperparams.n_conv_in2,hyperparams.n_conv_in3),
                    nn.BatchNorm1d(hyperparams.n_conv_in3),
                    nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
                    nn.Dropout(p=hyperparams.DropOut),
                    hyperparams.activation,
                    nn.Linear(hyperparams.n_conv_in3,output_size),
                )

            def forward(self,x):
                if isinstance(x,skfda.representation.grid.FDataGrid):
                    data_matrix=torch.tensor(x.data_matrix).float().cuda()
                    
                    input=data_matrix.reshape(data_matrix.shape[0],data_matrix.shape[2],data_matrix.shape[1])
                elif isinstance(x,torch.Tensor):
                    input=x
                else:
                    raise ValueError("NN input must be fdatagrid or torch tensor")
                Lin_out=self.fc_block(input)
                return Lin_out.float().unsqueeze(2).unsqueeze(3)
          

class LSTM_class(nn.Module):
    def __init__(self,hyperparams,input_size,output_size):
        super(LSTM_class,self).__init__()

        self.hyperparameters=hyperparams
        self.lstm=nn.LSTM(input_size,hidden_size=hyperparams.n_conv_in,num_layers=hyperparams.kernel_size_1,batch_first=True,bidirectional=hyperparams.bidirectional)
        
        self.n_features=(hyperparams.n_conv_in*hyperparams.n_channel*(hyperparams.bidirectional+1))
        self.fc_block=nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=self.n_features,out_features=hyperparams.n_conv_in2),
        nn.BatchNorm1d(hyperparams.n_conv_in2),
        nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
        
        hyperparams.activation,
        nn.Linear(hyperparams.n_conv_in2,hyperparams.n_conv_in3),
        nn.BatchNorm1d(hyperparams.n_conv_in3),
        nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
        nn.Linear(hyperparams.n_conv_in3,output_size)
        )

    def forward(self,x):
        if isinstance(x,skfda.representation.grid.FDataGrid):
            data_matrix=torch.tensor(x.data_matrix).float().cuda()
            input=data_matrix.reshape(data_matrix.shape[0],data_matrix.shape[2],data_matrix.shape[1])
        elif isinstance(x,torch.Tensor):
            input=x
        
        lstm_out,_=self.lstm(input)
        Lin_out=self.fc_block(lstm_out)

        return Lin_out.float().unsqueeze(2).unsqueeze(3)
      
class GRU(nn.Module):
    def __init__(self,hyperparams,input_size,output_size):
        super(GRU,self).__init__()
        self.hyperparameters=hyperparams
        self.gru=nn.LSTM(input_size,hidden_size=hyperparams.n_conv_in,num_layers=hyperparams.kernel_size_1,batch_first=True,bidirectional=hyperparams.bidirectional)
        self.fc_block=nn.Sequential(
        nn.Flatten(),
        nn.Linear(hyperparams.n_conv_in*hyperparams.n_channel*(hyperparams.bidirectional+1),hyperparams.n_conv_in2),
        nn.BatchNorm1d(hyperparams.n_conv_in2),
        nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
        hyperparams.activation,
        nn.Linear(hyperparams.n_conv_in2,hyperparams.n_conv_in3),
        nn.BatchNorm1d(hyperparams.n_conv_in3),
        nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
        nn.Linear(hyperparams.n_conv_in3,output_size)
        )

    def forward(self,x):
        if isinstance(x,skfda.representation.grid.FDataGrid):
            data_matrix=torch.tensor(x.data_matrix).float().cuda()
            input=data_matrix.reshape(data_matrix.shape[0],data_matrix.shape[2],data_matrix.shape[1])
        elif isinstance(x,torch.Tensor):
            input=x
        
        gru_out,_=self.gru(input)
        Lin_out=self.fc_block(gru_out)

        return Lin_out.float().unsqueeze(2).unsqueeze(3)
        

class TSCNN_no_pool(nn.Module):
    def __init__(self, hyperparams,output_size):
        super(TSCNN_no_pool, self).__init__()

        self.hyperparameters=hyperparams
        self.convlayer1=nn.Sequential(
            nn.Conv1d(self.hyperparameters.n_channel*len(hyperparams.derivative),hyperparams.n_conv_in,kernel_size=hyperparams.kernel_size_1,stride=hyperparams.stride_1,padding=hyperparams.padding_1,dilation=hyperparams.dilation_1),
            nn.BatchNorm1d(hyperparams.n_conv_in),
            nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
            nn.Dropout(p=hyperparams.DropOut),
            hyperparams.activation,
            nn.MaxPool1d(kernel_size=hyperparams.kernel_size_pool_1,stride=hyperparams.stride_pool_1,padding=hyperparams.padding_pool_1,dilation=hyperparams.dilation_pool_1),
            # nn.LPPool1d(norm_type=hyperparams.norm_type,kernel_size=hyperparams.kernel_size_pool_1,stride=hyperparams.stride_pool_1),
            nn.BatchNorm1d(hyperparams.n_conv_in),
            nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
            nn.Dropout(p=hyperparams.DropOut),
            )
        self.convlayer2=nn.Sequential(
            nn.Conv1d(hyperparams.n_conv_in,hyperparams.n_conv_in2,kernel_size=hyperparams.kernel_size_2,stride=hyperparams.stride_2,padding=hyperparams.padding_2,dilation=hyperparams.dilation_2),
            nn.BatchNorm1d(hyperparams.n_conv_in2),
            nn.LeakyReLU(hyperparams.negative_slope),
            
            nn.Dropout(p=hyperparams.DropOut),
            nn.MaxPool1d(kernel_size=hyperparams.kernel_size_pool_2,stride=hyperparams.stride_pool_2,padding=hyperparams.padding_pool_2,dilation=hyperparams.dilation_pool_2),
            nn.BatchNorm1d(hyperparams.n_conv_in2),
            nn.LeakyReLU(hyperparams.negative_slope),
        )
        
        self.convlayer3=nn.Sequential(

            nn.Conv1d(hyperparams.n_conv_in2,hyperparams.n_conv_in3,kernel_size=hyperparams.kernel_size_3,stride=hyperparams.stride_3,padding=hyperparams.padding_3,dilation=hyperparams.dilation_3),
            nn.BatchNorm1d(hyperparams.n_conv_in3),
            nn.LeakyReLU(hyperparams.negative_slope),
            
            nn.MaxPool1d(kernel_size=hyperparams.kernel_size_pool_3,stride=hyperparams.stride_pool_3,padding=hyperparams.padding_pool_3,dilation=hyperparams.dilation_pool_3),
            nn.BatchNorm1d(hyperparams.n_conv_in3),
            nn.LeakyReLU(hyperparams.negative_slope),
            nn.Dropout(p=hyperparams.DropOut),
        )
        self.fc_block1=nn.Sequential(
            nn.Flatten(),
            nn.Linear(hyperparams.n_conv_out*hyperparams.n_conv_in3,hyperparams.n_Flat_out),
            nn.BatchNorm1d(hyperparams.n_Flat_out),
            nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
            hyperparams.activation,
            nn.Dropout(p=hyperparams.DropOut),
            nn.Linear(hyperparams.n_Flat_out,output_size)
            )
        # self.fc_block2=nn.Sequential(
        #     nn.Linear(hyperparams.n_conv_in2,hyperparams.n_conv_in3),
        #     nn.BatchNorm1d(hyperparams.n_conv_in3),
        #     nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
        #     hyperparams.activation,
        #     nn.Dropout(p=hyperparams.DropOut),
        #     )
        # self.fc_block3=nn.Sequential(
        #     nn.Linear(hyperparams.n_conv_in3,hyperparams.n_Flat_out),
        #     nn.BatchNorm1d(hyperparams.n_Flat_out),
        #     nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
        #     hyperparams.activation,
        #     nn.Dropout(p=hyperparams.DropOut),
        #     nn.Linear(hyperparams.n_Flat_out,output_size)
        # )
class TSCNN_une_couche(nn.Module):
    def __init__(self, hyperparams,output_size):
        super(TSCNN_une_couche, self).__init__()

        self.hyperparameters=hyperparams
        self.convlayer1=nn.Sequential(
            nn.Conv1d(self.hyperparameters.n_channel*len(hyperparams.derivative),hyperparams.n_conv_in,kernel_size=hyperparams.kernel_size_1,stride=hyperparams.stride_1,padding=hyperparams.padding_1,dilation=hyperparams.dilation_1),
            nn.BatchNorm1d(hyperparams.n_conv_in),
            nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
            nn.Dropout(p=hyperparams.DropOut),
            hyperparams.activation,
            nn.MaxPool1d(kernel_size=hyperparams.kernel_size_pool_1,stride=hyperparams.stride_pool_1,padding=hyperparams.padding_pool_1,dilation=hyperparams.dilation_pool_1),
            # nn.LPPool1d(norm_type=hyperparams.norm_type,kernel_size=hyperparams.kernel_size_pool_1,stride=hyperparams.stride_pool_1),
            nn.BatchNorm1d(hyperparams.n_conv_in),
            nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
            nn.Dropout(p=hyperparams.DropOut),
            )
        self.fc_block1=nn.Sequential(
            nn.Flatten(),
            nn.Linear(hyperparams.n_conv_out*hyperparams.n_conv_in,hyperparams.n_conv_in2),
            nn.BatchNorm1d(hyperparams.n_conv_in2),
            nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
            hyperparams.activation,
            nn.Dropout(p=hyperparams.DropOut),
            nn.Linear(hyperparams.n_conv_in2,output_size)
            )
        self.fc_block2=nn.Sequential(
            nn.Linear(hyperparams.n_conv_in2,hyperparams.n_conv_in3),
            nn.BatchNorm1d(hyperparams.n_conv_in3),
            nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
            hyperparams.activation,
            nn.Dropout(p=hyperparams.DropOut),
            )
        self.fc_block3=nn.Sequential(
            nn.Linear(hyperparams.n_conv_in3,hyperparams.n_Flat_out),
            nn.BatchNorm1d(hyperparams.n_Flat_out),
            nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
            hyperparams.activation,
            nn.Dropout(p=hyperparams.DropOut),
            nn.Linear(hyperparams.n_Flat_out,output_size)
        )




    
    def Granulator(self,x):
        x_grid=from_torch_to_Datagrid(x=x)
        if "inter" not in self.hyperparameters.Smoothing_mode:     
            Recons_train=torch.zeros([x_grid.shape[0],self.hyperparameters.n_channel*len(self.hyperparameters.derivative),self.hyperparameters.granulation]).float().cuda()
            i=0
            for channel in range(self.hyperparameters.n_channel):
                for deriv in self.hyperparameters.derivative:
                    eval_points=linspace(1,x_grid.grid_points[0].shape[0],self.hyperparameters.granulation)
                    coefs_torch=torch.from_numpy(x_grid.to_basis(basis=self.hyperparameters.basis).coefficients).float().cuda()
                    basis_eval=self.hyperparameters.basis(eval_points=eval_points,derivative=deriv)
                    basis_fc = torch.from_numpy(basis_eval).float().cuda()
                    # coefs_torch=torch.from_numpy(coefs).float().cuda()        
                    Recons_train[:,i,:]=torch.matmul(coefs_torch,basis_fc[:,:,channel])
                    i+=1    
        
        else:
            x_grid.interpolation=Smoothing_method(n_knots=self.hyperparameters.n_knots,Mode=self.hyperparameters.Smoothing_mode,order=self.hyperparameters.order).smoothing()
            eval_points=linspace(1,x_grid.grid_points[0].shape[0],self.hyperparameters.granulation)
            Recons_train=x_grid.interpolation._evaluate(fdata=x_grid,eval_points=eval_points)[:,:,0]
            Recons_train=torch.tensor(Recons_train).reshape(Recons_train.shape[0],self.hyperparameters.n_channel,Recons_train.shape[1])
        
        return Recons_train.float().cuda()



    def forward(self,x):
        Granulated_x_train=self.Granulator(x)
        Conv_out=self.convlayer1(Granulated_x_train)
        Conv_out2=self.fc_block1(Conv_out)
        Conv_out3=self.fc_block2(Conv_out2)
        Lin_out=self.fc_block3(Conv_out3)

        return Lin_out.float().unsqueeze_(2).unsqueeze_(3)





def Compile_class(model_class="avec",hyperparams=HyperParameters(),output_size=1,x_train=torch.zeros(6,6,7)):


    if isinstance(x_train,torch.Tensor):
        hyperparams.n_channel=x_train.shape[1]
    if isinstance(x_train,skfda.representation.grid.FDataGrid):
            hyperparams.n_channel=x_train.data_matrix.shape[2]

    
    if "TSC" in model_class:
    # if ("Conv" in model_class) or("Smooth" in model_class) or ("TSC" in model_class):
        if isinstance(x_train,skfda.representation.grid.FDataGrid):
                if x_train.data_matrix.shape[2]!=1:
                    grid_T=x_train.data_matrix.shape[1]
                    t0=x_train.grid_points[0][0]
                    basis_list=[]
                    for channel in range(hyperparams.n_channel):
                        basis_channel=Smoothing_method(n_knots=hyperparams.n_knots,t0=t0,T=grid_T,order=hyperparams.order,Mode=hyperparams.Smoothing_mode).smoothing()
                        basis_list.append(basis_channel)
                    hyperparams.basis=MultiBasis(basis_list=basis_list)
        elif isinstance(x_train,torch.Tensor):
            grid_T=x_train.shape[2]
            # t0=x_train.grid_points[0][0]
            # hyperparams.n_knots=grid_T//5
            hyperparams.basis=Smoothing_method(n_knots=hyperparams.n_knots,order=hyperparams.order,T=grid_T,Mode=hyperparams.Smoothing_mode).smoothing()

        hyperparams.n_conv_out=conv_total_out(hyperparams=hyperparams)
        module=TSCNN(hyperparams=hyperparams,output_size=output_size)
    
    elif "une couche" in model_class:
    # if ("Conv" in model_class) or("Smooth" in model_class) or ("TSC" in model_class):
        if isinstance(x_train,skfda.representation.grid.FDataGrid):
                if x_train.data_matrix.shape[2]!=1:
                    grid_T=x_train.data_matrix.shape[1]
                    t0=x_train.grid_points[0][0]
                    basis_list=[]
                    for channel in range(hyperparams.n_channel):
                        basis_channel=B(knots=linspace(t0,grid_T,hyperparams.n_knots),order=hyperparams.order)
                        basis_list.append(basis_channel)
                    hyperparams.basis=MultiBasis(basis_list=basis_list)
        elif isinstance(x_train,torch.Tensor):
            grid_T=x_train.shape[2]
            # t0=x_train.grid_points[0][0]
            
            # hyperparams.n_knots=grid_T//5
            hyperparams.basis=B(knots=linspace(1,grid_T,hyperparams.n_knots),order=hyperparams.order)
        hyperparams.n_conv_out=conv_total_out_une_couche(hyperparams=hyperparams)
        module=TSCNN_une_couche(hyperparams=hyperparams,output_size=output_size)
    
    elif "Ada" in model_class:
        module=AdaFNN()

    elif "sans Max" in model_class:

        if isinstance(x_train,skfda.representation.grid.FDataGrid):
                if x_train.data_matrix.shape[2]!=1:
                    grid_T=x_train.data_matrix.shape[1]
                    basis_list=[]
                    t0=x_train.grid_points[0][0]
                    for channel in range(hyperparams.n_channel):
                        basis_channel=B(knots=linspace(t0,grid_T,hyperparams.n_knots),order=hyperparams.order)
                        basis_list.append(basis_channel)
                    hyperparams.basis=MultiBasis(basis_list=basis_list)
        elif isinstance(x_train,torch.Tensor):
            grid_T=x_train.shape[2]
            # hyperparams.n_knots=grid_T//5
            hyperparams.basis=B(knots=linspace(1,grid_T,hyperparams.n_knots),order=hyperparams.order)
        hyperparams.n_conv_out=conv_total_out(hyperparams=hyperparams,Maxpooling=True)
        module=TSCNN_no_pool(hyperparams=hyperparams,output_size=output_size)
    elif "1D" in model_class:

        if isinstance(x_train,skfda.representation.grid.FDataGrid):
                if x_train.data_matrix.shape[2]!=1:
                    grid_T=x_train.data_matrix.shape[1]
                    t0=x_train.grid_points[0][0]
                    basis_list=[]
                    for channel in range(hyperparams.n_channel):
                        basis_channel=B(knots=linspace(t0,grid_T,hyperparams.n_knots),order=hyperparams.order)
                        basis_list.append(basis_channel)
                    hyperparams.basis=MultiBasis(basis_list=basis_list)
        elif isinstance(x_train,torch.Tensor):
            grid_T=x_train.shape[2]
            # hyperparams.n_knots=grid_T//5
            hyperparams.basis=B(knots=linspace(1,grid_T,hyperparams.n_knots),order=hyperparams.order)
        hyperparams.n_conv_out=conv_total_out(hyperparams=hyperparams,lp=False)
        module=FCNN(hyperparams=hyperparams,output_size=output_size)

    elif ("mlp" in model_class) or ("MLP" in model_class):

        if isinstance(x_train,skfda.representation.grid.FDataGrid):
            data_matrix=torch.tensor(x_train.data_matrix).float().cuda()
            input_size=data_matrix.shape[1]
        elif isinstance(x_train,torch.Tensor):
            input_size=x_train.shape[2]
            
                
        module=MLP(hyperparams,input_size=input_size,output_size=output_size)

    elif ("Proj" in model_class) or ("proj" in model_class):
        module=Project_classifier(hyperparams=hyperparams,output_size=output_size)

    elif ("lstm" in model_class) or ("LSTM" in model_class):
        
        if isinstance(x_train,skfda.representation.grid.FDataGrid):
            data_matrix=torch.tensor(x_train.data_matrix).float().cuda()
            input_size=data_matrix.shape[1]
        elif isinstance(x_train,torch.Tensor):
            input_size=x_train.shape[2] 
        
        module=LSTM_class(hyperparams=hyperparams,input_size=input_size,output_size=output_size)
    
    elif ('Gru' in model_class) or ('GRU' in model_class) or ('gru' in model_class):
        if isinstance(x_train,skfda.representation.grid.FDataGrid):
            data_matrix=torch.tensor(x_train.data_matrix).float().cuda()
            input_size=data_matrix.shape[1]
        elif isinstance(x_train,torch.Tensor):
            input_size=x_train.shape[2] 
        
        module=GRU(hyperparams,input_size=input_size,output_size=output_size)

    else:
        raise ValueError("model_class shoud be a class already defined")
    
    return module.cuda().apply(weights_init_normal)


def Compile_train(module, hyperparams,X,Y,data_amount=100000):
    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,Y,shuffle=True)    
    x_train,y_train=x_train[:data_amount],y_train[:data_amount]
    opt=hyperparams.opt
    lr=hyperparams.lr
    loss=hyperparams.loss
    batch_size=hyperparams.batch_size
    if opt == "Adam":
        optimizer = optim.Adam(params=module.parameters(), 
                            lr=lr,
                            betas=hyperparams.betas,
                            fused=hyperparams.fused,
                            eps=hyperparams.eps,
                            weight_decay=hyperparams.decay,)
    elif opt=="SGD":
    
        optimizer = optim.SGD(params=module.parameters(),
                              lr=lr,
                              weight_decay=hyperparams.decay,
                              nesterov=hyperparams.nesterov,
                              momentum=hyperparams.momentum,
                              dampening=hyperparams.dampening)
        
    elif opt=="Adadelta" or ("Delta" in opt):
    
        optimizer = optim.Adadelta(params=module.parameters(),
                              lr=lr,weight_decay=hyperparams.decay,rho=hyperparams.rho,foreach=hyperparams.foreach,eps=hyperparams.eps
                              )
        
        
    elif opt=="Adagrad" or ("Grad" in opt):
    
        optimizer = optim.Adagrad(params=module.parameters(),
                              lr=lr,lr_decay=hyperparams.lr_decay,foreach=hyperparams.foreach,eps=hyperparams.eps
                              )
        

        

    def train(n_epochs, module, optimizer, loss, batch_size):
        testing_acc=torch.tensor([0])

        for epoch in tqdm(range(n_epochs)):
            train_loss = torch.tensor(0).cuda().long()
            
            # Mélanger les données d'entraînement
            indices = list(range(len(x_train)))
            random.shuffle(indices)
            
            batch_index = 0  # Indice de batch
            
            for i in range(int(len(x_train) / batch_size)):
                # Obtenir les indices des données mélangées
                batch_indices = indices[batch_index:batch_index+batch_size]
                functions_train = x_train[batch_indices]
                labels_train = y_train[batch_indices]
                optimizer.zero_grad()
                output = module(functions_train)
                loss_value = loss(input=output, target=labels_train)
                loss_value.backward()
                optimizer.step()
                train_loss += loss_value.long()
                batch_index += batch_size  # Passer au prochain batch
                
            if len(unique(y_train.cpu()))<y_train.shape[0]//2:
                # accuracy_training=((torch.sum(torch.argmax(module(x_train),dim=1)==y_train)/x_train.shape[0])*100).cpu().unsqueeze(0)
                accuracy=((torch.sum(torch.argmax(module(x_test),dim=1)==y_test)/x_test.shape[0])*100).cpu().unsqueeze(0)
                testing_acc=torch.cat([testing_acc,accuracy],dim=0)
                # training_acc=torch.cat([training_acc,accuracy_training],dim=0)
            else:
                mse_loss_test=torch.sqrt(nn.MSELoss()(module(x_test),y_test)).unsqueeze(0).cpu()
                # mse_loss_train=nn.MSELoss()(module(x_train),y_train).unsqueeze(0).cpu()
                testing_acc=torch.cat([testing_acc,mse_loss_test.cpu()],dim=0).cpu()
                # training_acc=torch.cat([training_acc,mse_loss_train.cpu()],dim=0).cpu()
        return testing_acc.detach().cpu()
    
    return lambda n_epochs: train(n_epochs, module, optimizer, loss, batch_size)




def Hyperparameter_Test(X,Y,supra_epochs=50,alpha=0.95,model_class="smooth",hyperparameters=HyperParameters()):
    monte_carlo_test_acc=torch.zeros(hyperparameters.n_epochs+1,1)
    # monte_carlo_train_acc=torch.zeros(hyperparameters.n_epochs+1,1)
    if len(unique(Y.cpu()))<Y.shape[0]//2:
        output_size=len(unique(Y.cpu()))
        # hyperparameters.loss=nn.CrossEntropyLoss()
    else:
        output_size=Y.shape[1]
        hyperparameters.loss=nn.MSELoss()
        # hyperparameters.lr=hyperparameters.lr*10

    for epoch in tqdm((range(supra_epochs))):
        ##Compilation de la classe 
        Model=Compile_class(hyperparams=hyperparameters,output_size=output_size,model_class=model_class,x_train=X)
        train_fn = Compile_train(module=Model, hyperparams=hyperparameters,X=X,Y=Y,data_amount=X.shape[0])
        monte_carlo_test=train_fn(n_epochs=hyperparameters.n_epochs)
        monte_carlo_test_acc=torch.cat([monte_carlo_test_acc,monte_carlo_test.unsqueeze(1)],dim=1)
        # monte_carlo_train_acc=torch.cat([monte_carlo_train_acc,monte_carlo_train.unsqueeze(1)],dim=1)

    gc.collect()
    torch.cuda.empty_cache()
    # mean_acc_train=torch.mean(monte_carlo_train_acc[1:,1:],dim=1).float()
    # var_acc_train=torch.var(monte_carlo_test_acc[1:,1:],dim=1).float()
    
    chiffre = alpha
    quartile = norm.ppf((1 + chiffre) / 2)
    mean_acc_test=torch.mean(monte_carlo_test_acc[1:,1:],dim=1).float()
    var_acc_test=torch.var(monte_carlo_test_acc[1:,1:],dim=1).float()
    IC_acc_test=torch.cat([(mean_acc_test-quartile*sqrt(var_acc_test/supra_epochs)).unsqueeze(1),(mean_acc_test+quartile*sqrt(var_acc_test/supra_epochs)).unsqueeze(1)],dim=1)

    return monte_carlo_test_acc[1:,1:], mean_acc_test,IC_acc_test


def Hyperparameter_Test_n_data(X,Y,supra_epochs=50,alpha=0.95,model_class="smooth",hyperparameters=HyperParameters(),data_step=15):
    n_data=len(range(2,X.shape[0]*3//4,data_step))
    max_monte_carlo_test_acc=torch.zeros(n_data,supra_epochs)
    # max_monte_carlo_train_acc=torch.zeros(n_data,supra_epochs)
    if len(unique(Y.cpu()))<Y.shape[0]//2:
        output_size=len(unique(Y.cpu()))
        hyperparameters.loss=nn.CrossEntropyLoss()
    else:
        output_size=Y.shape[1]
        hyperparameters.loss=nn.MSELoss()
        hyperparameters.lr=hyperparameters.lr*10

    
    for epoch in tqdm((range(supra_epochs))):
        

        chiffre = alpha
        quartile = norm.ppf((1 + chiffre) / 2)
        
        
        ##Compilation de la classe 

        monte_carlo_test_acc=torch.zeros(hyperparameters.n_epochs+1,1)
        
        for data in range(2,3*X.shape[0]//4,data_step):
            hyperparameters.batch_size=data
            Model=Compile_class(hyperparams=hyperparameters,output_size=output_size,model_class=model_class,x_train=X)
            train_fn = Compile_train(module=Model, hyperparams=hyperparameters,X=X,Y=Y,data_amount=data)
            monte_carlo_test=train_fn(n_epochs=hyperparameters.n_epochs)
            monte_carlo_test_acc=torch.cat([monte_carlo_test_acc,monte_carlo_test.unsqueeze(1)],dim=1)
            # monte_carlo_train_acc=torch.cat([monte_carlo_train_acc,monte_carlo_train.unsqueeze(1)],dim=1)

        

        

        gc.collect()
        torch.cuda.empty_cache()

        # max_acc_train=torch.max(monte_carlo_train_acc[1:,1:],dim=0).values.float()
        max_acc_test=torch.max(monte_carlo_test_acc[1:,1:],dim=0).values.float()
        max_monte_carlo_test_acc[:,epoch]=max_acc_test
        # max_monte_carlo_train_acc[:,epoch]=max_acc_train
    
    
        gc.collect()
        torch.cuda.empty_cache()
    # mean_acc_train=torch.mean(max_monte_carlo_train_acc,dim=1).float()
    # # max_acc_train=torch.max(monte_carlo_train_acc[1:,1:],dim=1).float()
    # var_acc_train=torch.var(max_monte_carlo_train_acc,dim=1).float()

    
    mean_acc_test=torch.mean(max_monte_carlo_test_acc,dim=1).float()

    var_acc_test=torch.var(max_monte_carlo_test_acc,dim=1).float()
    IC_acc_test=torch.cat([(mean_acc_test-quartile*sqrt(var_acc_test/supra_epochs)).unsqueeze(1),(mean_acc_test+quartile*sqrt(var_acc_test/supra_epochs)).unsqueeze(1)],dim=1)
    # IC_acc_train=torch.cat([(mean_acc_train-quartile*sqrt(var_acc_train/supra_epochs)).unsqueeze(1),(mean_acc_train+quartile*sqrt(var_acc_train/supra_epochs)).unsqueeze(1)],dim=1)

    return max_monte_carlo_test_acc,mean_acc_test,IC_acc_test


def Compare_epochs(
            params=None,
            spec_param=None,
            datasets=None,
            models=None,
            supra_epochs=50,
            alpha=0.95,
            colors=None,
            label=None,
            Conf_int=True,):

    
    
    # fig, axes = plt.subplots(int(floor(sqrt(len(datasets)))),int(floor(sqrt(len(datasets)))), figsize=(16, 16))
    num_datasets = len(datasets)
    num_plots_per_row = int(sqrt(num_datasets))
    num_plots_per_col = (num_datasets + num_plots_per_row - 1) // num_plots_per_row

    # Création de la figure et des axes
    fig, axes = plt.subplots(num_plots_per_col, num_plots_per_row, figsize=(12, 12))
    # fig, axes = plt.subplots(((len(datasets))), figsize=(16, 16))
    string1='Précision de '

    monte_carlo_test_acc=torch.zeros(params[0].n_epochs,supra_epochs,len(datasets),len(params),len(models))
    mean_acc_test=torch.zeros(params[0].n_epochs,len(datasets),len(params),len(models))
    IC_acc_test=torch.zeros(params[0].n_epochs,2,len(datasets),len(params),len(models))
    
    for i,dataset in enumerate((datasets)):  
        print(dataset['dataset_name'])
        name=dataset['dataset_name']  
    # Boucle pour créer chaque subplot
        X,Y=dataset['X'],dataset['Y']
        window_left = i // num_plots_per_row
        window_right = i % num_plots_per_row
        ax = axes[window_left, window_right]
        X=from_torch_to_Datagrid(X)
            
        T=X.data_matrix.shape[1]     
        for j,hyperparams in enumerate(params):
            # hyperparams.batch_size=X.shape[0]
            
           for k,model in enumerate(models):

                print(model+label[k])
                if len(unique(Y.cpu()))>Y.shape[0]//2:
                    spec_param[model][name].loss=nn.MSELoss()
                    # spec_param[k][i].lr=10*spec_param[k][i].lr
                    reg=True
                else:
                    spec_param[model][name].loss=nn.CrossEntropyLoss()
                    reg=False
                spec_param[model][name].batch_size=X.shape[0]//4
                # if i==0 :
                #     # spec_param[k].derivatives=[0]
                # elif i==0:
                #     spec_param[k].n_conv_in3=128

                # elif i==1:

                #     spec_param[k].loss=nn.MSELoss()
                #     spec_param[k].n_conv_in3=256
                #     spec_param[k].derivatives=[0,1,2]



                # elif i==3:
                #     spec_param[k].derivatives=[0,1]


                # spec_param[k].granulation=T

                
                # spec_param[k].padding_1=T//10
                # spec_param[k].dilation_1=((spec_param[k].granulation//(T-1))-2)
                # spec_param[k].stride_1=((spec_param[k].granulation//(T-1)))//2
                

                

                    


                


                
                
                monte_carlo_test_acc[:,:,i,j,k],mean_acc_test[:,i,j,k],IC_acc_test[:,:,i,j,k]=Hyperparameter_Test(
                    model_class=model,
                    hyperparameters=spec_param[model][name],

                    supra_epochs=supra_epochs,
                    X=X,
                    Y=Y,
                    alpha=alpha,
                    )
                
                
                # ax.plot(np.arange(hyperparams.n_epochs+1)[1:],mean_acc_test[:,i,j,k], label=string1+model+' '+"hyperparamètre"+str(j+1),color=colors[j])
                # ax.plot(np.arange(hyperparams.n_epochs+1)[1:],IC_acc_test[:,:,i,j,k],linestyle="dashed",color=colors[j])
                ax.plot(np.arange(hyperparams.n_epochs+1)[1:],mean_acc_test[:,i,j,k], label=string1+model+label[k],color=colors[k])
                if Conf_int:
                    ax.plot(np.arange(hyperparams.n_epochs+1)[1:],IC_acc_test[:,:,i,j,k],linestyle="dashed",color=colors[k])
                # ax.plot(np.arange(hyperparams.n_epochs+1)[1:],IC_acc_test[:,:,i,j,k],linestyle="dashed",color="")
                # ax.plot(np.arange(hyperparams.n_epochs+1)[1:],mean_acc_test[:,i,j,k], label=string1+model+' '+str(j+1)+"ème hyperparamètre")
                ax.set_title(dataset['dataset_name'])
                # ax.set_ylim((0,100))
                ax.set_xlabel("epochs")
                ax.set_ylabel("Pourcentage bien classés dans l'ensemble de test")
                if reg:
                    ax.legend(loc="upper right")       
                else:
                    ax.legend(loc="lower right")       

    plt.tight_layout()  # Ajuster automatiquement les espacements entre les subplots
    
    plt.show()

    return fig,monte_carlo_test_acc,mean_acc_test,IC_acc_test
    # Premier dataset 

# def Comp_plot(
#         models=None,
#         Title="Title",
#         colors=None,
#         label=None,

# )

def printer(
            params=None,
            spec_param=None,
            datasets=None,
            models=None,
            colors=None,
            label=None,
            Conf_int=True,
            mean_acc_test=None,
            IC_acc_test=None,
            epochs=None,

):
    
    num_datasets = len(datasets)
    num_plots_per_row = int(sqrt(num_datasets))
    num_plots_per_col = (num_datasets + num_plots_per_row - 1) // num_plots_per_row

    # Création de la figure et des axes
    fig, axes = plt.subplots(num_plots_per_col, num_plots_per_row, figsize=(12, 12))
    # fig, axes = plt.subplots(((len(datasets))), figsize=(16, 16))
    string1='Précision de '

    for i,dataset in enumerate((datasets)):  
            print(dataset['dataset_name'])  
        # Boucle pour créer chaque subplot
            # X,Y=dataset['X'],dataset['Y']
            if 
                window_left = i // num_plots_per_row
                window_right = i % num_plots_per_row
                ax = axes[window_left, window_right]
                # X=from_torch_to_Datagrid(X)
            # T=X.data_matrix.shape[1]     
            for j,hyperparams in enumerate(params):
                # hyperparams.batch_size=X.shape[0]
                
                for k,model in enumerate(models):

                    print(model+label[k])
                    # if len(unique(Y.cpu()))<Y.shape[0]//2:
                    #     spec_param[k].loss=nn.CrossEntropyLoss()
                    # else:
                    #     spec_param[k].loss=nn.MSELoss()
                    #     spec_param[k].lr=10*spec_param[k].lr

                    # spec_param[k].batch_size=X.shape[0]//4

                    ax.plot(np.arange(hyperparams.n_epochs+1)[1:epochs[1]-epochs[0]+1],mean_acc_test[epochs[0]:epochs[1],i,j,k], label=string1+model+label[k],color=colors[k])
                    if Conf_int:
                        ax.plot(np.arange(hyperparams.n_epochs+1)[1:epochs[1]-epochs[0]+1],IC_acc_test[epochs[0]:epochs[1],:,i,j,k],linestyle="dashed",color=colors[k])
                    ax.set_title(dataset['dataset_name'])
                    ax.set_xlabel("epochs")
                    ax.set_ylabel("Pourcentage bien classés dans l'ensemble de test")
                    ax.legend(loc="lower right")       

    # plt.tight_layout()  # Ajuster automatiquement les espacements entre les subplots
    
    plt.show()
    return fig


def Compare_n_datas(params=None,
            datasets=None,
            models=None,
            supra_epochs=50,
            spec_param=None,
            alpha=0.95,
            checkpoints_number=None,
            colors=None):
    
    # label=[" "," avec dérivées"]
    # fig, axes = plt.subplots(int(floor(sqrt(len(datasets)))),int(floor(sqrt(len(datasets)))), figsize=(16, 16))
    num_datasets = len(datasets)
    num_plots_per_row = int(sqrt(num_datasets))
    num_plots_per_col = (num_datasets + num_plots_per_row - 1) // num_plots_per_row

    # Création de la figure et des axes
    fig, axes = plt.subplots(num_plots_per_col, num_plots_per_row, figsize=(12, 12))
    # fig, axes = plt.subplots(((len(datasets))), figsize=(16, 16))
    string1='Précision de '

    for i,dataset in enumerate((datasets)):   
        print(dataset['dataset_name']) 
        
    # Boucle pour créer chaque subplot
        X,Y=dataset['X'],dataset['Y']
        data_step=((X.shape[0])//checkpoints_number)
        window_left = i // num_plots_per_row
        window_right = i % num_plots_per_row
        ax = axes[window_left, window_right]
        n_datas=len(range(2,X.shape[0]*3//4,data_step))
        
        for j,hyperparams in enumerate(params):

            
            for k,model in enumerate(models):
                print(model)
                if len(unique(Y.cpu()))<Y.shape[0]//2:
                    spec_param[k].loss=nn.CrossEntropyLoss()
                else:
                    spec_param[k].loss=nn.MSELoss()
                    spec_param[k].lr=spec_param[k].lr*10

        
                monte_carlo_test_acc=torch.zeros(n_datas,supra_epochs,len(datasets),len(params),len(models))
                mean_acc_test=torch.zeros(n_datas,len(datasets),len(params),len(models))
                IC_acc_test=torch.zeros(n_datas,2,len(datasets),len(params),len(models))
                
                
                monte_carlo_test_acc[:,:,i,j,k],mean_acc_test[:,i,j,k],IC_acc_test[:,:,i,j,k]=Hyperparameter_Test_n_data(model_class=model,

                    hyperparameters=spec_param[k],
                    supra_epochs=supra_epochs,
                    X=X,
                    Y=Y,
                    alpha=alpha,
                    data_step=data_step,
                    )
                
                
                # ax.plot(np.arange(hyperparams.n_epochs+1)[1:],mean_acc_test[:,i,j,k], label=string1+model+' '+"hyperparamètre"+str(j+1),color=colors[j])
                # ax.plot(np.arange(hyperparams.n_epochs+1)[1:],IC_acc_test[:,:,i,j,k],linestyle="dashed",color=colors[j])
                ax.plot((np.arange(n_datas+1)[1:])*data_step,mean_acc_test[:,i,j,k], label=string1+model+label[j],color=colors[k])
                ax.plot((np.arange(n_datas+1)[1:])*data_step,IC_acc_test[:,:,i,j,k],linestyle="dashed",color=colors[k])
                # ax.plot(np.arange(hyperparams.n_epochs+1)[1:],IC_acc_test[:,:,i,j,k],linestyle="dashed",color="")
                # ax.plot(np.arange(hyperparams.n_epochs+1)[1:],mean_acc_test[:,i,j,k], label=string1+model+' '+str(j+1)+"ème hyperparamètre")
                ax.set_title(dataset['dataset_name'])
                ax.set_xlabel("Nombre de données d'entrainement ")
                ax.set_ylabel("Accuracy test max atteinte")
                ax.legend(loc="lower right")
            




        

    plt.tight_layout()  # Ajuster automatiquement les espacements entre les subplots
    
    plt.show()

    return fig,monte_carlo_test_acc,mean_acc_test,IC_acc_test



def Hyper_parameter_GridSearch(hyperparams,parameter, grid,model_class,x,y,supra_epochs=50):
    Final_acc = torch.tensor([0])
    norm=torch.zeros(len(grid))
    Optimum_parameter = grid[0]
    if len(unique(y.cpu()))<y.shape[0]//2:
        output_size=len(unique(y.cpu()))
        
        # hyperparams.loss=nn.CrossEntropyLoss()
    else:
        output_size=y.shape[1]
        hyperparams.loss=nn.MSELoss()
        
    
    
    # Obtenir l'attribut correspondant au paramètre spécifié
    attribute = getattr(hyperparams, parameter)
    
    for i,value in enumerate(grid):
        # Modifier la valeur de l'attribut de la classe HyperParameters
        setattr(hyperparams, parameter, value)
        print(parameter,"=",value)
        # Utiliser l'instance de HyperParameters pour effectuer les tests
        monte_carlo_test_acc,monte_carlo_train_acc,mean_acc_train,var_acc_train,IC_acc_train, mean_acc_test,var_acc_test,IC_acc_test = Hyperparameter_Test(supra_epochs=supra_epochs,hyperparameters=hyperparams,model_class=model_class,X=x,Y=y)
        if torch.norm(monte_carlo_test_acc)>torch.norm(Final_acc.float()):
            Final_acc=mean_acc_test
            hyperparams.parameter=value
    
    return hyperparams,Final_acc


def Hyperparameter_Search(hyperparams, grids, parameters,model_class,x,y,supra_epochs=25):

    best_parameters = hyperparams
    best_accuracy = 0.0
    mean_acc_base=0.0
    var_acc=0.0
    
    # Boucle sur les paramètres
    for param in tqdm(parameters):
        # Vérifier si le paramètre est dans la grille
        if param in grids:
            grid_values = grids[param]  # Récupérer les valeurs de la grille pour le paramètre donné

            # Boucle sur les valeurs de la grille pour le paramètre
            for value in grid_values:
                # Mettre à jour les hyperparamètres avec la valeur actuelle du paramètre
                setattr(best_parameters, param, value)

                # Appeler la fonction de Grid Search avec les paramètres spécifiés
                Opt_params,Final_acc = Hyper_parameter_GridSearch(best_parameters,grid=grid_values,parameter=param,model_class=model_class,x=x,y=y,supra_epochs=supra_epochs)
                
                # Mettre à jour le meilleur résultat si nécessaire
                
                    
    return Opt_params,Final_acc

# def Hyperparameter_Test_Mse(hyperparameters,model_class,x,y,output_size=1,supra_epochs=50,alpha=1.96):

#     for epochs in tqdm(range(supra_epochs)):
        
#         from scipy.stats import norm

#         chiffre = alpha
#         quartile = norm.ppf((1 + chiffre) / 2)

        

#         x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,shuffle=True)    
#         ##Compilation de la classe 
#         T=x.shape[2]

#         Model=Compile_class(hyperparams=hyperparameters,output_size=output_size,model_class=model_class,x_train=x_train)
        
#         Loss_min_monte_carlo=torch.tensor([0])
#         mean_accuracy=torch.tensor([0])
#         train_fn = Compile_train(module=Model, hyperparams=hyperparameters,x_train=x_train,y_train=y_train)


#         for i in tqdm(range(hyperparameters.n_epochs)):
#                 train_fn(n_epochs=1)
#                 accuracy=nn.MSELoss()(Model(x_test),y_test)
#                 mean_accuracy=torch.cat([mean_accuracy,accuracy.cpu().unsqueeze(0)],dim=0)
#                 Loss_min=torch.min(mean_accuracy[1:].float())
#                 n_epochs_max_acc=torch.argmax(mean_accuracy)    
        
#         Loss_min_monte_carlo=torch.cat([Loss_min,Loss_min_monte_carlo.unsqueeze(0)],dim=0)
#         mean_mse=torch.mean(Loss_min_monte_carlo[1:].float())
#         var_mse=torch.var(Loss_min_monte_carlo[1:].float())
#         IC_mse=[mean_mse-quartile*sqrt(var_mse)/sqrt(supra_epochs),mean_mse+quartile*sqrt(var_mse)/sqrt(supra_epochs)]
# # print("Loss moyenne =",((torch.mean(mean_accuracy[1:].float()))).detach().cpu().numpy())  
#         # print("Loss min=",((torch.min(mean_accuracy[1:].float()))).detach().cpu().numpy())  
#         # print("Variance des précisions =",((torch.var(mean_accuracy[1:].float()))).detach().cpu().numpy())  
#         # # print(mean_accuracy.unsqueeze(1)[1:]) 
        
        
#     gc.collect()
#     torch.cuda.empty_cache()

#     return Model, mean_mse,var_mse,IC_mse
