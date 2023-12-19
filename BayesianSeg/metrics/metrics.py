import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
from torchbnn.utils import epistemic_uncertainty, aleatoric_uncertainty
import torchbnn.nn as bnn
from ..misc import *

class IoU:
    def __init__(self, smooth=1e-6, cutoff=0.5):
        self.smooth=smooth
        self.cutoff=cutoff

    def __repr__(self):
        return "IoU"
    
    def __str__(self):
        return "IoU"

    def __call__(self, pred, true):
        pred = (pred.detach()>self.cutoff)
        true = (true.detach()>self.cutoff)

        intersection = torch.logical_and(pred, true).sum()
        union = torch.logical_or(pred, true).sum()
        
        return  (intersection + self.smooth) / (union + self.smooth)
    

class TI:
    def __init__(self, alpha=0.5, beta=0.5, smooth=1, gamma=1):
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.gamma = gamma
    
    def __repr__(self):
        return "TI"
    
    def __str__(self):
        return "TI"

    def __call__(self, pred, true):
        
        pred = (pred>0.5)
        true = (true>0.5)
        
        #True Positives, False Positives & False Negatives
        TP = torch.logical_and(pred, true).sum()    
        FP = torch.logical_and(pred, torch.logical_not(true)).sum()
        FN = torch.logical_and(torch.logical_not(pred), true).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)  
        FocalTversky = (1 - Tversky)**self.gamma
        
        return FocalTversky

class Uncertainty:
    @staticmethod
    def aleatoric(model, input_im, num_runs):
        if not isinstance(model, bnn.BayesModule):
            raise ValueError("Input model must be of type torchbnn.nn.BayesModule for aleatoric uncertainty!")
        model.freeze()
        pred = normalize(model(input_im))
        return aleatoric_uncertainty(pred).squeeze()

    @staticmethod
    def epistemic(model, input_im, num_runs):
        if not isinstance(model, bnn.BayesModule):
            raise ValueError("Input model must be of type torchbnn.nn.BayesModule for epistemic uncertainty!")
        model.unfreeze()
        pred = torch.vstack([normalize(model(input_im)) for _ in range(num_runs)])
        return epistemic_uncertainty(pred).squeeze()

    @staticmethod
    def div(model, input_im, num_runs):
        if not isinstance(model, bnn.BayesModule):
            raise ValueError("Input model must be of type torchbnn.nn.BayesModule for predictive divergence!")
        model.unfreeze()
        pred = torch.vstack([normalize(model(input_im)) for _ in range(num_runs)]).var(dim=0).mean(dim=0)
        return pred

    @staticmethod
    def ent(model, input_im, num_runs):
        model.freeze()
        pred = torch.mean(torch.vstack([normalize(model(input_im)) for _ in range(num_runs)]), dim=0)
        u = -torch.sum(pred*pred.log(), dim=0)
        return u

    @staticmethod
    def mar(model, input_im, num_runs):
        if not isinstance(model, bnn.BayesModule):
            raise ValueError("Input model must be of type torchbnn.nn.BayesModule for predictive divergence!")
        model.freeze()
        u = torch.vstack([normalize(model(input_im)) for _ in range(num_runs)]).mean(dim=0).mean(dim=0)
        return u


    @staticmethod
    def rand():
        return torch.rand(1)[0].item()


class Sampler:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, model, input_im, num_runs, alpha=0.2):
        if self.mode=='smart':
            return Uncertainty.epistemic(model, input_im, num_runs).flatten(start_dim=1).max() + alpha*Uncertainty.aleatoric(model, input_im, num_runs).flatten(start_dim=1).max() + Uncertainty.div(model, input_im, num_runs).flatten(start_dim=1).max()
        elif self.mode=="epi_div":
            return Uncertainty.epistemic(model, input_im, num_runs).flatten(start_dim=1).max() + Uncertainty.div(model, input_im, num_runs).flatten(start_dim=1).max()
        elif self.mode=="epi_alea":
            return Uncertainty.epistemic(model, input_im, num_runs).flatten(start_dim=1).max() + alpha*Uncertainty.aleatoric(model, input_im, num_runs).flatten(start_dim=1).max()
        elif self.mode=="epi":
            return Uncertainty.epistemic(model, input_im, num_runs).flatten(start_dim=1).max()
        elif self.mode=="alea":
            return Uncertainty.aleatoric(model, input_im, num_runs).flatten(start_dim=1).max()
        elif self.mode=="MAR":
            return Uncertainty.mar(model, input_im, num_runs).flatten(start_dim=1).max()
        elif self.mode=="ENT":
            return Uncertainty.ent(model, input_im, num_runs).flatten(start_dim=1).max()
        elif self.mode=="RAND":
            return Uncertainty.rand()
        else:
            raise NameError(f'{self.mode} sampling not found')