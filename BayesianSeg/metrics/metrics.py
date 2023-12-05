import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
import math
from skimage.segmentation import slic
from .. import modules

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
    def aleatoric(preds):
        alea = (preds*(1-preds)).mean(dim=0).mean(dim=0)
        return -alea.mean().item()

    @staticmethod
    def epistemic(preds):
        # epistemic = preds.softmax(dim=1).mean(dim=0)
        return (preds*preds.log()).sum(dim=0).mean().item()

    @staticmethod
    def div(preds):
        preds = torch.flatten(preds, start_dim=-2, end_dim=-1).mean(dim=0).var(dim=1).mean().item()
        return preds

    @staticmethod
    def ent(preds):
        preds = torch.mean(preds, dim=0)
        return torch.sum(preds*torch.log2(preds) + (1-preds)*torch.log2(1-preds)).item()

    @staticmethod
    def mar(preds):
        preds = torch.mean(preds, dim=0)
        return torch.max(F.softmax(preds), dim=0).item()

    @staticmethod
    def rand():
        return torch.rand(1)[0].item()


class Sampler:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, preds, alpha=0.2):
        if self.mode=='smart':
            return Uncertainty.epistemic(preds) + alpha*Uncertainty.aleatoric(preds) + Uncertainty.div(preds)
        elif self.mode=="epi_div":
            return Uncertainty.epistemic(preds) + Uncertainty.div(preds)
        elif self.mode=="epi_alea":
            return Uncertainty.epistemic(preds) + alpha*Uncertainty.aleatoric(preds)
        elif self.mode=="epi":
            return Uncertainty.epistemic(preds)
        elif self.mode=="alea":
            return Uncertainty.aleatoric(preds)
        elif self.mode=="MAR":
            return Uncertainty.mar(preds)
        elif self.mode=="ENT":
            return Uncertainty.ent(preds)
        elif self.mode=="RAND":
            return Uncertainty.rand()
        else:
            raise NameError(f'{self.mode} sampling not found')