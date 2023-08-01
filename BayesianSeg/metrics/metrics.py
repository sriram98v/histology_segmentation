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
        return torch.mean(torch.sum(preds*torch.nan_to_num(torch.log(preds)), dim=1)).item()

    @staticmethod
    def epistemic(preds):
        p_t = torch.flatten(preds, start_dim=-3, end_dim=-1)
        p_bar = torch.mean(p_t, dim=0)
        div = max([torch.matmul((p_t[i]-p_bar), (p_t[i]-p_bar).T).item() for i in range(p_t.shape[0])])

        return div/len(preds)

    @staticmethod
    def div(preds):
        I = math.prod(preds[0].shape[:2])
        x = 0
        count = 0
        for i in range(len(preds)):
            for j in range(i, len(preds)):
                x += preds[i]*torch.log(preds[i]/preds[j])
                count += 1
        return (1/(I*count))*(torch.max(torch.nan_to_num(x), dim=0)[0])

    @staticmethod
    def ent(preds):
        preds = torch.mean(preds, dim=0)
        return torch.sum(preds*torch.log2(preds) + (1-preds)*torch.log2(1-preds)).item()

    @staticmethod
    def mar(preds):
        preds = torch.mean(preds, dim=0)
        return torch.max(F.softmax(preds), dim=0).item()

    @staticmethod
    def rand(preds):
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