import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
import math
from skimage.segmentation import slic
from .. import modules


def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

def get_beta(batch_idx, m, beta_type, epoch, num_epochs):
    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    return beta

def IoU(pred, true, smooth=1e-6):

    pred = (pred>0.5)
    true = (true>0.5)

    intersection = torch.logical_and(pred, true).sum()
    union = torch.logical_or(pred, true).sum()
    
    return  (intersection + smooth) / (union + smooth)  # We smooth our devision to avoid 0/0

    
def get_TI(pred, true, alpha=0.5, beta=0.5, smooth=1, gamma=1):
    
    pred = (pred>0.5).astype(np.uint8)
    true = (true>0.5).astype(np.uint8)
    
    #True Positives, False Positives & False Negatives
    TP = torch.logical_and(pred, true).sum()    
    FP = torch.logical_and(pred, torch.logical_not(true)).sum()
    FN = torch.logical_and(torch.logical_not(pred), true).sum()
    
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
    FocalTversky = (1 - Tversky)**gamma
    
    # if self.val:
    #     return FocalTversky
    # else:
    return FocalTversky
    

def get_aleatoric(preds):
    preds = torch.stack(preds)
    return torch.mean(torch.sum(preds*torch.nan_to_num(torch.log(preds)), dim=1), dim=0)
    
def sp_mean(out, segments):
    out_2 = torch.zeros_like(out)
    for i in torch.unique(segments):
        out_2[torch.where(segments==i)] = torch.mean(out[torch.where(segments==i)], dim=0)
    
    return out_2

def get_epistemic(preds):
    p_t = torch.stack([torch.flatten(i, start_dim=1) for i in preds])
    p_bar = torch.flatten(torch.mean(p_t, dim=0), start_dim=1)
    div = 0
    for i in preds:
        div += torch.max(torch.matmul((torch.flatten(i, start_dim=1)-p_bar).T.cuda(), (torch.flatten(i, start_dim=1)-p_bar).cuda()))

    return div/len(preds)

def get_div(preds):
    I = math.prod(preds[0].shape[:2])
    x = 0
    count = 0
    for i in range(len(preds)):
        for j in range(i, len(preds)):
            x += preds[i]*torch.log(preds[i]/preds[j])
            count += 1
    # print(torch.max(torch.nan_to_num(x), dim=0)[0])
    return (1/(I*count))*(torch.max(torch.nan_to_num(x), dim=0)[0])

def get_ENT(preds):
    preds = torch.mean(torch.stack(preds), dim=0)
    return torch.sum(preds*torch.log2(preds) + (1-preds)*torch.log2(1-preds))

def get_MAR(preds):
    preds = torch.mean(torch.stack(preds), dim=0)
    return torch.max(F.softmax(preds))

def get_RAND(preds):
    return torch.rand(1)[0]

def get_entropies(preds, mode="smart"):
            
    if mode=='smart':
        return get_epistemic(preds)
    elif mode=="MAR":
        return get_MAR(preds)
    elif mode=="ENT":
        return get_ENT(preds)
    elif mode=="RAND":
        return get_RAND(preds)
    else:
        return None

def _kl_loss(mu_0, log_sigma_0, mu_1, log_sigma_1) :
    """
    An method for calculating KL divergence between two Normal distribtuion.
    Arguments:
        mu_0 (Float) : mean of normal distribution.
        log_sigma_0 (Float): log(standard deviation of normal distribution).
        mu_1 (Float): mean of normal distribution.
        log_sigma_1 (Float): log(standard deviation of normal distribution).
   
    """
    kl = log_sigma_1 - log_sigma_0 + (torch.exp(log_sigma_0)**2 + (mu_0-mu_1)**2)/(2*math.exp(log_sigma_1)**2) - 0.5
    return kl.sum()

def bayesian_kl_loss(model, reduction='mean', last_layer_only=False) :
    """
    An method for calculating KL divergence of whole layers in the model.
    Arguments:
        model (nn.Module): a model to be calculated for KL-divergence.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'mean'``: the sum of the output will be divided by the number of
            elements of the output.
            ``'sum'``: the output will be summed.
        last_layer_only (Bool): True for return only the last layer's KL divergence.    
        
    """
    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    kl = torch.Tensor([0]).to(device)
    kl_sum = torch.Tensor([0]).to(device)
    n = torch.Tensor([0]).to(device)

    for m in model.modules() :
        if isinstance(m, modules.conv.BayesConv2d):
            kl = _kl_loss(m.W_mu, m.W_rho, m.prior_mu, m.prior_log_sigma)
            kl_sum += kl
            n += len(m.W_mu.view(-1))

            if m.bias :
                kl = _kl_loss(m.bias_mu, m.bias_rho, m.prior_mu, m.prior_log_sigma)
                kl_sum += kl
                n += len(m.bias_mu.view(-1))

    if last_layer_only or n == 0 :
        return kl
    
    if reduction == 'mean' :
        return kl_sum/n
    elif reduction == 'sum' :
        return kl_sum
    else :
        raise ValueError(reduction + " is not valid")