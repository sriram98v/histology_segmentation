import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import modules

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def __repr__(self):
        return "DiceLoss"
    
    def __str__(self):
        return "DiceLoss"

    def forward(self, inputs, targets, smooth=1):
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, reduction=True):
        super(DiceBCELoss, self).__init__()

    def __repr__(self):
        return "DiceBCELoss"
    
    def __str__(self):
        return "DiceBCELoss"

    def forward(self, inputs, targets, smooth=1):      
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
    
class CELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(CELoss, self).__init__()
        self.reduction = reduction

    def __repr__(self):
        return "CELoss"
    
    def __str__(self):
        return "CELoss"

    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, torch.argmax(targets, dim=1), reduction=self.reduction)


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def __repr__(self):
        return "IoULoss"
    
    def __str__(self):
        return "IoULoss"

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha=0.8, gamma=1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def __repr__(self):
        return "FocalLoss"
    
    def __str__(self):
        return "FocalLoss"

    def forward(self, inputs, targets):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
                       
        return focal_loss


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha=0.5, beta=0.5, smooth=1):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def __repr__(self):
        return "TverskyLoss"
    
    def __str__(self):
        return "TverskyLoss"

    def forward(self, inputs, targets):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)  
        
        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1, gamma=1):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.gamma = gamma
    
    def __repr__(self):
        return "FocalTverskyLoss"
    
    def __str__(self):
        return "FocalTverskyLoss"

    def forward(self, inputs, targets):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)  
        FocalTversky = (1 - Tversky)**self.gamma
                       
        return FocalTversky

class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1, eps=1e-9, ce_ratio=0.5):
        super(ComboLoss, self).__init__()
        self.ce_ratio = ce_ratio
        self.alpha = alpha
        self.smooth = smooth
        self.eps = eps

    def __repr__(self):
        return "ComboLoss"
    
    def __str__(self):
        return "ComboLoss"
    
    def forward(self, inputs, targets):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()    
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        inputs = torch.clamp(inputs, self.eps, 1.0 - self.eps)       
        out = - (self.alpha * ((targets * torch.log(inputs)) + ((1 - self.alpha) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (self.ce_ratio * weighted_ce) - ((1 - self.ce_ratio) * dice)
        
        return combo


class ELBO(nn.Module):
    def __init__(self, train_size):
        super(ELBO, self).__init__()
        self.train_size = train_size

    def __repr__(self):
        return "ELBO"
    
    def __str__(self):
        return "ELBO"

    def forward(self, input, target, kl, beta):
        assert not target.requires_grad
        # print(input.shape, target.shape)
        return F.nll_loss(input, torch.argmax(target, dim=1), reduction='mean') * self.train_size + beta * kl
    
class NLL_FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1, gamma=1):
        super(NLL_FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.gamma = gamma

    def __repr__(self):
        return "NLL_FocalTverskyLoss"
    
    def __str__(self):
        return "NLL_FocalTverskyLoss"

    def forward(self, inputs, targets):
        assert not targets.requires_grad

        TP_L = (targets*torch.log(inputs) + (1-targets)*torch.log(1-inputs)).sum()
        FP_L = ((1-targets)*torch.log(inputs) + targets*torch.log(1-inputs)).sum()

        Tversky = (TP_L + self.smooth) / (TP_L + self.alpha*FP_L + self.beta*FP_L + self.smooth) 
  
        NLL_FocalTversky = (1 - Tversky)**self.gamma
                       
        return NLL_FocalTversky

class ELBO_Jaccard(nn.Module):
    def __init__(self, dim=0):
        super(ELBO, self).__init__()
        self.dim = dim
    
    def __repr__(self):
        return "ELBO_Jaccard"
    
    def __str__(self):
        return "ELBO_Jaccard"
    
    def forward(self, input, target, kl, beta):
        assert not target.requires_grad
        # print(input.shape, target.shape)
        return F.cross_entropy(input, torch.argmax(target, dim=self.dim), reduction='mean')

class ELBO_FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1, gamma=1):
        super(ELBO_FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.gamma = gamma
        # self.train_size = train_size
        # self.val = False
    
    def __repr__(self):
        return "ELBO_FocalTverskyLoss"
    
    def __str__(self):
        return "ELBO_FocalTverskyLoss"

    def forward(self, inputs, targets, kl, beta=1e-7):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)  
        FocalTversky = (1 - Tversky)**self.gamma
        
        # if self.val:
        #     return FocalTversky
        # else:
        return FocalTversky + beta * kl

class ELBO_FocalLogTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1, gamma=1):
        super(ELBO_FocalLogTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.gamma = gamma
        # self.train_size = train_size
        # self.val = False

    def __repr__(self):
        return "ELBO_FocalLogTverskyLoss"
    
    def __str__(self):
        return "ELBO_FocalLogTverskyLoss"

    def forward(self, inputs, targets, kl, beta=1e-7):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (torch.log2(inputs) * targets).sum()    
        FP = ((1-targets) * torch.log2(inputs)).sum()
        FN = (targets * torch.log2(1-inputs)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)  
        FocalTversky = (1 - Tversky)**self.gamma
        
        # if self.val:
        #     return FocalTversky
        # else:
        return FocalTversky + beta * kl
           
class BKLLoss(nn.Module):
    """
    Loss for calculating KL divergence of baysian neural network model.
    Arguments:
        device (string): Specifies compute device to run KL Div computations on. 
                (NOTE: must be same device as model)
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'mean'``: the sum of the output will be divided by the number of
            elements of the output.
            ``'sum'``: the output will be summed.
        last_layer_only (Bool): True for return only the last layer's KL divergence.    
    """

    def __init__(self, device, reduction='mean', last_layer_only=False):
        super(BKLLoss, self).__init__()
        self.last_layer_only = last_layer_only
        self.device = torch.device(device)
        self.reduction = reduction

    def __repr__(self):
        return "BKLLoss"
    
    def __str__(self):
        return "BKLLoss"
    
    @staticmethod
    def kl_div_gauss(mu_0, sigma_0, mu_1, sigma_1) :
        """
        An method for calculating KL divergence between two Normal distribtuions p and q.
        .. math::
            KL(p, q) = \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2 \sigma_2^2} - \frac{1}{2} 
        
        Arguments:
            mu_0 (Float) : mean of normal distribution.
            log_sigma_0 (Float): log(standard deviation of normal distribution).
            mu_1 (Float): mean of normal distribution.
            log_sigma_1 (Float): log(standard deviation of normal distribution).
    
        """
        kl = (sigma_1/sigma_0).log() + (sigma_0**2 + (mu_0-mu_1)**2)/(2*(sigma_1**2)) - 0.5
        return kl.sum()

    def forward(self, model):
        """
        Arguments:
            model (nn.Module): a model to be calculated for KL-divergence.
        """
        kl = torch.Tensor([0]).to(self.device)
        kl_sum = torch.Tensor([0]).to(self.device)
        n = torch.Tensor([0]).to(self.device)

        for m in model.modules() :
            if isinstance(m, modules.conv.BayesConv2d):
                kl = self.kl_div_gauss(m.weight_mu, m.weight_sigma, m.prior_mu, m.prior_sigma)
                kl_sum += kl
                n += len(m.weight_mu.contiguous().view(-1))

                if m.bias :
                    kl = self.kl_div_gauss(m.bias_mu, m.bias_sigma, m.prior_mu, m.prior_sigma)
                    kl_sum += kl
                    n += len(m.bias_mu.contiguous().view(-1))

        if self.last_layer_only or n == 0 :
            return kl
        
        if self.reduction == 'mean' :
            return kl_sum/n
        elif self.reduction == 'sum' :
            return kl_sum
        else :
            raise ValueError(self.reduction + " is not valid")