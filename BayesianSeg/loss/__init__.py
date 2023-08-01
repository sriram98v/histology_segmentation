from .losses import *

def Loss(method, device, **kwargs):
    match method:
        case "DiceLoss":
            return DiceLoss(kwargs["weight"], kwargs["size_average"])
        case "DiceBCELoss":
            return DiceBCELoss(kwargs["reduction"])
        case "CELoss":
            return CELoss(kwargs["reduction"])
        case "IoULoss":
            return IoULoss()
        case "FocalLoss":
            return FocalLoss(kwargs["weight"], kwargs["size_average"], kwargs["alpha"], kwargs["gamma"])
        case "TverskyLoss":
            return TverskyLoss(kwargs["weight"], kwargs["size_average"], kwargs["alpha"], kwargs["beta"], kwargs["smooth"])
        case "FocalTverskyLoss":
            return FocalTverskyLoss(kwargs["alpha"], kwargs["beta"], kwargs["smooth"], kwargs["gamma"])
        case "ComboLoss":
            return ComboLoss(kwargs["alpha"], kwargs["smooth"], kwargs["eps"], kwargs["ce_ratio"])
        case "ELBO":
            return ELBO(kwargs["train_size"])
        case "NLL_FocalTverskyLoss":
            return NLL_FocalTverskyLoss(kwargs["alpha"], kwargs["beta"], kwargs["smooth"], kwargs["gamma"])
        case "ELBO_Jaccard":
            return ELBO_Jaccard(kwargs["dim"])
        case "ELBO_FocalTverskyLoss":
            return ELBO_FocalTverskyLoss(kwargs["alpha"], kwargs["beta"], kwargs["smooth"], kwargs["gamma"])
        case "ELBO_FocalLogTverskyLoss":
            return ELBO_FocalLogTverskyLoss(kwargs["alpha"], kwargs["beta"], kwargs["smooth"], kwargs["gamma"])
        case "BKLLoss":
            return BKLLoss(device, kwargs["reduction"], kwargs["last_layer_only"])
        case _:
            raise NameError(f'Loss {method} not found')
