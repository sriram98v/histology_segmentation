from .metrics import *

def Metric(method, **kwargs):
    match method:
        case "IoU":
            return IoU(kwargs["smooth"], kwargs["cutoff"])
        case "TI":
            return TI(kwargs["alpha"], kwargs["beta"], kwargs["smooth"], kwargs["gamma"])
        case _:
            raise NameError(f'Metric {method} not found')