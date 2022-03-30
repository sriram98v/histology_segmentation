import cv2
import numpy as np

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def highlight_im(im, mask, threshold=0.5):
    threshold = int(threshold*255)
    mask = (mask*255).astype(np.uint8)
    ret, binary = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    
    contours,hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        cv2.drawContours(im, contour, -1, (255,0,0), thickness = 1)

    b,g,r = cv2.split(im)

    r = cv2.add(b, 90, dst = r, mask = binary, dtype = cv2.CV_8U)

    return cv2.merge((b,g,r), im)

def entropy(probs):
#     log_prob = np.log(out)
    return probs*np.nan_to_num(np.log2(probs))*-1

from torch import nn


class ModuleWrapper(nn.Module):
    """Wrapper for nn.Module with support for arbitrary flags and a universal forward pass"""

    def __init__(self):
        super(ModuleWrapper, self).__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def freeze_rho(self):
            for module in self.modules():
                if hasattr(module, 'train_mu'):
                    module.train_mu()

    def freeze_mu(self):
            for module in self.modules():
                if hasattr(module, 'train_var'):
                    module.train_var()

    def switch_custom_dropout(self, activate:bool=True, verbose:bool=False):

        for module in self.modules():
                if hasattr(module, 'switch_activate'):
                    module.switch_activate(activate)
        
        print(f"Switched all dropout to {activate}")

    def get_kl(self):
        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()
        
        return kl

    def mode(self, mode="infer"):
        for module in self.modules():
            if hasattr(module, 'switch_mode'):
                module.switch_mode(mode)


