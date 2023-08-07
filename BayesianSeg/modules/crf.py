import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianMask(nn.Module):
    """
    Break down Gaussian kernel (2nd part of appearance kernel) into CNN
    kj = (I(j) - I(i))**2/2*bandwidth**2, j#i
    but compute all maps instead of 1 kernel
    """
    def __init__(self, in_channels, kernel_size, bandwidth, iskernel=True):
        super(GaussianMask, self).__init__()
        assert bandwidth > 0, 'bandwidth of kernel must be > 0'
        assert kernel_size%2 != 0, 'kernel must be odd'
        self.in_channels = in_channels
        self.bandwidth = bandwidth
        self.iskernel = iskernel
        self.n_kernels = kernel_size**2-1
        self.kernel_weight = self._make_kernel_weight(in_channels, kernel_size, self.n_kernels).view(in_channels*self.n_kernels, 1, kernel_size, kernel_size)
        self.register_buffer("gauss_filter", self.kernel_weight)
        self.padding = kernel_size//2

    @staticmethod
    def make_onehot_kernel(kernel_size, index):
        """
        Make 2D one hot square kernel, i.e. h=w
        k[kernel_size, kernel_size] = 0 except k.view(-1)[index] = 1
        """
        kernel = torch.zeros(kernel_size, kernel_size)
        kernel.view(-1)[index] = 1
        return kernel.view(1, 1, kernel_size, kernel_size)

    def _make_kernel_weight(self, in_channels, kernel_size, n_kernels):
        #! Be carefull with contruct weight, otherwise, output will be mixed in unwanted order 
        kernel_weight = torch.zeros(in_channels, n_kernels, kernel_size, kernel_size)
        for i in range(n_kernels):
            index = i if i < (n_kernels//2) else i+1
            kernel_i = self.make_onehot_kernel(kernel_size, index)
            kernel_weight[:, i, :] = kernel_i
        return kernel_weight

    def forward(self, X):
        #compute (I(j)-I(i))**2/(2*bandwidth**2)
        batch_size, in_channels, H, W = X.shape

        Xj = F.conv2d(X, self.gauss_filter, bias=None, stride=1, padding=self.padding, dilation=1, groups=self.in_channels).view(batch_size, in_channels, self.n_kernels, H, W)
        if not self.iskernel:
            return Xj
        Xi = X.unsqueeze(dim=2)
        K = (Xj-Xi)**2 / (2*(self.bandwidth**2))
        K = torch.exp(-K)
        return K #size B*C*N_ker*H*W


class SpatialFilter(nn.Module):
    """
    Break down spatial filter (smoothest kernel) into CNN blocks
    refer: https://arxiv.org/pdf/1210.5644.pdf
    """
    def __init__(self, n_classes, kernel_size, theta_gamma):
        super(SpatialFilter, self).__init__()
        self.n_classes = n_classes
        self.padding = kernel_size//2
        self.kernel_weight = torch.Tensor(n_classes, 1, kernel_size, kernel_size).copy_(self.make_spatial_kernel(kernel_size, theta_gamma))
        self.register_buffer("spatial_filter", self.kernel_weight)
        
    @staticmethod
    def make_spatial_kernel(kernel_size, bandwidth, isreshape=True):
        """
        Make 2D square smoothness kernel, i.e. h=w
        k = 1/bandwidth * exp(-(pj-pi)**2/(2*bandwidth**2))
        pj, pi = location of pixel
        """
        assert bandwidth > 0, 'bandwidth of kernel must be > 0'
        assert kernel_size%2 != 0, 'kernel must be odd'
        p_end = (kernel_size-1)//2 #kernel center indices
        X = torch.linspace(-p_end, p_end, steps=kernel_size).expand(kernel_size, kernel_size)
        Y = X.clone().t()
        kernel = torch.exp(-(X**2+Y**2)/(2*(bandwidth**2)))
        kernel[p_end, p_end] = 0 #! due to the require of paper: j#i, thus when j=i, kernel=0
        if isreshape:
            return kernel.view(1, 1, kernel_size, kernel_size)
        return kernel

    def forward(self, Q):
        Qtilde = F.conv2d(Q, self.spatial_filter, bias=None, padding=self.padding, groups=self.n_classes)
        norm_weight = F.conv2d(torch.ones_like(Q), self.spatial_filter, bias=None, padding=self.padding, groups=self.n_classes)
        Qtilde = Qtilde / norm_weight
        return Qtilde


class BilateralFilter(nn.Module):
    """
    Break down bilateral filter (appearance kernel) into CNN blocks
    remember that exp(-a-b) =exp(-a)*exp(b)
    """
    def __init__(self, in_channels, n_classes, kernel_size, theta_alpha, theta_beta):
        super(BilateralFilter, self).__init__()
        #need 6 dims for later purpose
        kernel_weight = self.make_spatial_kernel(kernel_size, theta_alpha, isreshape=False)
        self.spatial_weight = kernel_weight[kernel_weight > 0].view(1, 1, 1, -1, 1, 1)
        self.register_buffer("bilateral_filter", self.spatial_weight)
        self.gauss_mask_I = GaussianMask(in_channels, kernel_size, theta_beta)
        self.guass_mask_Q = GaussianMask(n_classes, kernel_size, 1, iskernel=False)

    @staticmethod
    def make_spatial_kernel(kernel_size, bandwidth, isreshape=True):
        """
        Make 2D square smoothness kernel, i.e. h=w
        k = 1/bandwidth * exp(-(pj-pi)**2/(2*bandwidth**2))
        pj, pi = location of pixel
        """
        assert bandwidth > 0, 'bandwidth of kernel must be > 0'
        assert kernel_size%2 != 0, 'kernel must be odd'
        p_end = (kernel_size-1)//2 #kernel center indices
        X = torch.linspace(-p_end, p_end, steps=kernel_size).expand(kernel_size, kernel_size)
        Y = X.clone().t()
        kernel = torch.exp(-(X**2+Y**2)/(2*(bandwidth**2)))
        kernel[p_end, p_end] = 0 #! due to the require of paper: j#i, thus when j=i, kernel=0
        if isreshape:
            return kernel.view(1, 1, kernel_size, kernel_size)
        return kernel

    def forward(self, Q, I):
        #make masks for filters
        Ij = self.gauss_mask_I(I) #size B*C*N_ker*H*W
        Qj = self.guass_mask_Q(Q) #size B*N_class*N_ker*H*W
        Qj = Ij.unsqueeze(dim=2) * Qj.unsqueeze(dim=1) #size B*C*N_class*N_ker*H*W
        #multiply with spatial weight on N_ker dimension
        Qj = Qj * self.bilateral_filter
        #sum over spatial weight dimension
        Qtilde = Qj.sum(dim=3) ##size B*C*N_class*H*W, thus C=M in the paper
        #norm
        norm_weight = Ij * self.bilateral_filter.squeeze(dim=2) #size B*C*N_ker*H*W
        norm_weight = norm_weight.sum(dim=2) #size B*C*H*W
        Qtilde = Qtilde / norm_weight.unsqueeze(dim=2)
        return Qtilde


class MessagePassing(nn.Module):
    """
    Combine bilateral filter (appearance filter)
    and spatial filter to make message passing
    """
    def __init__(self, in_channels, n_classes, kernel_size=[3,],
                theta_alpha=[2.,], theta_beta=[2.,], theta_gamma=[2.,]):
        super(MessagePassing, self).__init__()
        assert len(theta_alpha) == len(theta_beta), 'theta_alpha and theta_beta have different lengths'
        # self.bilateralfilter = BilateralFilter(in_channels, n_classes, kernel_size, theta_alpha, theta_beta)
        # self.spatialfilter = SpatialFilter(n_classes, kernel_size, theta_gamma)
        self.n_bilaterals, self.n_spatials = len(theta_alpha), len(theta_gamma)
        for i in range(self.n_bilaterals):
            self.add_module( \
                'bilateral{}'.format(i), \
                BilateralFilter(in_channels, n_classes, kernel_size[i], theta_alpha[i], theta_beta[i]))
        for i in range(self.n_spatials):
            self.add_module(
                'spatial{}'.format(i), SpatialFilter(n_classes, kernel_size[i], theta_gamma[i]))

    def _get_child(self, child_name):
        return getattr(self, child_name)

    def forward(self, Q, I):
        # bilateralQ = self.bilateralfilter(Q, I) #B*n_bilaterals*N_class*H*W
        # spatialQ = self.spatialfilter(Q) #B*N_class*H*W
        filteredQ = []
        for i in range(self.n_bilaterals):
            tmp_bilateral = self._get_child('bilateral{}'.format(i))(Q, I)
            filteredQ.append(tmp_bilateral)
        for i in range(self.n_spatials):
            tmp_spatial = self._get_child('spatial{}'.format(i))(Q)
            filteredQ.append(tmp_spatial.unsqueeze(dim=1))
        Qtilde = torch.cat(filteredQ, dim=1) #B*(n_bilaterals+n_spatials)*N_class*H*W
        return Qtilde


class CRF(nn.Module):
    """ Break meanfields down as CNN and do iteration """
    def __init__(self, n_iter, in_channels, n_classes, kernel_size=[3, 3], theta_alpha=[1.5, 2.5], 
                 theta_beta=[1.5, 2.5], theta_gamma=[1.5,], returns="proba"):
        super(CRF, self).__init__()
        self.n_iter = n_iter
        self.n_classes = n_classes
        n_filters = in_channels * len(theta_alpha) + len(theta_gamma)
        self.returns = returns
        self.messagepassing = MessagePassing(
            in_channels, n_classes, kernel_size=kernel_size,
            theta_alpha=theta_alpha, theta_beta=theta_beta, theta_gamma=theta_gamma)
        self.weightfiltering = nn.Parameter(torch.Tensor(1, n_filters, n_classes, 1, 1))
        self.compatibilitytransf = nn.Conv2d( \
            n_classes, n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self._weight_initial()

    def _weight_initial(self):
        nn.init.kaiming_normal_(self.weightfiltering)
        nn.init.kaiming_normal_(self.compatibilitytransf.weight)

    def forward(self, U, I):
        Q = U.clone()
        for _ in range(self.n_iter):
            #normalize
            Q = Q.softmax(dim=1)
            #message passing
            Q = self.messagepassing(Q, I)
            #weight filtering
            Q = Q * self.weightfiltering
            Q = Q.sum(dim=1)
            #compatibility transform
            #need to minus Q*weight because sum(mu_l'l * Q_l') with l'#l
            Q = self.compatibilitytransf(Q) \
                - Q * self.compatibilitytransf.weight.squeeze().diag().view(1, self.n_classes, 1, 1)
            #adding unary
            Q = U - Q

        if self.returns == 'logits':
            output = Q
        elif self.returns == 'proba':
            output = F.softmax(Q, dim=1)
        elif self.returns == 'log-proba':
            output = F.log_softmax(Q, dim=1)
        else:
            raise ValueError("Attribute ``returns`` must be 'logits', 'proba' or 'log-proba'.")

        if self.n_classes == 1:
            output = output[:, 0] - output[:, 1] if self.returns == 'logits' else output[:, 0]
            output.unsqueeze_(1)

        return output

