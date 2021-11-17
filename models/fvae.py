import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils
import numpy as np 

from models import BaseVAE
from torch.utils import model_zoo 
from .types_ import *
from .tools import *

class PlanarFlow(nn.Module):
    def __init__(self, dim):
        """Instantiates one step of planar flow.

        Args:
            dim: input dimensionality.
        """
        super(PlanarFlow, self).__init__()

        self.linear_u = nn.Linear(dim, dim)
        self.linear_w = nn.Linear(dim, dim)
        self.linear_b = nn.Linear(dim, 1)

    def forward(self, x, v):
        """Forward pass.

        Args:
            x: input tensor (B x D).
            v: output from last layer of encoder (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """
        u, w, b = self.linear_u(v), self.linear_w(v), self.linear_b(v)

        def m(x):
            return F.softplus(x) - 1.
        def h(x):
            return torch.tanh(x)
        def h_prime(x):
            return 1. - h(x)**2

        inner = (w * u).sum(dim=1, keepdim=True)
        u = u + (m(inner) - inner) * w / (w * w).sum(dim=1, keepdim=True)
        activation = (w * x).sum(dim=1, keepdim=True) + b
        x = x + u * h(activation)
        psi = h_prime(activation) * w
        log_det = torch.log(torch.abs(1. + (u * psi).sum(dim=1, keepdim=True)))

        return x, v, log_det

class RadialFlow(nn.Module):
    def __init__(self, dim):
        """Instantiates one step of radial flow.

        Args:
            dim: input dimensionality.
        """
        super(RadialFlow, self).__init__()

        self.linear_a = nn.Linear(dim, 1)
        self.linear_b = nn.Linear(dim, 1)
        self.linear_c = nn.Linear(dim, dim)
        self.d = dim

    def forward(self, x, v):
        """Forward pass.

        Args:
            x: input tensor (B x D).
            v: output from last layer of encoder (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """
        a, b, c = self.linear_a(v), self.linear_b(v), self.linear_c(v)

        def m(x):
            return F.softplus(x)
        def h(r):
            return 1. / (a + r)
        def h_prime(r):
            return -h(r)**2

        a = torch.exp(a)
        b = -a + m(b)
        r = (x - c).norm(dim=1, keepdim=True)
        tmp = b * h(r)
        x = x + tmp * (x - c)
        log_det = (self.d - 1) * torch.log(1. + tmp) + torch.log(1. + tmp + b * h_prime(r) * r)

        return x, v, log_det

class HouseholderFlow(nn.Module):
    def __init__(self, dim):
        """Instantiates one step of householder flow.

        Args:
            dim: input dimensionality.
        """
        super(HouseholderFlow, self).__init__()

        self.linear_v = nn.Linear(dim, dim)

    def forward(self, x, v):
        """Forward pass.

        Args:
            x: input tensor (B x D).
            v: output from last layer of encoder (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """
        v = self.linear_v(v)
        [B, D] = list(v.size())
        outer = v.reshape(B, D, 1) * v.reshape(B, 1, D)
        v_sqr = (v * v).sum(dim=1)
        H = torch.eye(D).cuda() - 2 * outer / v_sqr.reshape(B, 1, 1)
        x = (H * x.reshape(B, 1, D)).sum(dim=2)
        
        return x, v, 0

class Flow(nn.Module):
    def __init__(self, dim, type, length):
        """Instantiates a chain of flows.

        Args:
            dim: input dimensionality.
            type: type of flow.
            length: length of flow.
        """
        super(Flow, self).__init__()

        if type == 'planar':
            self.flow = nn.ModuleList([PlanarFlow(dim) for _ in range(length)])
        elif type == 'radial':
            self.flow = nn.ModuleList([RadialFlow(dim) for _ in range(length)])
        elif type == 'householder':
            self.flow = nn.ModuleList([HouseholderFlow(dim) for _ in range(length)])
        else:
            self.flow = nn.ModuleList([])

    def forward(self, x, v):
        """Forward pass.

        Args:
            x: input tensor (B x D).
            v: output from last layer of encoder (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """
        [B, _] = list(x.size())
        log_det = torch.zeros(B, 1).cuda()
        for i in range(len(self.flow)):
            x, v, inc = self.flow[i](x, v)
            log_det = log_det + inc

        return x, log_det

class GatedLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        """Instantiates a gated MLP layer.

        Args:
            in_dim: input dimensionality.
            out_dim: output dimensionality.
        """
        super(GatedLayer, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.gate = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Sigmoid())

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor (B x D).
        Returns:
            transformed x.
        """
        return self.linear(x) * self.gate(x)

class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, gate):
        """Instantiates an MLP layer.

        Args:
            in_dim: input dimensionality.
            out_dim: output dimensionality.
            gate: whether to use gating mechanism.
        """
        super(MLPLayer, self).__init__()

        if gate:
            self.layer = GatedLayer(in_dim, out_dim)
        else:
            self.layer = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor (B x D).
        Returns:
            transformed x.
        """
        return self.layer(x)

class FlowVAE(BaseVAE): 
    num_iter = 0
    def __init__(self, in_dim = 512 * 4 * 4,  out_dim = 3 * 32 * 32, layer = 2, hidden_dim=4096, latent_dim = 64, gate = 0, flow = 'radial', length = 8, **kwargs):
        """Instantiates a VAE.

        Args:
            dataset: dataset to be modeled.
            layer: number of hidden layers.
            in_dim: input dimensionality.
            hidden_dim: hidden dimensionality.
            latent_dim: latent dimensionality.
            gate: whether to use gating mechanism.
            flow: type of the flow (None if do not use flow).
            length: length of the flow.
        """
        super(FlowVAE, self).__init__()
        in_dim = out_dim
        # pretrained network out : b x 4096
        self.pretrained = Half_VGG() 
        #print(self.pretrained)
  
        self.hypermeter = kwargs["hypermeter"]
        self.dataset = 'cifar10'
        self.latent_dim = latent_dim
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)
        self.v = nn.Linear(latent_dim * 2, latent_dim)

        self.encoder = nn.ModuleList(
            [MLPLayer(in_dim, hidden_dim, gate)] + \
            [MLPLayer(hidden_dim, hidden_dim, gate) for _ in range(layer - 1)])
        self.flow = Flow(latent_dim, flow, length)

        self.decoder = nn.ModuleList(
            [MLPLayer(latent_dim, hidden_dim, gate)] + \
            [MLPLayer(hidden_dim, hidden_dim, gate) for _ in range(layer - 1)] + \
            [nn.Linear(hidden_dim, out_dim)] )

    def encode(self, x):
        """Encodes input.

        Args:
            x: input tensor (B x D).
        Returns:
            mean and log-variance of the gaussian approximate posterior.
        """
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
        return self.mean(x), self.log_var(x)

    def transform(self, mean, log_var):
        """Transforms approximate posterior.

        Args:
            mean: mean of the gaussian approximate posterior.
            log_var: log-variance of the gaussian approximate posterior.
        Returns:
            transformed latent codes and the log-determinant of the Jacobian.
        """
        std = torch.exp(.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)
        v = self.v(torch.cat((mean, log_var), dim=1))

        return self.flow(z, v)

    def decode(self, z):
        """Decodes latent codes.

        Args:
            z: latent codes.
        Returns:
            reconstructed input.
        """
        for i in range(len(self.decoder)):
            z = self.decoder[i](z)
        return z

    def sample(self, num_samples:int, current_device: int, **kwargs) -> Tensor: 
        """Generates samples from the prior.

        Args:
            size: number of samples to generate.
        Returns:
            generated samples.
        """
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        if self.dataset == 'mnist':	
        	return torch.sigmoid(self.decode(z))
        else:
        	return self.decode(z)
   
    def loss_function(self, *args, **kwargs):
        """Computes overall loss.

        Args:
            x: original input (B x D).
            x_hat: reconstructed input (B x D).
            mean: mean of the gaussian approximate posterior.
            log_var: log-variance of the gaussian approximate posterior.
            log_det: log-determinant of the Jacobian.
        Returns:
            sum of reconstruction and KL loss over the minibatch.
        """
        self.num_iter += 1 
        x_in, x_hat, mean, log_var, log_det, log_det_pre = args  
        #print(x_in.shape, x_hat.shape, mean.shape, log_var.shape, log_det.shape, log_det_pre.shape)
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        a1, a2, a3 = self.hypermeter
 
        reconloss, kldloss = self.loss(x_in, x_hat, mean, log_var, log_det) 
        loss = reconloss + kldloss - log_det.mean() - log_det_pre 
  
        if self.num_iter % 10 == 1:
            print("KLD:{:.4f} RecLoss:{:.4f} log_det_pre:{:.4f}".format(kldloss.item(), reconloss.item(), log_det_pre.item()))

        return {'loss': loss, 'Reconstruction_Loss':reconloss, 'KLD':kldloss, 'log_det_pre': log_det_pre}

    def reconstruction_loss(self, x, x_hat):
        """Computes reconstruction loss.

        Args:
            x: original input (B x D).
            x_hat: reconstructed input (B x D).
        Returns: sum of reconstruction loss over the minibatch.
        """
        if self.dataset == 'mnist':
            return nn.BCEWithLogitsLoss(reduction='none')(x_hat, x).sum(dim=1, keepdim=True)
        else:
            return nn.MSELoss(reduction='none')(x_hat, x).sum(dim=1, keepdim=True)

    def latent_loss(self, mean, log_var, log_det):
        """Computes KL loss.

        Args:
            mean: mean of the gaussian approximate posterior.
            log_var: log-variance of the gaussian approximate posterior.
            log_det: log-determinant of the Jacobian.
        Returns: sum of KL loss over the minibatch.
        """
        kl = -.5 * torch.sum(1. + log_var - mean.pow(2) - log_var.exp(), dim=1, keepdim=True)
        return kl

    def loss(self, x, x_hat, mean, log_var, log_det):
        """Computes overall loss.

        Args:
            x: original input (B x D).
            x_hat: reconstructed input (B x D).
            mean: mean of the gaussian approximate posterior.
            log_var: log-variance of the gaussian approximate posterior.
            log_det: log-determinant of the Jacobian.
        Returns:
            sum of reconstruction and KL loss over the minibatch.
        """
        return self.reconstruction_loss(x, x_hat).mean(), self.latent_loss(mean, log_var, log_det).mean()

    def forward(self, input: Tensor, **kwargs):
        """Forward pass.

        Args:
            x: input tensor (B x D).
        Returns:
            average loss over the minibatch.
        """
        x_in, log_det_pre = self.logit_transform(input) 
        B, C, H, W = x_in.shape 
        x_in = x_in.reshape(-1, C*H*W).detach() 
        log_det_pre = log_det_pre.detach()

        mean, log_var = self.encode(x_in)
        z, log_det = self.transform(mean, log_var)
        self.lantent_z = z
        x_hat = self.decode(z) 
        #print(x_in.shape, x_hat.shape, mean.shape, log_var.shape, log_det.shape, log_det_pre.shape)
        return [x_in, x_hat, mean, log_var, log_det, log_det_pre] 
 
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[1].reshape(-1, 3, 32, 32)

    def train_step(self, batch, **kwargs): 
        input, target = batch  
        return self.forward(input, **kwargs) 

    def logit_transform(self, x, constraint=0.9, reverse=False):
        '''Transforms data from [0, 1] into unbounded space.

        Restricts data into [0.05, 0.95].
        Calculates logit(alpha+(1-alpha)*x).

        Args:
            x: input tensor.
            constraint: data constraint before logit.
            reverse: True if transform data back to [0, 1].
        Returns:
            transformed tensor and log-determinant of Jacobian from the transform.
            (if reverse=True, no log-determinant is returned.)
        '''
        if reverse:
            x = 1. / (torch.exp(-x) + 1.)    # [0.05, 0.95]
            x *= 2.             # [0.1, 1.9]
            x -= 1.             # [-0.9, 0.9]
            x /= constraint     # [-1, 1]
            x += 1.             # [0, 2]
            x /= 2.             # [0, 1]
            return x, 0
        else: 
            [B, C, H, W] = list(x.size())
            # dequantization
            noise = distributions.Uniform(0., 1.).sample((B, C, H, W)).to(x.device)
      

            x = (x * 255. + noise) / 256.
            
            # restrict data
            x *= 2.             # [0, 2]
            x -= 1.             # [-1, 1]
            x *= constraint     # [-0.9, 0.9]
            x += 1.             # [0.1, 1.9]
            x /= 2.             # [0.05, 0.95]

            # logit data
            logit_x = torch.log(x) - torch.log(1. - x)

            # log-determinant of Jacobian from the transform
            pre_logit_scale = torch.tensor(
                np.log(constraint) - np.log(1. - constraint))
            log_diag_J = F.softplus(logit_x) + F.softplus(-logit_x) \
                - F.softplus(-pre_logit_scale)  
            return logit_x, torch.sum(log_diag_J, dim=(1, 2, 3)).mean()
