import numpy as np 
import torch
from torch import nn
from torch.nn import functional as F
import math
from models import BaseVAE
from torch.utils import model_zoo 
from .types_ import *
from .tools import *

class VitVAE_Transformer(BaseVAE):
    """
    Args:
        name (str): Model name, e.g. 'B_16'
        pretrained (bool): Load pretrained weights
        in_channels (int): Number of channels in input data 
    References:
        [1] https://openreview.net/forum?id=YicbFdNTTy
    """
    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(
        self, 
        name: str = None, 
        pretrained: bool = False, 
        patches: int = 16,      
        dim: int = 128,
        ff_dim: int = 4096,
        num_heads: int = 8,
        num_layers: int = 6,
        learnable_parameter_dim: int = 64,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.01,
        representation_size: str = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        positional_embedding: str = '1d',
        in_channels: int = 3, 
        img_size: str = None, 
        latent_dim: str = None ,
        **kwargs
    ):
        super().__init__()   
        self.in_channels = in_channels
        #dim = learnable_parameter_dim * num_heads
        self.latent_dim = latent_dim
        #dim = self.latent_dim * num_heads
        # Image and patch sizes 
        self.hypermeter = kwargs["hypermeter"]
        self.image_size = img_size
        fh, fw = as_tuple(int(math.sqrt(patches))) # patch sizes
        gh, gw = self.image_size // fh, self.image_size // fw  # number of patches
        seq_len = fh * fw
 
        self.pre_encoder = nn.Conv2d(in_channels, dim, kernel_size=(gh, gw), stride=(gh, gw))

        pre_logits_size = dim * seq_len
        hidden_dims = 2048
        self.fc_pre = nn.Linear(pre_logits_size , hidden_dims)
 
        self.norm = nn.LayerNorm(hidden_dims, eps=1e-6)
        self.fc_mu = nn.Linear(hidden_dims , latent_dim)
        self.fc_var = nn.Linear(hidden_dims , latent_dim) 

        # Build Decoder
        modules = []
        hidden_dims = [32, 64, 128, 256, 512]

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
 
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Conv2d(hidden_dims[-1], out_channels=  self.in_channels,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())
 
    def encode(self, x: Tensor) -> List[Tensor]:
        b, c, fh, fw = x.shape 
        x = self.pre_encoder(x)   
        x = torch.flatten(x, start_dim=1) 
        x = self.fc_pre(x) 
        mu, log_var = self.fc_mu(x), self.fc_var(x)

        return [mu, log_var]

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z: Tensor): 
        result = self.decoder_input(z) 
        result = result.view(-1, 512, 2, 2) 
        result = self.decoder(result) 
        result = self.final_layer(result) 

        return result

    def forward(self, input: Tensor, **kwargs) -> Tensor:  
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        self.lantent_z = mu.detach()  
        output = self.decode(z) 
        return  [output, input, mu, log_var]
    
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
 
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        a1, a2, a3 = self.hypermeter
        
        comparechannels = 3
        recons = recons[:,:comparechannels,:,:]
        input = input[:,:comparechannels,:,:]

        recons_loss = self.in_channels * self.image_size * self.image_size * F.mse_loss(recons, input)

        kld_loss = kld_weight * torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        
        hashing_loss = (1 - mu ** 2).clamp(0).mean()
        
        loss =  recons_loss +  kld_loss + hashing_loss * a3  #+ distriloss  
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss, 'hashing_loss': hashing_loss * a3}

    def sample(self, num_samples:int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def multi_channel_test(self, input: Tensor):
        mu, log_var = self.encode(input)
        self.every_channel_kld_loss = (-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 0)).cpu().numpy()
        lantent_z = self.reparameterize(mu, log_var)
        sample_num = lantent_z.shape[0]        
        multi_test = {} 
        
        multi_test["kldloss"] = self.every_channel_kld_loss
        return multi_test
    
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]

