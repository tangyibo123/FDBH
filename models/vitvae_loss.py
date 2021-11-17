import numpy as np 
import torch
from torch import nn
from torch.nn import functional as F
import math
from models import BaseVAE
from torch.utils import model_zoo 
from .types_ import *
from .tools import *

class VitVAE_loss(BaseVAE):
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

        # Patch embedding
        self.rgbpatch_embedding = nn.Conv2d(in_channels - 1, dim, kernel_size=(gh, gw), stride=(gh, gw))
        self.nirpatch_embedding = nn.Conv2d(1, dim, kernel_size=(gh, gw), stride=(gh, gw))
        # Class token
        if classifier == 'token':
            self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
            seq_len += 1
        
        # Positional embedding
        if positional_embedding.lower() == '1d':
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        else:
            raise NotImplementedError()
        
        # Transformer
        self.transformer = DoubleChannelTransformer(num_layers=num_layers, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate)
 
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # Representation layer
        if representation_size and load_repr_layer:
            self.pre_logits = nn.Linear(dim, representation_size)
            pre_logits_size = representation_size
        else:
            pre_logits_size = dim * seq_len

        # Lantent z
        
        self.norm = nn.LayerNorm(pre_logits_size, eps=1e-6)
        self.rgbfc_mu = nn.Linear(pre_logits_size , latent_dim)
        self.rgbfc_var = nn.Linear(pre_logits_size , latent_dim) 
        self.nirfc_mu = nn.Linear(pre_logits_size , latent_dim)
        self.nirfc_var = nn.Linear(pre_logits_size , latent_dim) 
         
        '''
        for i in range(latent_dim):
            self.fc_mu.append(nn.Linear(seq_len, 1))
            self.fc_var.append(nn.Linear(seq_len, 1))
        '''
        #self.fc_mu = nn.Linear(pre_logits_size, latent_dim)
        #self.fc_var = nn.Linear(pre_logits_size, latent_dim) 
        
        #self.tokenz = torch.randn(learnable_parameter_dim, requires_grad=True)
          
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
                            nn.Conv2d(hidden_dims[-1], out_channels=  in_channels,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

 
        # Initialize weights
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        #self.apply(_init) 

        #nn.init.constant_(self.fc_mu.weight, 0)
        #nn.init.constant_(self.fc_mu.bias, 0)
        #nn.init.constant_(self.fc_var.weight, 0)
        #nn.init.constant_(self.fc_var.bias, 0)

        #nn.init.normal_(self.decoderpos_embedding.pos_embedding, std=0.02)
        nn.init.normal_(self.positional_embedding.pos_embedding, std=0.02)  # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)
        nn.init.constant_(self.class_token, 0)
        #nn.init.constant_(self.class_token2, 0)

    def encode(self, input: Tensor) -> List[Tensor]:
        b, c, fh, fw = input.shape
        rgb = input[:,:3,:,:]
        nir = input[:,3:,:,:]
 
        rgb = self.rgbpatch_embedding(rgb)
        nir = self.nirpatch_embedding(nir)
        rgb = rgb.flatten(2).transpose(1, 2)  # b,gh*gw,d
        nir = nir.flatten(2).transpose(1, 2)  # b,gh*gw,d
  
        if hasattr(self, 'class_token'):
            rgb = torch.cat((self.class_token.expand(b, -1, -1), rgb), dim=1)  # b,gh*gw+1,d
            nir = torch.cat((self.class_token.expand(b, -1, -1), nir), dim=1)  # b,gh*gw+1,d

        # TODO: compare test
        if hasattr(self, 'positional_embedding'): 
            rgb = self.positional_embedding(rgb)  # b,gh*gw+1,d 
            nir = self.positional_embedding(nir)  # b,gh*gw+1,d 
        rgb, nir = self.transformer(rgb, nir)  # b,gh*gw+1,d
          
        rgb = torch.flatten(rgb, start_dim=1) 
        nir = torch.flatten(nir, start_dim=1) 
 
        rgb_mu = self.rgbfc_mu(rgb)[:, 0:self.latent_dim - 16] 
        rgb_var = self.rgbfc_var(rgb)[:, 0:self.latent_dim - 16] 
        nir_mu = self.nirfc_mu(nir)[:,  self.latent_dim - 16:] 
        nir_var = self.nirfc_var(nir)[:,  self.latent_dim - 16:] 

        mu =  (torch.cat((rgb_mu, nir_mu), dim=1)) * 2    # b,gh*gw+1,d
        log_var = torch.cat((rgb_var, nir_var), dim=1)  # b,gh*gw+1,d
        return [mu, log_var]

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z: Tensor):
        #if self.tokenz.device != z.device:
        #    self.tokenz = self.tokenz.to(z.device)
        #tokenz = self.tokenz * torch.ones((z.shape[0], self.tokenz.shape[0]), device = z.device)
         
        #z = torch.cat((z, tokenz), dim=1)
        #z = z + tokenz
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        """
        result = self.decoder_input(z)
        result = result.view(-1, 16, 128)
        
        # if hasattr(self, 'class_token'):
            #result = torch.cat((self.class_token2.expand(result.shape[0], -1, -1), result), dim=1)  # b,gh*gw,d
        #if hasattr(self, 'positional_embedding'): 
        #    result = self.decoderpos_embedding(result)  # b,gh*gw+1,d 

        result = self.decoder(result)#[:,1:,:]
        #result = result.view(-1, 8, 16, 16)  # b,gh*gw,d
        result = self.finliner(result)
        R = result[:,0::3,:]
        G = result[:,1::3,:]
        B = result[:,2::3,:]
        R = torch.reshape(R.view(-1, 4, 4, 8, 8).transpose(2, 3), (-1, 1, 32, 32))
        G = torch.reshape(G.view(-1, 4, 4, 8, 8).transpose(2, 3), (-1, 1, 32, 32))
        B = torch.reshape(B.view(-1, 4, 4, 8, 8).transpose(2, 3), (-1, 1, 32, 32)) 
        result = torch.cat((R,G,B), dim=1)
        result = self.final_layer(result)
        """

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

        '''
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        comparechannels = 3
        recons = recons[:,:comparechannels,:,:]
        input = input[:,:comparechannels,:,:]

        recons_loss = F.mse_loss(recons, input, size_average=False)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_loss  #+ distriloss 
        '''
        
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

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss, 'hashing_loss': torch.zeros([1])}

    def sample(self, num_samples:int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
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
