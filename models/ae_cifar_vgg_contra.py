import numpy as np 
import torch
from torch import nn
from torch.nn import functional as F
import math
from models import BaseVAE
from torch.utils import model_zoo 
from .types_ import *
from .tools import *
import torchvision.models as models


def simsam_dis(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

class AE_CIFAR_VGG_Contra(BaseVAE):
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
        '''
        # pretrained network out : b x 512 x 4 x 4 
        self.pretrained = models.vgg16(pretrained=True).features
        for param in self.pretrained.parameters():
            param.requires_grad = False 
        '''
        self.pretrained = Half_VGG()  
        in_dims = 2048 * 4   #25088 
        out_dims = 2048 * 4  #25088 
        self.in_dims = in_dims
        self.out_dims = out_dims 

        hidden_dims = [4096, 512, 128]

        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_dims , h_dim),
                    nn.ReLU())
            )
            in_dims = h_dim

        modules.append(nn.Sequential(
            nn.Linear(hidden_dims[-1] , self.latent_dim) ,
            nn.BatchNorm1d(self.latent_dim)
        ))
        self.pre_encoder = nn.Sequential(*modules) 
        self.proj = projection_MLP(in_dim = self.latent_dim, hidden_dim=128, out_dim=self.latent_dim) 
        # projection head
        """
        self.predictor = nn.Sequential(nn.Linear(self.latent_dim, 512, bias=False), 
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), 
                               nn.Linear(512, self.latent_dim, bias=True)
                               )
        """
        self.predictor = nn.Sequential(IRConv2d(in_channels=self.latent_dim, out_channels=512, kernel_size = 1, padding= 0, bias=True),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True), 
                        IRConv2d(in_channels=512, out_channels=self.latent_dim, kernel_size = 1, padding= 0, bias=True)
                        ) 
        
        # Build Encoder
        
        in_dims = self.latent_dim  
        hidden_dims = [4096, 512, 128]
        hidden_dims.reverse()
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_dims , h_dim),
                    nn.ReLU())
            )
            in_dims = h_dim
  
        modules.append(nn.Sequential(nn.Linear(hidden_dims[-1] , self.out_dims)))
        self.decoder = nn.Sequential(*modules) 

    def encode(self, x: Tensor) -> Tensor: 
        x = self.pre_encoder(x)  
        mu = self.fc_mu(x) 
        return mu, mu
        
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor: 
        return mu

    def decode(self, z: Tensor):
        return self.decoder(z)

    def train_step(self, batch: List[Tensor], **kwargs) -> Tensor:
        pos1, pos2, target = batch
        output1, input1, mu1, log_var1, z1, out1 = self.forward(pos1)
        output2, input2, mu2, log_var2, z2, out2 = self.forward(pos2)
        return [output1, input1, mu1, log_var1, z1, out1, output2, input2, mu2, log_var2, z2, out2]

    def forward(self, input: Tensor, **kwargs) -> Tensor:   
        input = self.pretrained(input)
        _inputshape = input.shape
        #print(input.shape)
        input = torch.flatten(input, start_dim=1)  
        mu = log_var = self.pre_encoder(input)
        z = mu#self.proj(mu)

        feature = z.reshape(z.shape[0], z.shape[1], 1, 1) 
        feature = self.predictor(feature)
        feature = torch.flatten(feature, start_dim=1) 
        #feature = F.normalize(feature, dim=-1)

        self.lantent_z = feature.detach()

        output = self.decoder(mu)
        return  [output.reshape(_inputshape), input.reshape(_inputshape), mu, log_var, z, feature]
    
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        output1, input1, mu1, log_var1, z1, out1, output2, input2, mu2, log_var2, z2, out2 = args 
        
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        optimizer_idx = kwargs['optimizer_idx']
        a1, a2, a3, a4, temperature = self.hypermeter
  
        info_loss = simsam_dis(out1, z2) / 2 + simsam_dis(out2, z1) / 2
        distri_loss = (1 - (torch.cat([out1, out2], dim=0))**2).clamp(0).mean() 
        
        input = torch.cat([input1, input2], dim=0)
        recons = torch.cat([output1, output2], dim=0)
        recons_loss = F.mse_loss(input, recons) 
        loss = a1 * recons_loss + info_loss * a3 + distri_loss * a4
        if self.num_iter % 13 == 0:
            print("recons_loss:{:.4f} info_loss:{:.4f} distri_loss:{:.4f}  ".format(recons_loss.item(), info_loss.item(), distri_loss.item()))

        return {'loss': loss, 'recons_loss':recons_loss }
     
    def test(self, x: Tensor, **kwargs) -> Tensor:   
        
        x = self.pretrained(x) 
        x = torch.flatten(x, start_dim=1)  
        x = self.pre_encoder(x)
        #x = self.proj(x)
        x = x.reshape(x.shape[0], x.shape[1], 1, 1) 
        x = self.predictor(x)
        x = torch.flatten(x, start_dim=1) 
        #x = F.normalize(x, dim=-1)

        return x