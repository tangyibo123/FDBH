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
 
class VitVAE_Cifar_VGG_IB(BaseVAE):
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
        
        # pretrained network out : b x 4096
        #self.pretrained = Half_Resnet()
        self.pretrained = Half_VGG()
        #print(self.pretrained)

        in_dims = 2048 * 4 
        out_dims = 2048 * 4 
        self.in_dims = in_dims
        self.out_dims = out_dims 
        hidden_dims = [512 * 4 , 512, 128, 32, 128, 128]

        # Build Encoder 
        modules = []  
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_dims , h_dim),
                    nn.ReLU())
            )
            in_dims = h_dim

        self.pre_encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] , self.latent_dim) 
        
        # Build Decoder
        in_dims = self.latent_dim  
        hidden_dims = [out_dims , 512 * 4 , 512, 128, 32, 128, 128]
        hidden_dims.reverse()
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_dims , h_dim),
                    nn.ReLU())
            )
            in_dims = h_dim

        self.decoder = nn.Sequential(*modules)
        print(self.decoder)

    def encode(self, x: Tensor) -> List[Tensor]:  
        x = self.pre_encoder(x)  
        mu = self.fc_mu(x)

        return mu
         
    def reparameterize(self, mu: Tensor) -> Tensor:
        return mu
        
    def decode(self, z: Tensor):
        return self.decoder(z)

    def train_step(self, batch, **kwargs) -> Tensor:  
        image, labels = batch
        return self.forward(image)

    def forward(self, input: Tensor, **kwargs) -> Tensor:  
        input = self.pretrained(input)
        _inputshape = input.shape
        #print(input.shape)
        input = torch.flatten(input, start_dim=1)  
        z = self.encode(input) 
        self.lantent_z = z.detach()
        output = self.decoder(z)    
        return  [output.reshape(_inputshape), input.reshape(_inputshape), z]
    
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:

        self.num_iter += 1
        recons = args[0]
        input = args[1]
        z = args[2] 
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        a1, a2, a3 = self.hypermeter
        
        recons_loss = F.mse_loss(recons, input) 
        contentloss = F.mse_loss(self.pretrained.distance(recons), self.pretrained.distance(input)) 
        loss =  a1 * recons_loss + a2 * contentloss 

        if self.num_iter % 10 == 0:
            print("RecLoss:{:.4f} contentloss:{:.4f}".format(recons_loss.item(), contentloss.item()))

        return {'loss': loss, 'Reconstruction_Loss':recons_loss}

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

        samples = self.decoder(z)
        return samples
 
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]

