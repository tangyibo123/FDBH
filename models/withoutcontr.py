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

class WithoutContr(BaseVAE):
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

        backbone_dic = {"Cifar_AE" : Cifar_AE, "COCO_AE": COCO_AE}
        bacbone = backbone_dic[kwargs["backbone"]] 
        self.backbone = bacbone(latent_dim = self.latent_dim)

        #self.backbone = torch.load(kwargs['path']) 
        for p in self.backbone.pre_trained.parameters(): 
            p.requires_grad = False

        self.bth1d = nn.BatchNorm1d( self.latent_dim)
        #self.bat1d = nn.BatchNorm1d(self.latent_dim)
        self.projector = projection_MLP(in_dim = self.latent_dim, hidden_dim=32, out_dim=self.latent_dim)
 
 
        # projection head
        Conv2d = IRConv2d
        #Conv2d = nn.Conv2d
        self.predictor = nn.Sequential(Conv2d(in_channels=self.latent_dim, out_channels=512, kernel_size = 1, padding= 0, bias=True),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True), 
                        Conv2d(in_channels=512, out_channels=self.latent_dim, kernel_size = 1, padding= 0, bias=True)
                        ) 
        # Build Encoder
         
    def train_step(self, batch: List[Tensor], **kwargs) -> Tensor:
        pos1, pos2, target = batch
        output1, input1, mu1  = self.forward(pos1)
        output2, input2, mu2  = self.forward(pos2)
        return [output1, input1, mu1 , output2, input2, mu2  ]

    def forward(self, input: Tensor, **kwargs) -> Tensor:   
        input, output, recons_featrue = self.backbone(input) 
  
        return  [output, input, recons_featrue ]
    
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        output1, input1,  recons_featrue1, output2, input2,  recons_featrue2  = args 
        
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        optimizer_idx = kwargs['optimizer_idx']
        a1, a2, a3, a4, temperature = self.hypermeter
         
        recons_loss = F.mse_loss(torch.cat([input1, input2], dim=0), torch.cat([output1, output2], dim=0)) 
        loss = a1 * recons_loss  
        if self.num_iter % 13 == 0:
            print("recons_loss:{:.4f}   ".format(recons_loss.item() ))

        return {'loss': loss, 'recons_loss':recons_loss }
     
    def test(self, x: Tensor, **kwargs) -> Tensor:
        x = self.backbone.test(x) 
        x = torch.flatten(x, start_dim=1)  
 

        return x
   