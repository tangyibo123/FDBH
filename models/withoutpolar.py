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

class WithoutPolar(BaseVAE):
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
        output1, input1, mu1, log_var1, z1, out1 = self.forward(pos1)
        output2, input2, mu2, log_var2, z2, out2 = self.forward(pos2)
        return [output1, input1, mu1, log_var1, z1, out1, output2, input2, mu2, log_var2, z2, out2]

    def forward(self, input: Tensor, **kwargs) -> Tensor:   
        input, output, recons_featrue = self.backbone(input) 
        #contr_input = ((recons_featrue) + torch.sign(recons_featrue))
        contr_input = recons_featrue
        contr_middle = self.bth1d(contr_input) #self.projector(contr_input)
        
        contr_result = contr_middle.reshape(contr_middle.shape[0], contr_middle.shape[1], 1, 1) 
        contr_result = self.predictor(contr_result)  
        contr_result = torch.flatten(contr_result, start_dim=1)  

        return  [output, input, recons_featrue, contr_input, contr_middle, contr_result]
    
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        output1, input1,  recons_featrue1, contr_input1, contr_middle1, contr_result1, output2, input2,  recons_featrue2, contr_input2, contr_middle2, contr_result2 = args 
        
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        optimizer_idx = kwargs['optimizer_idx']
        a1, a2, a3, a4, temperature = self.hypermeter
        
        contr_result = torch.cat([contr_result1, contr_result2], dim=0)
        contr_middle = torch.cat([contr_middle1, contr_middle2], dim=0)
        contr_input = torch.cat([contr_input1, contr_input2], dim=0)
        recons_featrue = torch.cat([recons_featrue1, recons_featrue2], dim=0)

        hash_loss1 = ((torch.sign(contr_result) - contr_result) ** 2).mean() * 0.001
        hash_loss2 = F.mse_loss(torch.sign(contr_middle), contr_middle)      
        hash_loss3 = F.mse_loss(torch.sign(recons_featrue), recons_featrue)  


        #hash_loss1 = (1 - contr_result**2).clamp(0).mean() 
        hash_loss = hash_loss1 #+ hash_loss3 
        info_loss = simsam_dis(contr_result1, contr_middle2) / 2 + simsam_dis(contr_result2, contr_middle1) / 2 
        recons_loss = F.mse_loss(torch.cat([input1, input2], dim=0), torch.cat([output1, output2], dim=0))
        hash_loss1 = F.mse_loss(contr_result1.sign(), contr_middle2.sign()) + F.mse_loss(contr_result2.sign(), contr_middle1.sign())
        loss = a1 * recons_loss + info_loss * a3 #+ hash_loss1 * 500 
        if self.num_iter % 13 == 0:
            print("recons_loss:{:.4f} info_loss:{:.4f}  hash_loss1:{:.4f} hash_loss2:{:.4f} hash_loss3:{:.4f}  ".format(recons_loss.item(), info_loss.item(), hash_loss1.item(), hash_loss2.item(), hash_loss3.item()))

        return {'loss': loss, 'recons_loss':recons_loss }
     
    def test(self, x: Tensor, **kwargs) -> Tensor:
        x = self.backbone.test(x) 
        #x = ((x) + torch.sign(x))
        
        x =  self.bth1d(x) #self.projector(contr_input)
        
        x = x.reshape(x.shape[0], x.shape[1], 1, 1) 
        x = self.predictor(x)  
        x = torch.flatten(x, start_dim=1)  
 

        return x
   