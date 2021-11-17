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
class VitVAE_Cifar_VGG_constrastive(BaseVAE):
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

        # pretrained network out : b x 512 x 4 x 4
        self.pretrained = (models.vgg16(pretrained=True).features)[0:23]
        for p in self.parameters():
            p.requires_grad = False

        in_dims = 512 * 4 * 4 
        hidden_dims = [512 * 4 , 512, 128, 32, 128, 128]

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
        self.fc_var = nn.Linear(hidden_dims[-1] , self.latent_dim) 


        self.discriminator = nn.Sequential(   
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.latent_dim * 2 , self.latent_dim),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.latent_dim, 2)
        ) 

        # projection head
        self.g = nn.Sequential(nn.Linear(self.latent_dim, 512, bias=False), 
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), 
                               nn.Linear(512, self.latent_dim, bias=True)
                               )
        # Build Encoder
        
        in_dims = self.latent_dim  
        hidden_dims = [512 * 4 * 4 , 512 * 4 , 512, 128, 32, 128, 128]
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

    def encode(self, x: Tensor) -> List[Tensor]:  
        x = self.pre_encoder(x)  
        mu, log_var = self.fc_mu(x), self.fc_var(x) 

        return [mu, log_var]
         

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z: Tensor):
        return self.decoder(z)

    def train_step(self, batch: List[Tensor], **kwargs) -> Tensor:
        pos1, pos2, target = batch
        output1, input1, mu1, log_var1, z1, out1 = self.forward(pos1)
        output2, input2, mu2, log_var2, z2, out2 = self.forward(pos2)
        return [output1, input1, mu1, log_var1, z1, out1, output2, input2, mu2, log_var2, z2, out2]

    def forward(self, input: Tensor, **kwargs) -> Tensor:   
        input = self.pretrained(input)
        #print(input.shape)
        input = torch.flatten(input, start_dim=1)  
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)

        feature = self.g(z.detach())
        feature = F.normalize(feature, dim=-1)

        self.lantent_z = feature.detach()

        output = self.decoder(z)
        return  [output, input, mu, log_var, z, feature]
    
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:

        self.num_iter += 1
        output1, input1, mu1, log_var1, z1, out1, output2, input2, mu2, log_var2, z2, out2 = args 
        
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        optimizer_idx = kwargs['optimizer_idx']
        a1, a2, temperature, a4 = self.hypermeter

        input = torch.cat([input1, input2], dim=0)
        recons = torch.cat([output1, output2], dim=0)
        mu = torch.cat([mu1, mu2], dim=0)
        log_var = torch.cat([log_var1, log_var2], dim=0)
        # [2*B, D]
        out = torch.cat([out1, out2], dim=0)
        
        # update VAE
        if optimizer_idx == 0:  
            batch_size = out1.shape[0] 
            # [2*B, 2*B]
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
            # [2*B, 2*B-1]
            sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
            
            # compute loss
            pos_sim = torch.exp(torch.sum(out1 * out2, dim=-1) / temperature)
    
            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            hashing_loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

            recons_loss = F.mse_loss(recons, input)
            kld_loss = kld_weight * torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
            
            true_labels = torch.ones(input.size(0), dtype= torch.long, requires_grad=False).to(mu.device)
            d_fake = self.discriminator(out)  
            dis_loss = 0.5 * (F.cross_entropy(d_fake, true_labels))

            loss =  512 * 4 * 4 * recons_loss + kld_loss + hashing_loss + a4 * dis_loss

            if self.num_iter % 13 == 0:
                print("KLD:{:.4f} RecLoss:{:.4f} hashing_loss{:.4f} dis_loss:{:.4f} ".format(kld_loss.item(), recons_loss.item(), hashing_loss.item(), dis_loss.item()))

            return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss, 'hashing_loss': hashing_loss}
 
        # Update the Discriminator
        elif optimizer_idx == 1:
            device = input.device
            true_labels = torch.ones(input.size(0), dtype= torch.long, requires_grad=False).to(device)
            false_labels = torch.zeros(input.size(0), dtype= torch.long, requires_grad=False).to(device)

            # target distribution
            halfshape = [out.shape[0], int(out.shape[1] / 2)]
            left = torch.randn(halfshape, device = input.device, requires_grad=False) * 0.5 - 0.8660254037844386 
            right = torch.randn(halfshape, device = input.device, requires_grad=False) * 0.5 + 0.8660254037844386 
            target_distri = torch.cat([left, right], dim= 1)

            #recons = recons.detach() # Detach so that VAE is not trained again 
            #d_fake = self.discriminator(torch.flatten(recons, start_dim=1))
            #d_real = self.discriminator(torch.flatten(input, start_dim=1))
            out_detach = out.detach() # Detach so that VAE is not trained again 
            d_fake = self.discriminator(out_detach)  
            d_real = self.discriminator(target_distri)  

            D_tc_loss = 0.5 * (F.cross_entropy(d_fake, false_labels) + F.cross_entropy(d_real, true_labels))

            if self.num_iter % 13 == 1:
                print("discriminator loss:{:.4f} ".format(D_tc_loss.item()))
            return {'loss': D_tc_loss, 'loss': D_tc_loss, 'loss': D_tc_loss, 'loss': D_tc_loss}

        
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

    def multi_channel_test(self, input: Tensor):
        
        input = self.pretrained(input)
        input = torch.flatten(input, start_dim=1)  
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

