import numpy as np 
import torch
from torch import nn
from torch.nn import functional as F
from models import BaseVAE
from torch.utils import model_zoo 
from .types_ import *
import math
import torchvision.models as torchmodels
from torch.autograd import Function 
def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class LinearResidualLayer(nn.Module):
    def __init__(self,
                 in_channels: int):
        super(LinearResidualLayer, self).__init__() 
        self.resblock = nn.Sequential(nn.Linear(in_channels, in_channels, bias=True), nn.ReLU(True))

    def forward(self, input: Tensor) -> Tensor: 
        return input + self.resblock(input)

class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def calh(self, q, k, v):
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2) 
        return h
    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : [q, k, v]
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
             
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        h = self.calh(q, k, v)
        if mask is not None:
            q, _, _ = mask[0], mask[1], mask[2]
            h = (h + self.calh(q, k, v)) / 2
        return h, q, k, v

class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))

class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, num_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        atten, q, k, v = self.attn(self.norm1(x), mask)
        h = self.drop(self.proj(atten))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x, q, k, v 

class DoubleChannelTransformer(nn.Module):
    """DoubleChannelTransformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.rgbblocks = nn.ModuleList([
            Block(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.nirblocks = nn.ModuleList([
            Block(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
    
    def forward(self, rgb, nir,mask=None):
        '''
        b, patch, dim = x.shape
        half_patch = patch / 2
        rgb = x[:, :half_patch, :]
        nir = x[:, half_patch:, :]
        '''
        for blockid in range(len(self.rgbblocks)):
            nir, q, k, v = self.nirblocks[blockid](nir, mask)
            rgb, _, _, _ = self.rgbblocks[blockid](rgb, [q,k,v])

        return rgb, nir

class Transformer(nn.Module):
    """DoubleChannelTransformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
    
    def forward(self, x, mask=None):
        '''
        b, patch, dim = x.shape
        half_patch = patch / 2
        rgb = x[:, :half_patch, :]
        nir = x[:, half_patch:, :]
        '''
        for block in self.blocks:
            x, q, k, v = block(x, mask) 
        return x

class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""
    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))
    
    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding
    
class Half_VGG(nn.Module): 
    def __init__(
        self
    ) -> None:
        super(Half_VGG, self).__init__()
        targetmodel = torchmodels.vgg16(pretrained=True) 
        self.features = targetmodel.features 
        self.avgpool = targetmodel.avgpool
        self.classifier = targetmodel.classifier

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features[:23](x) 
        return x

    def distance(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features[23:](x) 
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier[0:5](x)
        return x

class Half_Resnet(nn.Module): 
    def __init__(
        self
    ) -> None:
        super(Half_Resnet, self).__init__()
  
        targetmodel = torchmodels.resnet101(pretrained=True)
        self.conv1    = targetmodel.conv1 
        self.bn1      = targetmodel.bn1   
        self.relu     = targetmodel.relu  
        self.maxpool  = targetmodel.maxpool      
        self.layer1   = targetmodel.layer1  
        self.layer2   = targetmodel.layer2  
        self.layer3   = targetmodel.layer3  
        self.layer4   = targetmodel.layer4     
 
        self.avgpool   = targetmodel.avgpool   
        print(targetmodel)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) 
        x = self.avgpool(x)  
        return x
    
    def distance(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x = self.features[23:](x) 
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier[0:5](x)
        '''
        return x



class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None

class IRConv2d(nn.Conv2d): 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(IRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda() 
    def forward(self, input):
        w = self.weight
        a = input
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = torch.pow(torch.tensor([2]*bw.size(0)).cuda().float(), (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(bw.size(0), 1, 1, 1).detach()
        bw = BinaryQuantize().apply(bw, self.k, self.t)
        ba = BinaryQuantize().apply(a, self.k, self.t)
        bw = bw * sw
        output = F.conv2d(ba, bw, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        ) 
    def forward(self, x): 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  
        return x 

class Cifar_AE(BaseVAE):
    num_iter = 0
    def __init__(
        self, 
        latent_dim ,
        out_dims = 2048 * 4,
        in_dims = 2048 * 4, 
        pretrained = False 
    ): 
        super(Cifar_AE, self).__init__() 

        self.in_dims = in_dims
        self.out_dims = out_dims 
        self.latent_dim = latent_dim 
        targetmodel = torchmodels.vgg16(pretrained=True) 
        self.pre_trained = targetmodel.features[:23]  

        for p in self.pre_trained.parameters():
            p.requires_grad = False
 
        hidden_dims = [4096, 512, 128]

        modules = []
        modules.append(nn.Sequential(
            nn.Linear(self.in_dims , 4096),
            nn.ReLU(),
            nn.Linear(4096 , 512),
            nn.ReLU(),
            nn.Linear(512 , 128),
            nn.ReLU(),
            nn.Linear(128, self.latent_dim) ,
            nn.BatchNorm1d(self.latent_dim)
        ))
        
        self.encoder = nn.Sequential(*modules) 
         
        # Build Decoder 
        modules = []
        modules.append(nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128 , 512),
            nn.ReLU(),
            nn.Linear(512 , 4096),
            nn.ReLU(),
            nn.Linear(4096, self.out_dims) 
        ))
 
        self.decoder = nn.Sequential(*modules) 

        if pretrained: 
            for p in self.parameters():
                p.requires_grad = False
         
    def train_step(self, batch: List[Tensor], **kwargs) -> Tensor:
        pos1, pos2, target = batch
        x1, rec_x1, hidden1 = self.forward(pos1)
        x2, rec_x2, hidden2 = self.forward(pos2)
        return [x1, rec_x1, hidden1, x2, rec_x2, hidden2]

    def forward(self, x: Tensor, **kwargs) -> Tensor:    
        x = self.pre_trained(x)
        x_in = torch.flatten(x, start_dim=1)
        h = self.encoder(x_in)
        x_out = self.decoder(h)  
        return x_in, x_out, h

    def loss_function(self,
                      *args,
                      **kwargs) -> dict: 
        self.num_iter += 1
        x1, rec_x1, hidden1, x2, rec_x2, hidden2 = args 
        x_in = torch.cat([x1, x2], dim=0)
        x_out = torch.cat([rec_x1, rec_x2], dim=0)
        hidden = torch.cat([hidden1, hidden2], dim=0)
        a1, a2 = kwargs["hypermeter"]

        recons_loss = F.mse_loss(x_in, x_out)
        hash_loss = F.mse_loss(torch.sign(hidden), hidden) 

        loss = a1 * recons_loss + hash_loss * a2 
        if self.num_iter % 13 == 0:
            print("loss:{:.4f}  recons_loss:{:.4f} hash_loss:{:.4f}  ".format(loss.item(), recons_loss.item(), hash_loss.item()))

        return {'loss': loss, 'recons_loss':recons_loss }

    def test(self, x):
        x = self.pre_trained(x)
        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x) 
        return x


class COCO_AE(BaseVAE):
    num_iter = 0
    def __init__(
        self, 
        latent_dim   ,
        out_dims = 4096,
        in_dims = 4096, 
        pretrained = False 
    ): 
        super(COCO_AE, self).__init__() 

        self.in_dims = in_dims
        self.out_dims = out_dims 
        self.latent_dim = latent_dim 
        
        self.pre_trained = torchmodels.vgg16(pretrained=True)  
        self.pre_trained.classifier = self.pre_trained.classifier[:-3]
   

        for p in self.pre_trained.parameters():
            p.requires_grad = False
 
        hidden_dims = [4096, 512, 128]

        modules = []
        modules.append(nn.Sequential(
            nn.Linear(self.in_dims , 4096),
            nn.ReLU(),
            nn.Linear(4096 , 512),
            nn.ReLU(),
            nn.Linear(512 , 128),
            nn.ReLU(),
            nn.Linear(128, self.latent_dim) ,
            nn.BatchNorm1d(self.latent_dim)
        ))
        
        self.encoder = nn.Sequential(*modules) 
         
        # Build Decoder 
        modules = []
        modules.append(nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128 , 512),
            nn.ReLU(),
            nn.Linear(512 , 4096),
            nn.ReLU(),
            nn.Linear(4096, self.out_dims) 
        ))
 
        self.decoder = nn.Sequential(*modules) 

        if pretrained: 
            for p in self.parameters():
                p.requires_grad = False
         
    def train_step(self, batch: List[Tensor], **kwargs) -> Tensor:
        pos1, pos2, target = batch
        x1, rec_x1, hidden1 = self.forward(pos1)
        x2, rec_x2, hidden2 = self.forward(pos2)
        return [x1, rec_x1, hidden1, x2, rec_x2, hidden2]

    def forward(self, x: Tensor, **kwargs) -> Tensor:    
        x = self.pre_trained(x)
        x_in = torch.flatten(x, start_dim=1)
        h = self.encoder(x_in)
        x_out = self.decoder(h)  
        return x_in, x_out, h

    def loss_function(self,
                      *args,
                      **kwargs) -> dict: 
        self.num_iter += 1
        x1, rec_x1, hidden1, x2, rec_x2, hidden2 = args 
        x_in = torch.cat([x1, x2], dim=0)
        x_out = torch.cat([rec_x1, rec_x2], dim=0)
        hidden = torch.cat([hidden1, hidden2], dim=0)
        a1, a2 = kwargs["hypermeter"]

        recons_loss = F.mse_loss(x_in, x_out)
        hash_loss = F.mse_loss(torch.sign(hidden), hidden) 

        loss = a1 * recons_loss + hash_loss * a2 
        if self.num_iter % 13 == 0:
            print("loss:{:.4f}  recons_loss:{:.4f} hash_loss:{:.4f}  ".format(loss.item(), recons_loss.item(), hash_loss.item()))

        return {'loss': loss, 'recons_loss':recons_loss }

    def test(self, x):
        x = self.pre_trained(x)
        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x) 
        return x