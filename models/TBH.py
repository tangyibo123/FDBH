import numpy as np 
import torch
from torch import nn
from torch.nn import functional as F
import math
from models import BaseVAE
from torch.utils import model_zoo 
from .types_ import *
from .tools import *
 
OVERFLOW_MARGIN = 1e-8

    
class Binary_activation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        eps = torch.ones_like(logits, device = logits.device) 
        prob = 1.0 / (1 + torch.exp(-logits))
        code = (torch.sign(prob - eps) + 1.0) / 2.0
        ctx.save_for_backward(prob)
        return code

    @staticmethod
    def backward(ctx, output_grad):
        prob = ctx.saved_tensors[0]
        input_grad = prob * (1 - prob) * output_grad
        return input_grad 
  

def build_adjacency_hamming(tensor_in): 
    code_length = torch.from_numpy(np.array(tensor_in.shape[1], dtype=np.float32))
    # code_length = torch.FloatTensor(tensor_in.shape[1])
    m1 = tensor_in - 1
    c1 = torch.matmul(tensor_in, m1.T)
    c2 = torch.matmul(m1, tensor_in.T)
    normalized_dist = torch.abs(c1 + c2) / code_length
    return torch.pow(1 - normalized_dist, 1.4)

import math 

class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """ 
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
            

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# noinspection PyAbstractClass
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim  
        self.fc = nn.Linear(in_dim , out_dim)

    # noinspection PyMethodOverriding
    def forward(self, values, adjacency, **kwargs): 
        fc_sc = self.fc(values)
        conv_sc = self.graph_laplacian(adjacency) @ fc_sc
        return conv_sc
  
  
    def graph_laplacian(self, adjacency): 
        graph_size = adjacency.shape[0]
        d = adjacency @ torch.ones([graph_size, 1], device = adjacency.device )
        d_inv_sqrt = torch.pow(d + OVERFLOW_MARGIN, -0.5)
        d_inv_sqrt = torch.eye(graph_size, device = adjacency.device) * d_inv_sqrt
        laplacian = d_inv_sqrt @ adjacency @ d_inv_sqrt
        return laplacian
 

class TwinBottleneck(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__(**kwargs) 
        self.gcn = GraphConvolution(in_dim, out_dim)

    # noinspection PyMethodOverriding
    def forward(self, hashing, z):
        adj = build_adjacency_hamming(hashing)
        return torch.nn.Sigmoid()(self.gcn(z, adj))


class Encoder(nn.Module): 
    def __init__(self, in_dim, middle_dim, hash_out, z_out): 
        super(Encoder, self).__init__()

        
        modules = [] 
        hidden_dims = [32, 64, 128, 256, 512]
        in_channels = 3
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size = 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.pre_encoder = nn.Sequential(*modules)  

        self.fc_pre = nn.Linear(hidden_dims[-1] , middle_dim)
        
        self.fc_hashing = nn.Linear(middle_dim , hash_out) 
        self.fc_z = nn.Linear(middle_dim , z_out)

    def forward(self, inputs, **kwargs): 
        fc_pre = self.pre_encoder(inputs)

        fc_pre = torch.flatten(fc_pre, start_dim=1)
        fc_pre = self.fc_pre(fc_pre)
        hashing = self.fc_hashing(fc_pre) 
        hashing = Binary_activation.apply(hashing)

        z_feat =  (self.fc_z(fc_pre))
        return hashing, z_feat

# noinspection PyAbstractClass
class Decoder(nn.Module):
    def __init__(self, prez_dim, middle_dim, out_dim): 
        super(Decoder, self).__init__() 

        modules = []
        hidden_dims = [32, 64, 128, 256, 512]

        self.decoder_input = nn.Linear(prez_dim, hidden_dims[-1] * 4)

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
                            nn.Conv2d(hidden_dims[-1], out_channels = 3,
                                      kernel_size= 3, padding= 1),
                            nn.ReLU())
          
    def forward(self, inputs, **kwargs): 
        result = self.decoder_input(inputs)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

class TBH(nn.Module): 
    def __init__(self, **kwargs):
        super().__init__()   
        
        input_dim  = 3 * 32 * 32
        middle_dim = 2048
        hash_out   = 128
        z_out      = 128

        self.encoder = Encoder(input_dim, middle_dim, hash_out, z_out)
        self.gcn     = TwinBottleneck(z_out, z_out)
        self.decoder = Decoder(z_out, middle_dim, input_dim)
        
        self.dis_1 = nn.Sequential(torch.nn.Linear(hash_out, int(hash_out/2)), torch.nn.Sigmoid(), \
            torch.nn.Linear(int(hash_out/2), 1), torch.nn.Sigmoid()) 
        self.dis_2 = nn.Sequential(torch.nn.Linear(z_out, int(z_out/2)), torch.nn.Sigmoid(), \
            torch.nn.Linear(int(z_out/2), 1), torch.nn.Sigmoid())  

                
        self.optimizer_D = torch.optim.Adam([
                {'params': self.dis_1.parameters()},
                {'params': self.dis_2.parameters()}], 
            lr=1e-5)
        self.optimizer_G = torch.optim.Adam([
                {'params': self.encoder.parameters()},
                {'params': self.gcn.parameters()},
                {'params': self.decoder.parameters()}],
                 lr=1e-5)

    def forward(self, feat_in: Tensor, labels:Tensor, **kwargs) -> Tensor:    

        hashing, lantentz = self.encoder(feat_in)
 
        lantentz = self.gcn(hashing, lantentz)
        
        feat_out = self.decoder(lantentz) 
          
        sample_hashing = (torch.sign(torch.empty_like(hashing).uniform_(0, 1)   - 0.5) + 1) / 2
        sample_z       = torch.empty_like(lantentz).uniform_(0, 1)
  
        dis_hashing = self.dis_1(hashing)
        dis_z = self.dis_2(lantentz)
        dis_hashing_sample = self.dis_1(sample_hashing)
        dis_z_sample = self.dis_2(sample_z)

        return [feat_in, feat_out, dis_hashing, dis_z, dis_hashing_sample, dis_z_sample]
    
    def configure_optimizers(self, LR, weight_decay):
        optims = []  
        optimizer = optim.Adam(self.parameters(),
                                lr=LR,
                                weight_decay=weight_decay)
        optims.append(optimizer)
        return optims

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:

        feat_input, feat_out, dis_hashing, dis_z, dis_hashing_sample, dis_z_sample = args
 
        adversarial_loss = torch.nn.BCELoss()
        hashingvalid = torch.autograd.Variable(torch.Tensor(dis_hashing.size(0), 1).fill_(1.0), requires_grad=False ).cuda()
        hashingfake = torch.autograd.Variable(torch.Tensor(dis_hashing.size(0), 1).fill_(0.0), requires_grad=False ).cuda()
        zvalid = torch.autograd.Variable(torch.Tensor(dis_z.size(0), 1).fill_(1.0), requires_grad=False ).cuda()
        zfake = torch.autograd.Variable(torch.Tensor(dis_z.size(0), 1).fill_(0.0), requires_grad=False ).cuda()

        actor_loss = F.mse_loss(feat_input, feat_out)  #+ 0.1 * adversarial_loss(dis_z, zvalid)#  + 0.01 *  adversarial_loss(dis_hashing, hashingvalid)  
 
        critic_loss = (adversarial_loss(dis_z_sample, zvalid) + adversarial_loss(dis_z, zfake)) / 2 
        return [actor_loss, critic_loss]
 