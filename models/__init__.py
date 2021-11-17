from .base import *
from .vitvae import *
from .vitvae_loss import *
from .vitvae_input import *
from .vitvae_Transformer import *
from .vitvae_cifar import *
from .vitvae_cifar_vgg_gan import *
from .vitvae_cifar_vgg import *
from .fvae import *
from .vitvae_cifar_vgg_contrastive import *
from .vitvae_cifar_vgg_IB import *
from .ae_cifar_vgg_contra import *
from .ae_cifar_vgg_contra_ir import *
from .hash_contr_ir import *
from .tools import Cifar_AE
from .withoutcontr import WithoutContr
from .withoutIR import Without_IR
from .withoutpolar import WithoutPolar
from .withoutae import WithoutAE
vae_models = {'BaseVAE' : BaseVAE,
              'VitVAE' : VitVAE,
              'VitVAE_loss' : VitVAE_loss,
              'VitVAE_Input' : VitVAE_Transformer,
              'VitVAE_Transformer' : VitVAE_Input, 
              'VitVAE_Cifar_VGG': VitVAE_Cifar_VGG,
              'VitVAE_Cifar_VGG_constrastive' : VitVAE_Cifar_VGG_constrastive,
              'VitVAE_Cifar_VGG_GAN':VitVAE_Cifar_VGG_GAN,
              'VitVAE_Cifar' : VitVAE_Cifar,
              'VitVAE_Cifar_VGG_IB' : VitVAE_Cifar_VGG_IB,
              'AE_CIFAR_VGG_Contra' : AE_CIFAR_VGG_Contra,
              'AE_CIFAR_VGG_Contra_IR' :AE_CIFAR_VGG_Contra_IR,
              'Hash_Contra_IR' : Hash_Contra_IR,
              'Cifar_AE' : Cifar_AE,
              'WithoutPolar' : WithoutPolar,
              'WithoutAE' : WithoutAE,
              'Without_IR' : Without_IR,
              'WithoutContr' : WithoutContr,
              'FlowVAE': FlowVAE}
