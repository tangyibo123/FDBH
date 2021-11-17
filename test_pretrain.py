import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import os
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader 
import torch
from models.tools import *
gpus = 1    
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpus)
 
ourmodel = Half_VGG() 
ourmodel = ourmodel.cuda()
  
def transfer_batch_to_gpu(batch, device):
    for i in range(len(batch)):
        batch[i] = batch[i].to(device)
    return batch
    
def Cifar_transforms():
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))

    transform = transforms.Compose([transforms.Resize(32),
                                    transforms.ToTensor()
                                    ])

    return transform



transform = Cifar_transforms()  
dataset = torchvision.datasets.CIFAR10(root =  "/home/bbct/wangfan/dataset/" , train=True, download=True, transform=transform)

num_train_imgs = len(dataset)
dataloader = DataLoader(dataset, batch_size= 128, shuffle = True, drop_last=False)

loaderlen = len(dataloader)

import time

timestart = time.time()

for batch_idx, batch in enumerate(dataloader): 
    batch = transfer_batch_to_gpu(batch, torch.device("cuda"))
    real_img, labels = batch 
    with torch.no_grad():
        result = ourmodel.forward(real_img) 
        
        dis = ourmodel.distance(result) 

    print(dis.shape)
    #print(result[0])
    print(result.shape)

    print(time.time() - timestart)
    break

