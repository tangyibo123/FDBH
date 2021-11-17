from torchvision import transforms
import torchvision
from models.TBH import TBH
import os
import torchvision.utils as vutils
import torch

from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader
#self.gpus = 1    
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(1)

def data_transforms():
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X)) 
    transform = transforms.Compose([
                                    #transforms.ToPILImage(mode=None),
                                    #transforms.Resize(32),
                                    transforms.ToTensor()#,
                                    #transforms.Grayscale()
                                    ]) 
    return transform

dataset = torchvision.datasets.CIFAR10(root = "/home/bbct/wangfan/dataset/", \
    train=True, download=True, transform=data_transforms())
dataloader  = DataLoader(dataset, 
                        batch_size  = 1024 * 2,
                        num_workers = 4,
                        shuffle     = True,
                        drop_last   = False)

model = TBH()
model  = model.cuda() 
  
def training_step(model, batch, batch_idx):
    real_img, labels = batch
    curr_device = real_img.device 

    results = model.forward(real_img, labels = labels) 
    actor_loss, critic_loss = model.loss_function(*results)
    
    inputimg = results[0].view(-1, 3, 32, 32)
    recons = results[1].view(-1, 3, 32, 32)
    recons = torch.cat([inputimg[0:12], recons[0:12]])

    return actor_loss, critic_loss, recons

for epoch in range(1000):
    for batch_idx, batch in enumerate(dataloader): 
        model.train()

        for i in range(len(batch)):
            batch[i] = batch[i].cuda()
        
        if batch_idx % 3 == 0:
            model.optimizer_G.zero_grad()
            actor_loss, critic_loss, recons = training_step(model, batch, batch_idx) 
            actor_loss.backward() 
            model.optimizer_G.step()
        else:
            model.optimizer_D.zero_grad()
            actor_loss, critic_loss, recons = training_step(model, batch, batch_idx) 
            critic_loss.backward()
            model.optimizer_D.zero_grad()
                 
    print("epoch{}/batch_idx:{} actor_loss:{:.4f}, critic_loss:{:.4f}".format(epoch, batch_idx, actor_loss, critic_loss)) 
    vutils.save_image(recons.cpu().data, "./logs/testTBH/{}.png".format(epoch), normalize=True, nrow=12)