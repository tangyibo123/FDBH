
import torchvision
import torchvision.transforms as transforms
import torch
root = "/home/bbct/wangfan/dataset/"


trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

print("OK")

class CIFAR():
    
    def __init__(self, root):
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True)