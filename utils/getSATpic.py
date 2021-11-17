import torchvision.utils as vutils
import numpy as np
from datasets import SAT4, SAT6
from torchvision import transforms
import torch

classes_sat4 = ["barren land", "trees", "grassland", "none"]
classes_sat6 = ["building", "barren land" ,"trees" ,"grassland" ,"road" , "water"]  
ourdatasets = {'SAT-4':SAT4, 'SAT-6':SAT6} 
dataset = SAT4(root = "./SAT", image_set="test", download=False, transform=None, target_transform = False) 

image, target = dataset[0:100]
image = torch.Tensor(image)

def getimagebylabel(imagebank, chars, label, labelname):
    image = imagebank.permute(3, 0, 1, 2).contiguous() 
    image = image[target == label][0:3]
    image = image.permute(0,3,1,2).contiguous() 
    savepath = "./logs/show_images/{}_{}.png".format(chars, label) 
    vutils.save_image(image.data, savepath, normalize=True, nrow=3)

getimagebylabel(image, "SAT4", 0, classes_sat4)
getimagebylabel(image, "SAT4", 1, classes_sat4)
getimagebylabel(image, "SAT4", 2, classes_sat4)
getimagebylabel(image, "SAT4", 3, classes_sat4) 

dataset = SAT6(root = "./SAT", image_set="test", download=False, transform=None, target_transform = False) 

image, target = dataset[0:1000]
image = torch.Tensor(image) 

getimagebylabel(image, "SAT6", 0, classes_sat6)
getimagebylabel(image, "SAT6", 1, classes_sat6)
getimagebylabel(image, "SAT6", 2, classes_sat6)
getimagebylabel(image, "SAT6", 3, classes_sat6)
getimagebylabel(image, "SAT6", 4, classes_sat6)
getimagebylabel(image, "SAT6", 5, classes_sat6)