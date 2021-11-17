import torch
import numpy as np  
from PIL import Image
import torch.utils.data as data
import os 

def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    #from torchvision import get_image_backend
    #if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    #else:
    return pil_loader(path)

class ImageList(object):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """  
    def __init__(self, root, train=True, download=False, labels=None, transform=None, target_transform=None,
                 loader=default_loader):
        if train: 
            txtfilename = "train_21.txt"
        elif download:
            txtfilename = "database_21.txt"
        else:
            txtfilename = "test_21.txt"
        txtpath = os.path.join(root, txtfilename) 
        image_list = open(txtpath).readlines()
        self.imgs = make_dataset(image_list, labels)  
        self.imgs = [(os.path.join(root, imgpath), target) for (imgpath, target) in self.imgs]
        
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
 
        #self.database = [(self.loader(imagepath), target) for (imagepath, target) in self.imgs]
        #print("TOTAL LEN ", len(self.database))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        imgpath, target = self.imgs[index]
        img = self.loader(imgpath)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class PairedImageList(ImageList):
    def __getitem__(self, index):   

        imgpath, target = self.imgs[index] 
        img = self.loader(imgpath)
        #img = Image.fromarray(img)
        
        if self.transform is not None: 
            pos_1 = self.transform[0](img)
            pos_2 = self.transform[1](img)

        if self.target_transform is not None:
            target = self.target_transform(target) 
        return pos_1, pos_2, target
