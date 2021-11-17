import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader#, GaussianBlur
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from datasets import SAT4, SAT6, CIFAR10Pair, ImageList, PairedImageList
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from scipy.io import savemat
import os  
from PIL import Image

def get_optimizer(model, lr, momentum, weight_decay):

    predictor_prefix = ('module.predictor', 'predictor')
    parameters = [{
        'name': 'base',
        'params': [param for name, param in model.named_parameters() if not name.startswith(predictor_prefix)],
        'lr': lr
    },{
        'name': 'predictor',
        'params': [param for name, param in model.named_parameters() if name.startswith(predictor_prefix)],
        'lr': lr
    }] 
     
    optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
 
    return optimizer

class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch, constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr+0.5*(base_lr-final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))
        
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        print(self.lr_schedule)
        print((self.lr_schedule).shape)
        print((self.lr_schedule).max())
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0
    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]
        
        self.iter += 1
        self.current_lr = lr
        return lr
    def get_lr(self):
        return self.current_lr

class VAEXperiment(pl.LightningModule):
    
    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = vae_model
        self.params = params 
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
        self.train_lantent_mat = None
        self.train_label_mat = None
        self.val_lantent_mat = None
        self.val_label_mat = None
        self.val_img = None
        self.train_img = None  
        self.saving_path = "./logs/test_lantent_z/{}_{}/".format(self.params['dataset'], self.params['lantentznames'])
        
    def forward(self, input: List[Tensor], **kwargs) -> Tensor:
        return self.model.train_step(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0): 
        results = self.forward(batch)
        self.train_loss = self.model.loss_function(*results,
                                              M_N = self.params['batch_size']/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        return self.train_loss

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        # discard the version number
        items.pop("v_num", None)
        # discard the loss
        #items.pop("loss", None)
        # Override tqdm_dict logic in `get_progress_bar_dict`
        # call .item() only once but store elements without graphs
        recLoss = self.train_loss["Reconstruction_Loss"]
        recon_loss = (
            recLoss.cpu().item()
            if recLoss is not None
            else float("NaN")
        )
        # convert the loss in the proper format
        items["rec"] = f"{recon_loss:.4f}"

        
        klloss = self.train_loss["KLD"]
        kl_loss = (
            klloss.cpu().item()
            if klloss is not None
            else float("NaN")
        )
        # convert the loss in the proper format
        items["kld"] = f"{kl_loss:.4f}"
        '''
        distriloss = self.train_loss["distri"]
        distri_loss = (
            distriloss.cpu().item()
            if klloss is not None
            else float("NaN")
        )
        # convert the loss in the proper format
        items["distri"] = f"{distri_loss:.4f}"
        '''
        
        hashing_loss = self.train_loss["hashing_loss"]
        number_loss = (
            hashing_loss.cpu().item()
            if klloss is not None
            else float("NaN")
        )
        # convert the loss in the proper format
        items["NUM"] = f"{number_loss:.4f}"
        return items

    def validation_step(self, batch, batch_idx, loader_idx, optimizer_idx = 0):
        image, labels = batch    
        lantent = self.model.test(image).cpu().numpy()

        #img      = image[:,:3,:,:].permute(0, 1, 2, 3).cpu().numpy()

        #lantent  = self.model.lantent_z.cpu().numpy()
        z_labels = labels.cpu().numpy()
        if loader_idx == 0:
            if self.val_lantent_mat is None:
                self.val_lantent_mat = lantent
                self.val_label_mat = z_labels
                #self.val_img = img
            else :
                self.val_lantent_mat = np.append(self.val_lantent_mat, lantent, axis=0)
                self.val_label_mat = np.append(self.val_label_mat, z_labels, axis=0) 
                #self.val_img = np.append(self.val_img, img, axis = 0)
        else :
            if self.train_lantent_mat is None:
                self.train_lantent_mat = lantent
                self.train_label_mat = z_labels
                #self.train_img = img
            else :
                self.train_lantent_mat = np.append(self.train_lantent_mat, lantent, axis=0)
                self.train_label_mat = np.append(self.train_label_mat, z_labels, axis=0) 
                #self.train_img = np.append(self.train_img, img, axis = 0) 
        return 0

    def __check_path(self, folderpath):
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)

    def save_checkpoint(self, saveTag = ''):
        savepath = "./logs/test_lantent_z/{}_{}_model{}.ckpt".format(self.params['dataset'], self.params['lantentznames'], saveTag)
        torch.save(self.model, savepath)

    def load_checkpoint(self, saveTag= ''):
        ckpt_path = self.params['ckpt_path']

        savepath = "./logs/test_lantent_z/{}_{}_model{}.ckpt".format(self.params['dataset'], self.params['lantentznames'], saveTag)
        #savepath =  '/home/bbct/wangfan/code/ViT-VAE/logs/test_lantent_z_findbesthyper/SAT-4_VitVAE[1, 1, 3]_model.ckpt'
        if os.path.exists(ckpt_path):
            self.model = (torch.load(ckpt_path))
            self.model.eval()
            print("load from para paths") 
        elif os.path.exists(savepath):
            self.model = torch.load(savepath)
            self.model.eval()
            print("load model exists")
        else:
            print(savepath, "File not exists")

    def validation_epoch_end(self):
        self.sample_images()
        # file 文件  
        folder_path = self.saving_path + "epoch_{}/".format(self.current_epoch)
        self.__check_path(folder_path)

        if self.val_label_mat is not None:
            # 转成 onehot编码
            if self.val_label_mat.max() > 2: 
                labels = self.val_label_mat.max() + 1
                self.val_label_mat = np.reshape(self.val_label_mat, -1)
                self.val_label_mat = np.eye(labels ,labels)[self.val_label_mat]
 
            #kldloss = self.model.every_channel_kld_loss
            #lantent_channel = np.flip(np.argsort(kldloss))[0:32]  
  
            savemat(folder_path + "sat_4_test.mat"       , {"sat_4_test"       : self.val_lantent_mat}) 
            savemat(folder_path + "sat_4_test_lable.mat" , {"sat_4_test_lable" : self.val_label_mat}) 
            #savemat(folder_path + "sat_4_test_img.mat",    {"sat_4_test_img"   : self.val_img})
            self.val_lantent_mat = None
            self.val_label_mat   = None
            self.val_img         = None

        if self.train_label_mat is not None:
            # 转成 onehot编码
            if self.train_label_mat.max() > 2 :
                labels = self.train_label_mat.max() + 1
                self.train_label_mat = np.reshape(self.train_label_mat, -1)
                self.train_label_mat = np.eye(labels ,labels)[self.train_label_mat]
 
            #kldloss = self.model.every_channel_kld_loss
            #lantent_channel = np.flip(np.argsort(kldloss))[0:32]   

            savemat(folder_path + "sat_4_train.mat"      , {"sat_4_train":self.train_lantent_mat[0:100000]})
            savemat(folder_path + "sat_4_train_lable.mat", {"sat_4_train_lable":self.train_label_mat[0:100000]}) 
            #savemat(folder_path + "sat_4_train_img.mat",    {"sat_4_train_img"   : self.train_img[0:100000]})

            self.train_lantent_mat = None
            self.train_label_mat = None
            self.train_img = None
        #from rankingmap import ALLINONE
        # mAP = ALLINONE(folder_path)
        #print("This epoch mAP is {:.4f}".format(mAP))
        return None

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader)) 
        test_input = test_input.to(self.device)
        test_label = test_label.to(self.device)
        
        #channel_result = self.model.multi_channel_test(test_input)
        folder_path = self.saving_path + "epoch_{}/".format(self.current_epoch)
        self.__check_path(folder_path)  

        #np.save(folder_path + "channel_result_epoch" , channel_result)
        '''
        recons = self.model.generate(test_input, labels = test_label)
        
        recons = torch.cat([recons[:12*3, 0:3, :, :],test_input[:12*3, 0:3, :, :]])

        path = self.saving_path + "recons_epoch_{}.png".format(self.current_epoch)
        vutils.save_image(recons.data,
                        path,
                        normalize=True,
                        nrow=12) 
        '''
    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims, None
    def configure_optimizers2(self):
        optims = []
        scheds = []
        optimizer =  get_optimizer(self.model,
                               lr=self.params['LR'],
                               momentum = self.params["momentum"],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            print(self.params['LR_2'])
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(), lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0], gamma = self.params['scheduler_gamma'])
                
                scheduler = LR_Scheduler(
                    optimizer, 
                    warmup_epochs = 25, 
                    warmup_lr = 0 * self.params['batch_size'] /256, 
                    num_epochs = self.params['num_epochs'] , 
                    base_lr = self.params['LR'] * self.params['batch_size']/256, 
                    final_lr = 0 * self.params['batch_size']/256, 
                    iter_per_epoch = len(self.train_loader),
                    constant_predictor_lr = True # see the end of section 4.2 predictor
                )
                
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    def train_dataloader(self):
        self.train_lantent_mat = None
        self.train_label_mat = None
        self.train_img = None

        ourdatasets = {'SAT-4':SAT4, 'SAT-6':SAT6}
        unpaired_datasets = {'CIFAR': torchvision.datasets.CIFAR10, "mscoco":ImageList, "nuswild":ImageList}
        paired_datasets = {'CIFAR10Pair': CIFAR10Pair, "mscoco_paired":PairedImageList, "nuswild_paired":PairedImageList}
        if self.params['dataset'] in ourdatasets.keys() : 
            transform = self.SAT_transforms()   
            dataset =  ourdatasets[self.params['dataset']](root = self.params['data_path'], 
            image_set="train", 
            download=False, 
            transform=transform, 
            target_transform = False, 
            **self.params) 
        elif self.params['dataset'] in unpaired_datasets.keys() : 
            transform = self.Cifar_transforms()   
            dataset = unpaired_datasets[self.params['dataset']](root = self.params['data_path'], train=True, download=True, transform=transform)
        elif self.params['dataset'] in paired_datasets.keys() :
            #transform = [self.Cifar10Pair_train_transforms(), self.Cifar10Pair_test_transforms()]
            transform = [self.Cifar10Pair_train_transforms(), self.Cifar10Pair_train_transforms()]
            dataset = paired_datasets[self.params['dataset']](root=self.params['data_path'], train=True, transform=transform, download=True) 
        else :
            print("ERROR") 
        self.num_train_imgs = len(dataset)
        self.train_loader =  DataLoader(dataset,
                          batch_size= self.params['batch_size'],
                          num_workers = 8,
                          shuffle = True,
                          drop_last=True,
                          pin_memory=True)
        return self.train_loader
    def val_dataloader(self):
        self.val_lantent_mat = None
        self.val_label_mat = None 
        self.val_img = None

        ourdatasets = {'SAT-4':SAT4, 'SAT-6':SAT6}
        #cifardatasets = {'CIFAR': torchvision.datasets.CIFAR10}
        unpaired_datasets = {'CIFAR': torchvision.datasets.CIFAR10, "mscoco":ImageList, "nuswild":ImageList}
        paired_datasets = {'CIFAR10Pair': torchvision.datasets.CIFAR10}
        Imagelist_datasets = {"mscoco_paired":ImageList, "nuswild_paired":ImageList}
        
        if self.params['dataset'] in ourdatasets.keys() :
            self.sample_dataloader2 = self.train_dataloader()
            transform = self.SAT_transforms()
            dataset = ourdatasets[self.params['dataset']](root = self.params['data_path'] , image_set="test", download=False, transform=transform, target_transform = False, **self.params)
        elif self.params['dataset'] in unpaired_datasets.keys() :
            self.sample_dataloader2 = self.train_dataloader()
            transform = self.Cifar_transforms()
            dataset = unpaired_datasets[self.params['dataset']](root = self.params['data_path'], train=False, download=True, transform=transform)
        elif self.params['dataset'] in paired_datasets.keys() :
            transform = self.Cifar10Pair_test_transforms()   
            dataset =  paired_datasets[self.params['dataset']](root = self.params['data_path'], train=True, download=True, transform=transform)
            self.sample_dataloader2 = DataLoader(dataset, batch_size= self.params['batch_size'], num_workers = 4, shuffle = True, drop_last=False, pin_memory=True)

            transform = self.Cifar10Pair_test_transforms()
            dataset = paired_datasets[self.params['dataset']](root = self.params['data_path'], train=False, download=True, transform=transform) 
        elif self.params['dataset'] in Imagelist_datasets.keys() :
            transform = self.Cifar10Pair_test_transforms()   
            dataset =  ImageList(root = self.params['data_path'], train=False, download=True, transform=transform)
            self.sample_dataloader2 = DataLoader(dataset, batch_size= self.params['batch_size'], num_workers = 4, shuffle = False, drop_last=False, pin_memory=True)

            transform = self.Cifar10Pair_test_transforms()
            dataset = ImageList(root = self.params['data_path'], train=False, download=False, transform=transform)

        self.sample_dataloader = DataLoader(
            dataset, 
            batch_size= self.params['batch_size'],
            num_workers = 4, 
            shuffle = False, 
            drop_last=False,
            pin_memory=True)  

        self.num_val_imgs = len(self.sample_dataloader) 
        return [self.sample_dataloader, self.sample_dataloader2]
 
    def SAT_transforms(self):
        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))
        transform = transforms.Compose([transforms.ToPILImage(mode=None),
                                        transforms.Resize(self.params['img_size']),
                                        transforms.ToTensor()
                                        ])
 
        return transform

    def Cifar_transforms(self): 
        transform = transforms.Compose([transforms.Resize(self.params['img_size']),
                                        transforms.ToTensor()
                                        ])
 
        return transform
    
    def Cifar10Pair_train_transforms(self):
        image_size = self.params['img_size']
        p_blur = 0.5 if image_size > 32 else 0 # exclude cifar
        mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            #transforms.RandomApply([GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
            transforms.ToTensor(),
            transforms.Normalize(*mean_std)])
        '''
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.params['img_size']),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()])
        '''
        return train_transform

    def Cifar10Pair_test_transforms(self): 
        mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]
        image_size = self.params['img_size'] 
        resize = int(image_size*(8/7))
        test_transform = transforms.Compose([
            transforms.Resize([resize, resize], interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(*mean_std)])

        return test_transform