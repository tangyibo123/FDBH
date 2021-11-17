import torch
from experiment import VAEXperiment
import pytorch_lightning as pl
import os

class MyTrainer:
    lr= 0.
    def __init__(self, params: dict):
        self.epochs = params["epochs"]
        self.gpus = params["gpus"]
        #self.gpus = 0    
        #self.on_gpu = True if (gpus and torch.cuda.is_available()) else False

    def _process(self):  
        '''
        self.model.load_checkpoint(savepath='')
        with torch.no_grad():
            av_loss, accurate = self.validataion()
        '''
        self.train()
        if (self.cur_epoch + 1) % 20 == 0 :
            with torch.no_grad():
                print("+++++Start Val++++++")
                av_loss, accurate = self.validataion()

    def transfer_batch_to_gpu(self, batch, gpus):
        for i in range(len(batch)):
            batch[i] = batch[i].cuda(non_blocking=True)
        return batch

    def train(self):
        optimlen = len(self.optims)
        self.train_prepare()
        loaderlen = len(self.train_loader)
        for batch_idx, batch in enumerate(self.train_loader): 
            self.model.train() 
            self.model.zero_grad()
             
            #for optim_idx, optim in enumerate(self.optims): 
            # 每4 轮 使用一个optim——discriminater
            #optim_idx = batch_idx % optimlen
            optim_idx = batch_idx  % optimlen 
            batch = self.transfer_batch_to_gpu(batch, self.gpus) 
            _train_loss = self.model.training_step(batch, batch_idx, optimizer_idx = optim_idx)
            _train_loss['loss'].backward() 
            self.optims[optim_idx].step()      
            '''
            if batch_idx % 10 == 0:
                print("cur epoch {}/{} batch id {}/{} KLDLoss:{:.4f} RecLoss:{:.4f} hashing_loss:{:.4f}".format(self.cur_epoch, \
                    self.epochs, batch_idx, loaderlen, _train_loss['KLD'].item(), _train_loss['Reconstruction_Loss'].item(), _train_loss['hashing_loss'].item()))
            '''
        if self.scheds is not None:
            self.scheds[0].step() 
        self.lr = self.optims[0].param_groups[0]['lr']
        print('=========== epoch {}/{} end lr = {} ==========='.format(self.cur_epoch, self.epochs, self.lr))
        return None
    
    def train_prepare(self):
        self.train_loader = self.model.train_dataloader()
        
    def validataion(self):
        self.val_prepare()
        for (loader_idx, loader) in enumerate(self.val_loader):
            for batch_idx, batch in enumerate(loader): 
                batch = self.transfer_batch_to_gpu(batch, self.gpus) 
                _val_loss = self.model.validation_step(batch, batch_idx, loader_idx)
                '''
                if batch_idx % 100 == 1 :  
                    print("Test batch id {} KLDLoss:{:.4f} RecLoss:{:.4f} hashing_loss:{:.4f}".format(
                        batch_idx, _val_loss['KLD'].item(), _val_loss['Reconstruction_Loss'].item(), _val_loss['hashing_loss'].item()))
                '''
        _dict = self.model.validation_epoch_end()
        return _val_loss, _dict
    
    def val_prepare(self):
        self.val_loader = self.model.val_dataloader()

    def fit(self, lightmodule : pl.LightningModule):
        self.model  = lightmodule
        self.load_checkpoint()
        self.model  = self.model.cuda()
        self.device = self.model.device
        # self.model.model = self.model.model.cuda()
        # fit_prepare
        self.train_prepare()
        self.optims, self.scheds = self.model.configure_optimizers()
        print("We have {} optims".format(len(self.optims)))
        for i in range(self.epochs):
            self.model.current_epoch = i
            self.cur_epoch = i
            self._process()

        self.save_checkpoint()

    def save_checkpoint(self):
        self.model.save_checkpoint(saveTag = '') 

    def load_checkpoint(self):
        #self.model.load_checkpoint(saveTag = '') 
        pass