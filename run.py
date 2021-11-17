import yaml
import argparse
import numpy as np

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn 
from myTrainer import MyTrainer
from pytorch_lightning.logging import TestTubeLogger
import os

#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(description='Generic runner for VAE models')

parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
)

config['exp_params']['in_channels'] = config['model_params']['in_channels'] 
# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False
 
'''
a2array = [0.7, 2, 4]
a2array.reverse()
for i in range(len(a2array)):
    a4 = round(a2array[i], 4)

    config['model_params']['hypermeter'][2] = a4
    config['exp_params']['lantentznames'] = config['logging_params']['name'] + str(config['model_params']['hypermeter']) +  str(config['model_params']['latent_dim'])
    model = vae_models[config['model_params']['name']](**config['model_params'])
    path = config['logging_params']['save_dir'] + config['logging_params']['name'] + ".ckpt" 
    experiment = VAEXperiment(model, config['exp_params'])
    runner = MyTrainer(config['trainer_params'])
    print(f"======= Training {config['model_params']['name']} ======={str(config['model_params']['hypermeter'])}" )
    runner.fit(experiment)
'''
config['exp_params']['lantentznames'] = config['logging_params']['name'] + str(config['model_params']['hypermeter']) +  str(config['model_params']['latent_dim'])
model = vae_models[config['model_params']['name']](**config['model_params'])
path = config['logging_params']['save_dir'] + config['logging_params']['name'] + ".ckpt" 
experiment = VAEXperiment(model, config['exp_params'])
runner = MyTrainer(config['trainer_params'])
print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)
