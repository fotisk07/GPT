import torch
from model import GPT
from dataloader import DataloaderLite
from Trainer import train
import time
import torch.nn.functional as F
import wandb

# This class allows us to access dictionary keys as attributes
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
        
# Load config from yaml file
import yaml
with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
config = AttrDict(config) # convert to AttrDict
if config.wandb_log:
    run = wandb.init(
    # Set the project where this run will be logged
    project="GPT2",
    # Track hyperparameters and run metadata
    config=config
)

#---------------model initialization-------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.set_float32_matmul_precision('high')

original_model = GPT(config)
original_model.to(device) 
model = torch.compile(original_model) 
    
#---------------model training-------------------------------------------------------

# Set up the dataloader and optimizer
train_loader = DataloaderLite(config)
optimizer = model.configure_optimizers(config, device=str(device))

# Train the model (orignal_model is saved every checkpoint_steps)
train(model, original_model,  optimizer, train_loader, device, config)
        
import sys ; sys.exit(0)


#---------------model sampling-------------------------------------------------------

# generate! right now x is (B, T) where B = 5, T = 8

    




    