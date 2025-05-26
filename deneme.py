import torch
import os
from config.args_parser import get_args
from models.vit_autoencoder import ViTEncoderDecoder, freeze_vit_layers
from models.utils import get_vgg_model
from utils.checkpoint import load_checkpoint
from train.train import train_model
from test.test import test_model_with_patch_analysis
from utils.data_loader import get_data_loaders
from utils.device_utils import get_device

if __name__ == "__main__":
    args = get_args()
    device = get_device()
    
    print(args.test_data_path)
    
    a,_= get_data_loaders(args.test_data_path)
    print(a)
