import argparse
import os
import random
import time
import warnings
import sys
import numpy as np
import torch
import torch.nn as nn 
import torchvision.transforms as transforms
from collections import Counter, OrderedDict
import torch.optim
from sklearn.metrics import confusion_matrix
from torchvision.datasets import FGVCAircraft
import os
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torchvision.models as models
import glob
from torchvision.datasets import INaturalist

from dataclasses import dataclass
global args
@dataclass
class ARGS():
    LR = 0.0005
    EPOCHS = 300
    BATCHSIZE = 16
    MOMENTUM = 0.9
    WORKERS = 0
    WEIGHTDECAY = 0
    T_MAX = 150
    CLS_CLASS = 37
    SEG_CLASS = 3
    SIZE = 224
    SEED = 38
    
args = ARGS()


transform_train = transforms.Compose([
        transforms.Resize((args.SIZE,args.SIZE)),
        transforms.RandomCrop(args.SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_val = transforms.Compose([
        transforms.Resize((args.SIZE,args.SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


classification_dataset_kingdom = INaturalist(root='./data', version="2021_train_mini", target_type = 'kingdom',download=True, transform=transform_train)
classification_val_dataset_kingdom = INaturalist(root='./data', version="2021_valid", target_type = 'kingdom',download=True, transform=transform_val)

classification_train_dataset_phylum = INaturalist(root='./data', version="2021_train_mini", target_type  = 'phylum',download=True, transform=transform_train)
classification_val_dataset_phylum = INaturalist(root='./data', version="2021_valid", target_type  = 'phylum',download=True, transform=transform_val)

classification_train_dataset_order = INaturalist(root='./data', version="2021_train_mini", target_type  = 'order',download=True, transform=transform_train)
classification_val_dataset_order = INaturalist(root='./data', version="2021_valid", target_type  = 'order',download=True, transform=transform_val)

classification_train_dataset_genus = INaturalist(root='./data', version="2021_train_mini", target_type  = 'genus',download=True, transform=transform_train)
classification_val_dataset_genus = INaturalist(root='./data', version="2021_valid", target_type  = 'genus',download=True, transform=transform_val)

classification_train_dataset_class = INaturalist(root='./data', version="2021_train_mini", target_type  = 'class',download=True, transform=transform_train)
classification_val_dataset_class = INaturalist(root='./data', version="2021_valid", target_type  = 'class',download=True, transform=transform_val)

print("\n>>data done!")