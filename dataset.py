import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import glob
import os


def loader(path, batch_size=200, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    transform =   transforms.Compose([
                                 transforms.RandomHorizontalFlip(p=0.4),
                                 transforms.ToTensor(),
                                 normalize,
                             ])


    return data.DataLoader(
        datasets.ImageFolder(path, transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory),datasets.ImageFolder(path, transform).class_to_idx

def test_loader(path, batch_size=100, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    transform = transforms.Compose([
                                 transforms.ToTensor(),
                                 normalize,
                             ])


    return data.DataLoader(
        datasets.ImageFolder(path,transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)


'''
def loader(path, batch_size=100, num_workers=4, pin_memory=True,no_crop=False, grayscale = False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    grayscale_trans = None
    if grayscale:
        grayscale_trans = transforms.Grayscale(num_output_channels=1)
        normalize = transforms.Normalize(mean=[0.485], std=[0.229])

    if not no_crop:
        transform =   transforms.Compose([grayscale_trans,
                                 transforms.Resize(256),
                                 transforms.RandomResizedCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])
    else:
        transform =   transforms.Compose([grayscale_trans,
                                 transforms.Resize(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])


    return data.DataLoader(
        datasets.ImageFolder(path, transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory),datasets.ImageFolder(path, transform).class_to_idx

def test_loader(path, batch_size=100, num_workers=4, pin_memory=True,no_crop = False, grayscale = False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    grayscale_trans = None
    if grayscale:
        grayscale_trans = transforms.Grayscale(num_output_channels=1)
        normalize = transforms.Normalize(mean=[0.485], std=[0.229])
        
    if not no_crop:
        transform = transforms.Compose([grayscale_trans,
                                 transforms.Scale(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])
    else:
        transform = transforms.Compose([grayscale_trans,
                                 transforms.Resize(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])


    return data.DataLoader(
        datasets.ImageFolder(path,transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)
'''



#dataset = datasets.ImageFolder(data_folder, transform=data_transforms)
#print(dataset.class_to_idx)