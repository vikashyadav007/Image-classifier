import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image
import json
from matplotlib.ticker import FormatStrFormatter
from PIL import Image
from model import FlowerModel



def label_mapping():
    cat_to_name=''
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    return cat_to_name

def process_image(image):
    prc_img = Image.open(image)
   
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    output = img_transform(prc_img)
    
    return output

def create_data_loaders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomRotation(20),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    test_trasforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])


    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_trasforms)



    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=64)

    return train_loader, valid_loader, test_loader, train_datasets 


def cli_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory")
    parser.add_argument("--save_directory", default="./")
    parser.add_argument("--arch", action="store", default="densenet121")
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--hidden_units", default=512, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--gpu", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = cli_options()
   
    fm = FlowerModel(args.arch, args.hidden_units,
                          args.learning_rate, args.gpu)

    train_loader, valid_loader, test_loader, train_datasets = create_data_loaders(
        args.data_directory)
    
    cat_to_name = label_mapping()

    fm.train(args.save_directory, train_loader,
             valid_loader,test_loader, train_datasets, args.epochs)

    fm.predict(train_loader)

    fm.save_model("checkpoint_1.pth",train_datasets)
    fm.loading_checkpoint("checkpoint_1.pth")

    
    # fr.test(test_loader)