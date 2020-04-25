import os
import random
from  tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
#from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.optim as optim
import torch.nn.init as init

from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box, weight_init

from model import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


#Load data return data loaders
def LoadData(image_folder):

    train_labeled_scene_index = np.arange(0, 30)
    val_labeled_scene_index = np.arange(31, 50)
    
    unlabeled_trainset = UnlabeledDataset(image_folder=image_folder, scene_index=train_labeled_scene_index, first_dim='image', transform=transform)

    unlabeled_valset = UnlabeledDataset(image_folder=image_folder, scene_index=val_labeled_scene_index, first_dim='image', transform=transform)

    trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=8, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(unlabeled_valset, batch_size=8, shuffle=True, num_workers=2)

    return trainloader, valloader

def evaluate(model, valloader):

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    eval_acc = (100 * correct / total)
    return eval_acc

def main():

    args = parse_args()
    set_seed(args.seed)

    image_folder = args.data_dir
    model_dir = args.model_dir

    trainloader, valloader = LoadData(image_folder)
    
    image, camera_index = iter(trainloader).next()

    model = BasicClassifierSSL()

    model.to(args.device)
    model = model.apply(weight_init)
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10

    best_eval_acc = 0.0

    if (not os.path.exists(model_dir)):
        os.mkdir(model_dir)

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            model.train()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('[%d, %5d] loss: %f' % (epoch + 1, num_epochs, running_loss/len(trainloader)))
        eval_acc = evaluate(model, valloader)

        if eval_acc > best_eval_acc:
            torch.save(model.state_dict(), model_dir+'/model_'+str(epoch)+'.pth')
            best_eval_acc = eval_acc

        print('Eval Accuracy: %d %%' % eval_acc)
        print('Best Eval Accuracy: %d %%' % best_eval_acc)

if __name__ == '__main__':
    main()




