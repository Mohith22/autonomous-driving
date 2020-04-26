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
import torch.optim as optim
from arguments import parse_args

from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box, weight_init

from model import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

#transform = torchvision.transforms.ToTensor()
transform = transforms.Compose([
    transforms.ToTensor(),
])


#Load data return data loaders
def LoadData(image_folder, annotation_csv):
    train_labeled_scene_index = np.arange(106, 128) #128
    val_labeled_scene_index = np.arange(128, 134) #134
    labeled_trainset = LabeledDataset(image_folder=image_folder, annotation_file=annotation_csv, 
        scene_index=train_labeled_scene_index, transform=transform, extra_info=True)

    labeled_valset = LabeledDataset(image_folder=image_folder, annotation_file=annotation_csv,
        scene_index=val_labeled_scene_index,transform=transform,extra_info=True)

    trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=8, shuffle=True, num_workers=2, collate_fn=collate_fn)
    valloader = torch.utils.data.DataLoader(labeled_valset, batch_size=8, shuffle=True, num_workers=2, collate_fn=collate_fn)

    return trainloader, valloader

#ThreatScore - Determines Model Performance - Challenge Metric
def ThreatScore(true, pred):
    tp = (true * pred).sum()
    return tp * 1.0 / (true.sum() + pred.sum() - tp)

def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

def evaluate(model, valloader, args):
    model.eval()
    ts = 0
    with torch.no_grad():
        for data in valloader:
            sample, target, road_image, extra  = data
            target_seg_mask = torch.stack([torch.Tensor(x.numpy()) for x in target]).to(args.device)
            outputs = model(torch.stack(sample).to(args.device))
            outputs = torch.squeeze(outputs)
            ts += ThreatScore(target_seg_mask, outputs)

    return ts/len(valloader)

def main():

    args = parse_args()
    set_seed(args.seed)

    image_folder = args.data_dir
    annotation_csv = args.annotation_dir
    model_dir = args.model_dir

    trainloader, valloader = LoadData(image_folder, annotation_csv)

    model = Mini_Encoder_Decoder()
    model.to(args.device)
    model = model.apply(weight_init)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = args.num_train_epochs

    if (not os.path.exists(model_dir)):
        os.mkdir(model_dir)
    
    model.train()

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        data_len = len(trainloader)
        for i, data in enumerate(trainloader, 0):
            sample, target, road_image, extra  = data
            road_image_true = torch.stack([torch.Tensor(x.numpy()) for x in road_image]).to(args.device)
            optimizer.zero_grad()
            outputs = model(torch.stack(sample).to(args.device))
            outputs = torch.squeeze(outputs)
            loss = dice_loss(road_image_true, outputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        eval_acc = evaluate(model, valloader, args)
        print('[%d, %5d] Loss: %.3f Eval ThreatScore: %.3f' % (epoch + 1, num_epochs, running_loss / data_len, eval_acc))
    
        torch.save(model.state_dict(), os.path.join(model_dir,'model_'+str(epoch)+'.pth'))
        if eval_acc > best_eval_acc: 
            torch.save(model.state_dict(), os.path.join(model_dir,'model_'+str(epoch)+'.pth'))
            best_eval_acc = eval_acc

if __name__ == '__main__':
	main()




