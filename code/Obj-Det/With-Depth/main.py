# -- Imports -- #
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
from utils import *
from model import *

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_depth = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

#Load data return data loaders
def LoadData(depth_folder, image_folder, annotation_csv, args):
    train_labeled_scene_index = np.arange(106, 131) #128
    val_labeled_scene_index = np.arange(131, 134) #134
    labeled_trainset = LabeledDataset(depth_folder=depth_folder, image_folder=image_folder, annotation_file=annotation_csv, 
        scene_index=train_labeled_scene_index, transform=(transform, transform_depth), extra_info=True)

    labeled_valset = LabeledDataset(depth_folder=depth_folder, image_folder=image_folder, annotation_file=annotation_csv,
        scene_index=val_labeled_scene_index,transform=(transform, transform_depth),extra_info=True)

    trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=args.per_gpu_batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    valloader = torch.utils.data.DataLoader(labeled_valset, batch_size=args.per_gpu_batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)

    return trainloader, valloader


def evaluate(model, valloader, args, criterion):
    model.eval()
    ts = 0
    loss = 0
    with torch.no_grad():
        for data in valloader:
            sample, target, road_image, extra, depths  = data
            sample_with_depth = torch.cat((torch.stack(sample), torch.stack(depths)), dim=2)
            target_seg_mask = torch.stack([torch.Tensor(x.numpy()) for x in target]).to(args.device)
            outputs = model(sample_with_depth.to(args.device))
            outputs = torch.squeeze(outputs,dim=1)
            if (args.loss == "both"):
                loss = 0.5*criterion(outputs, target_seg_mask)
                outputs = torch.sigmoid(outputs)
                loss += 0.5*dice_loss(target_seg_mask, outputs)
            elif (args.loss == "dice"):
                outputs = torch.sigmoid(outputs)
                loss = dice_loss(target_seg_mask, outputs)
            elif (args.loss == "bce"):
                loss = criterion(outputs, target_seg_mask)
            outputs = (outputs >= args.thres).float()
            ts += BatchThreatScore(target_seg_mask, outputs)

    return loss/(args.per_gpu_batch_size*len(valloader)), ts/(args.per_gpu_batch_size*len(valloader))

def train_epoch(model, trainloader, args, criterion):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        sample, target, road_image, extra, depths  = data
        sample_with_depth = torch.cat((torch.stack(sample), torch.stack(depths)), dim=2)
        target_seg_mask = torch.stack([torch.Tensor(x.numpy()) for x in target]).to(args.device)
        optimizer.zero_grad()
        outputs = model(sample_with_depth.to(args.device))
        outputs = torch.squeeze(outputs,dim=1)
        if (args.loss == "both"):
            loss = 0.5*criterion(outputs, target_seg_mask)
            outputs = torch.sigmoid(outputs)
            loss += 0.5*dice_loss(target_seg_mask, outputs)
            loss.backward()
        elif (args.loss == "dice"):
            outputs = torch.sigmoid(outputs)
            loss = dice_loss(target_seg_mask, outputs)
            loss.backward()
        elif (args.loss == "bce"):
            loss = criterion(outputs, target_seg_mask)
            loss.backward()

        optimizer.step()
        running_loss += loss.item()
    return running_loss, model

def main():

    args = parse_args()
    set_seed(args.seed)

    image_folder = args.data_dir
    annotation_csv = args.annotation_dir
    model_dir = args.model_dir
    depth_folder = args.depth_dir

    trainloader, valloader = LoadData(depth_folder, image_folder, annotation_csv, args)

    model = UNet_Encoder_Decoder(3, args)
    model.to(args.device)
    model = model.apply(weight_init)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()
    num_epochs = args.num_train_epochs

    if (not os.path.exists(model_dir)):
        os.mkdir(model_dir)

    best_eval_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        data_len = len(trainloader)
        model.train()
        running_loss, model = train_epoch(trainloader, model, args, criterion)

        eval_loss, eval_acc = evaluate(model, valloader, args, criterion)
        print('[%d, %5d] Loss: %.3f Eval Loss: %.3f Eval ThreatScore: %.3f' % (epoch + 1, num_epochs, running_loss / (args.per_gpu_batch_size*data_len), eval_loss, eval_acc))
        
        torch.save(model.state_dict(), os.path.join(model_dir,'model_'+str(epoch)+'.pth'))
        if eval_acc > best_eval_acc: 
            torch.save(model.state_dict(), os.path.join(model_dir,'bestmodel_'+str(epoch)+'.pth'))
            best_eval_acc = eval_acc

if __name__ == '__main__':
	main()




