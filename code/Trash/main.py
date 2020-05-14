# -- Imports -- #
# Main file for data augmentation #
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
from data_transforms import data_transforms,data_jitter_hue,data_jitter_brightness,data_jitter_saturation,data_jitter_contrast

#transform = torchvision.transforms.ToTensor()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


#Load data return data loaders
def LoadData(image_folder, annotation_csv, args):
        train_labeled_scene_index = np.arange(106, 131)
        val_labeled_scene_index = np.arange(131, 134)

        extra_transforms = [data_transforms, data_jitter_brightness, data_jitter_hue, data_jitter_contrast, data_jitter_saturation]

        extra_datasets = []
        for t in extra_transforms:
                extra_datasets.append(LabeledDataset(image_folder=image_folder, annotation_file=annotation_csv,scene_index=train_labeled_scene_index, transform=t, extra_info=True))
        trainloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(extra_datasets), batch_size=args.per_gpu_batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory = True)

        labeled_valset = LabeledDataset(image_folder=image_folder, annotation_file=annotation_csv,
                scene_index=val_labeled_scene_index,transform=transform,extra_info=True)

        valloader = torch.utils.data.DataLoader(labeled_valset, batch_size=args.per_gpu_batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory = True)

        return trainloader, valloader



def evaluate(model, valloader, args, criterion):
    model.eval()
    ts_roadmap = 0
    ts_objdet = 0
    roadmap_loss = 0
    objdet_loss = 0
    with torch.no_grad():
        for data in valloader:
            sample, target, road_image, extra  = data

            road_image_true = torch.stack([torch.Tensor(x.numpy()) for x in road_image]).to(args.device)
            target_seg_mask = torch.stack([torch.Tensor(x.numpy()) for x in target]).to(args.device)
            outputs_roadmap, outputs_objdet = model(torch.stack(sample).to(args.device))

            outputs_roadmap = torch.squeeze(outputs_roadmap,dim=1)
            outputs_objdet = torch.squeeze(outputs_objdet,dim=1)

            if (args.loss == "both"):
                loss = 0.5*criterion(outputs, road_image_true)
                outputs = torch.sigmoid(outputs)
                loss += 0.5*dice_loss(road_image_true, outputs)
            elif (args.loss == "dice"):
                outputs = torch.sigmoid(outputs)
                loss = dice_loss(road_image_true, outputs)
            elif (args.loss == "bce"):
                loss1 = criterion(outputs_roadmap, road_image_true)
                outputs_objdet = torch.sigmoid(outputs_objdet)
                loss2 = 3 * dice_loss(target_seg_mask, outputs_objdet)
                roadmap_loss += loss1.item()
                objdet_loss += loss2.item()
            outputs_roadmap = (outputs_roadmap >= args.thres).float()
            ts_roadmap += BatchThreatScore(road_image_true, outputs_roadmap)
            ts_objdet += BatchThreatScore(target_seg_mask, outputs_objdet)

    return roadmap_loss/(args.per_gpu_batch_size*len(valloader)), objdet_loss/(args.per_gpu_batch_size*len(valloader)), ts_roadmap/(len(valloader)*args.per_gpu_batch_size), ts_objdet/(len(valloader)*args.per_gpu_batch_size)

def train_epoch(model, trainloader, args, criterion):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
            sample, target, road_image, extra  = data

            road_image_true = torch.stack([torch.Tensor(x.numpy()) for x in road_image]).to(args.device)
            target_seg_mask = torch.stack([torch.Tensor(x.numpy()) for x in target]).to(args.device)

            optimizer.zero_grad()
            outputs_roadmap, outputs_objdet = model(torch.stack(sample).to(args.device))

            outputs_roadmap = torch.squeeze(outputs_roadmap,dim=1)
            outputs_objdet = torch.squeeze(outputs_objdet,dim=1)

            if (args.loss == "both"):
                loss = 0.5*criterion(outputs, road_image_true)
                outputs = torch.sigmoid(outputs)
                loss += 0.5*dice_loss(road_image_true, outputs)
                loss.backward()
            elif (args.loss == "dice"):
                outputs = torch.sigmoid(outputs)
                loss = dice_loss(road_image_true, outputs)
                loss.backward()
            elif (args.loss == "bce"):
                loss = criterion(outputs_roadmap, road_image_true)
                outputs_objdet = torch.sigmoid(outputs_objdet)
                loss += 3*dice_loss(target_seg_mask, outputs_objdet)
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

    trainloader, valloader = LoadData(image_folder, annotation_csv, args)

    model = UNet_Encoder_Decoder(3,args)
    model.to(args.device)
    model = model.apply(weight_init)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()
    num_epochs = args.num_train_epochs

    if (not os.path.exists(model_dir)):
        os.mkdir(model_dir)

    best_eval_acc = 0.0

    #model.load_state_dict(torch.load(os.path.join(model_dir,"bestmodel_6.pth")))

    for epoch in tqdm(range(num_epochs)):
        data_len = len(trainloader)
        model.train()
        running_loss, model = train_epoch(model, trainloader, args, criterion)

        eval_roadmap_loss, eval_objdet_loss, eval_roadmap_acc, eval_objdet_acc  = evaluate(model, valloader, args, criterion)
        print('[%d, %5d] Loss: %.3f Eval Road Map Loss: %.3f Eval ObjDet Loss: %.3f Eval RoadMap ThreatScore: %.3f Eval ObjDet ThreatScore: %.3f' % (epoch + 1, num_epochs, running_loss / (args.per_gpu_batch_size*data_len), eval_roadmap_loss, eval_objdet_loss, eval_roadmap_acc, eval_objdet_acc))
        
        torch.save(model.state_dict(), os.path.join(model_dir,'model_'+str(epoch)+'.pth'))
        if eval_acc > best_eval_acc: 
            torch.save(model.state_dict(), os.path.join(model_dir,'bestmodel_'+str(epoch)+'.pth'))
            best_eval_acc = eval_acc

if __name__ == '__main__':
	main()




