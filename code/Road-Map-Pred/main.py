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
from helper import collate_fn, draw_box

from model import *

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

#transform = torchvision.transforms.ToTensor()
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
def weight_init(m):
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

#Load data return data loaders
def LoadData(image_folder, annotation_csv):
	train_labeled_scene_index = np.arange(106, 128)
	val_labeled_scene_index = np.arange(128, 134)
	labeled_trainset = LabeledDataset(image_folder=image_folder, annotation_file=annotation_csv, 
		scene_index=train_labeled_scene_index, transform=transform, extra_info=True)

	labeled_valset = LabeledDataset(image_folder=image_folder, annotation_file=annotation_csv,
		scene_index=val_labeled_scene_index,transform=transform,extra_info=True)

	trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)
	valloader = torch.utils.data.DataLoader(labeled_valset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)

	return trainloader, valloader

#ThreatScore - Determines Model Performance - Challenge Metric
def ThreatScore(true, pred):
	TP = 0
	FP = 0
	FN = 0
	n = len(true)
	for i in range(n):
		for j in range(n):
			if true[i][j] == True and pred[i][j] == True:
				TP += 1
			elif true[i][j] == False and pred[i][j] == True:
				FP += 1
			elif true[i][j] == True and pred[i][j] == False:
				FN += 1
	return TP/(TP+FP+FN)

def ComputeLoss(criterion, true, pred):
	loss = 0.0
	for i in range(800):
		for j in range(800):
			loss += criterion(true[:,:,i,j], pred[:,i,j])
	return loss


def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

def main():

    image_folder = '../data'
    annotation_csv = '../data/annotation.csv'
    trainloader, valloader = LoadData(image_folder, annotation_csv)
    
    sample, target, road_image, extra = iter(trainloader).next()
    #print(road_image)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    model = Mini_Encoder_Decoder()
    model.to(device)
    model.apply(weight_init)
    #criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 50
    #model.load_state_dict(torch.load("models/model_1.pth"))

    model.train()

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        data_len = len(trainloader)
        for i, data in enumerate(trainloader, 0):
            sample, target, road_image, extra  = data
            optimizer.zero_grad()
            outputs = model(torch.stack(sample).to(device))
            outputs = torch.squeeze(outputs)
            #loss = criterion(outputs, road_image_true) + 10*criterion(outputs*road_image_true, road_image_true)
            loss = dice_loss(road_image_true, outputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('[%d, %5d] loss: %.3f' % (epoch + 1, num_epochs, running_loss / data_len))
        if (not os.path.exists("models_senc_dec_norm")):
        	os.mkdir("models_senc_dec_norm")
        torch.save(model.state_dict(), 'models_senc_dec_norm/model_'+str(epoch)+'.pth')

if __name__ == '__main__':
	main()




