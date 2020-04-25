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
import torchvision
#from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.optim as optim
import torch.nn.init as init

from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box

from model import *

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

transform = torchvision.transforms.ToTensor()

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
	train_labeled_scene_index = np.arange(0, 80)
	val_labeled_scene_index = np.arange(81, 105)
	unlabeled_trainset = UnlabeledDataset(image_folder=image_folder, scene_index=train_labeled_scene_index, first_dim='sample', transform=transform)

	unlabeled_valset = UnlabeledDataset(image_folder=image_folder, scene_index=val_labeled_scene_index, first_dim='sample', transform=transform)

	trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=8, shuffle=True, num_workers=2)
	valloader = torch.utils.data.DataLoader(labeled_valset, batch_size=8, shuffle=True, num_workers=2)

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


def evaluate():

def main():

    image_folder = '../data'
    annotation_csv = '../data/annotation.csv'
    model_dir = ''
    trainloader, valloader = LoadData(image_folder, annotation_csv)
    
    image, camera_index = iter(trainloader).next()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Jigsaw_Encoder()
    model.to(device)
    model.apply(weight_init)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 50

    model.train()

    best_eval_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        data_len = len(trainloader)
        for i, data in enumerate(trainloader, 0):
            sample, target, road_image, extra  = data
            optimizer.zero_grad()
            outputs = model(torch.stack(sample).to(device))
            outputs = torch.squeeze(outputs)
            eval_acc = evaluate(model, valloader)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('[%d, %5d] loss: %.3f' % (epoch + 1, num_epochs, running_loss / data_len))
        if (not os.path.exists(model_dir):
        	os.mkdir(model_dir)
        if eval_acc > best_eval_acc:
            torch.save(model.state_dict(), model_dir+'/model_'+str(epoch)+'.pth')

if __name__ == '__main__':
	main()




