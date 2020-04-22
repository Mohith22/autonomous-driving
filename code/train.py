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

from model import Encoder_Decoder #From model.py

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
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
	train_labeled_scene_index = np.arange(106, 131)
	val_labeled_scene_index = np.arange(131, 134)
	labeled_trainset = LabeledDataset(image_folder=image_folder, annotation_file=annotation_csv, 
		scene_index=train_labeled_scene_index, transform=transform, extra_info=True)

	labeled_valset = LabeledDataset(image_folder=image_folder, annotation_file=annotation_csv,
		scene_index=val_labeled_scene_index,transform=transform,extra_info=True)

	trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=8, shuffle=True, num_workers=2, collate_fn=collate_fn)
	valloader = torch.utils.data.DataLoader(labeled_valset, batch_size=8, shuffle=True, num_workers=2, collate_fn=collate_fn)

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


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=(2,3)),
            nn.ReLU(inplace=True),
            #Current Size:- 64 x 256 x 308
            nn.MaxPool2d(kernel_size=2, stride=2),
            #Current Size:- 64 x 128 x 154
            nn.Conv2d(64, 192, kernel_size=3, padding=(1,2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            #Current Size:- 192 x 128 x 156
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #Current Size:- 384 x 128 x 156
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            #Current Size:- 256 x 128 x 156
            nn.MaxPool2d(kernel_size=2, stride=2),
            #Current Size:- 256 x 64 x 78
            nn.Conv2d(256, 192, kernel_size=(3,5), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            #Current Size:- 192 x 64 x 76
            nn.Conv2d(192, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #Current Size:- 64 x 64 x 76
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #Current Size:- 64 x 64 x 76
        )
        
    def forward(self,x):
        return self.encoder_features(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_features = nn.Sequential(
        	nn.Upsample(size=(100,100), mode='bilinear', align_corners=True),
        	#Current Size:- 384 x 100 x 100
        	nn.ConvTranspose2d(384, 256, kernel_size=4, stride=2, padding=1),
        	nn.ReLU(inplace=True),
                nn.BatchNorm2d(256),
        	#Current Size:- 256 x 200 x 200
        	nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        	#Current Size:- 192 x 200 x 200
        	nn.ConvTranspose2d(192, 64, kernel_size=4, stride=2, padding=1),
        	nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
        	#Current Size:- 64 x 400 x 400
        	nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #Current Size:- 32 x 400 x 400
        	nn.ConvTranspose2d(32, 1,kernel_size=4, stride=2, padding=1),
        	nn.ReLU(inplace=True),
        	#Current Size:- 1 x 800 x 800
        )
        
    def forward(self,x):
        return self.decoder_features(x)


class Encoder_Decoder(nn.Module):
    def __init__(self):
        super(Encoder_Decoder, self).__init__()
        #Input Size:- 3 x 256 x 306
        self.encoders = nn.ModuleList()
        for _ in range(6):
            self.encoders.append(Encoder())
        self.decoder = Decoder()

    def forward(self, x):
        x = x.permute(1,0,2,3,4)
        x = x-0.5
        encoder_outs = []
        for i in range(6):
            encoder_outs.append(self.encoders[i](x[i]))
        encoder_output = torch.stack(encoder_outs).permute(0,2,1,3,4)
        encoder_output = torch.cat([i for i in encoder_output]).permute(1,0,2,3)
        decoder_output = self.decoder(encoder_output)
        return decoder_output


def main():

    image_folder = '../data'
    annotation_csv = '../data/annotation.csv'
    trainloader, valloader = LoadData(image_folder, annotation_csv)
    
    sample, target, road_image, extra = iter(trainloader).next()
    #print(road_image)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    model = Encoder_Decoder()
    model.to(device)
    model.apply(weight_init)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    num_epochs = 150
    model.train()

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        data_len = len(trainloader)
        for i, data in enumerate(trainloader, 0):
            sample, target, road_image, extra  = data
            optimizer.zero_grad()
            outputs = model(torch.stack(sample).to(device))
            outputs = torch.squeeze(outputs)
            road_image_true = torch.stack([torch.Tensor(x.numpy()) for x in road_image]).to(device)
            loss = criterion(outputs, road_image_true) + 10*criterion(outputs*road_image_true, road_image_true)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('[%d, %5d] loss: %.3f' % (epoch + 1, num_epochs, running_loss / data_len))


if __name__ == '__main__':
	main()
