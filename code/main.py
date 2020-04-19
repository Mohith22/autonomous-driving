#Object Detection Baseline 

import os
import random

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
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.optim as optim

from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box

from model import Encoder_Decoder #From model.py

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

transform = torchvision.transforms.ToTensor()


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
	criterion = nn.BCEWithLogitsLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

	num_epochs = 4
	
	model.train()

	for epoch in range(num_epochs):
		running_loss = 0.0
		data_len = len(trainloader)
		for i, data in enumerate(trainloader, 0):
			sample, target, road_image, extra  = data
			optimizer.zero_grad()
			outputs = model(torch.stack(sample).to(device))
			outputs = torch.squeeze(outputs)
			road_image_true = torch.stack([torch.Tensor(x.numpy()) for x in road_image]).to(device)
			
			#loss = ComputeLoss(criterion, road_image_true, outputs)
			#print(road_image_true.size(), outputs.size())
			
			loss = criterion(outputs, road_image_true)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			#if i%10 == 0:
			#	print(loss.item())
			'''
			if i % mini_batch_size == mini_batch_size-1:
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / mini_batch_size))
				running_loss = 0.0
			'''
		print('[%d, %5d] loss: %.3f' % (epoch + 1, num_epochs, running_loss / data_len))

if __name__ == '__main__':
	main()




