import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import datasets


import os
from PIL import Image

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

#from helper import convert_map_to_lane_map, convert_map_to_road_map, convert_target_to_seg_mask

NUM_SAMPLE_PER_SCENE = 126
NUM_IMAGE_PER_SAMPLE = 1#6
image_names = [
    'CAM_FRONT_LEFT.jpeg',
    'CAM_FRONT.jpeg',
    'CAM_FRONT_RIGHT.jpeg',
    'CAM_BACK_LEFT.jpeg',
    'CAM_BACK.jpeg',
    'CAM_BACK_RIGHT.jpeg',
    ]

def collate_fn(batch):
    return tuple(zip(*batch))

np.random.seed(0)


#transform = torchvision.transforms.ToTensor()
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform1 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.49, 0.45, 0.4], std=[0.229, 0.224, 0.225]),
])



color_jitter = transforms.ColorJitter(0.8 * 1, 0.8 * 1, 0.8 * 1, 0.2 * 1)
data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomApply([color_jitter], p=0.8),
                                      transforms.RandomGrayscale(p=0.2),
                                      GaussianBlur(kernel_size=int(0.1 * 3)),
                                      transforms.ToTensor()])



class DataSetWrapper(object):

    def __init__(self, batch_size, num_workers, valid_size, input_shape, s):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.input_shape = eval(input_shape)

    def get_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform()

#        train_dataset = datasets.STL10('./data', split='train+unlabeled', download=True,
#                                       transform=SimCLRDataTransform(data_augment))

        train_dataset = torchvision.datasets.DatasetFolder(root, loader, extensions=None,
                                        transform=None, target_transform=None, is_valid_file=None)

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.input_shape[0]),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])),
                                              transforms.ToTensor()])
        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        return train_loader, valid_loader




    #Load data return data loaders
    def LoadData(self, image_folder, annotation_csv):
        train_unlabeled_scene_index = np.arange(0, 90)
        val_unlabeled_scene_index = np.arange(90, 106)
        unlabeled_trainset = UnlabeledDataset(image_folder=image_folder,
            scene_index=train_unlabeled_scene_index,first_dim='image', transform=transform)

        unlabeled_valset = UnlabeledDataset(image_folder=image_folder,
            scene_index=val_unlabeled_scene_index,first_dim='image', transform=transform)

        trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=self.batch_size,#args.per_gpu_batch_size,
            shuffle=True, num_workers=0, pin_memory=True)
        valloader = torch.utils.data.DataLoader(unlabeled_valset, batch_size=self.batch_size,#args.per_gpu_batch_size,
            shuffle=True, num_workers=0, pin_memory=True)

        return trainloader, valloader


# The dataset class for unlabeled data.
class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, scene_index, first_dim, transform):
        """
        Args:
            image_folder (string): the location of the image folder
            scene_index (list): a list of scene indices for the unlabeled data 
            first_dim ({'sample', 'image'}):
                'sample' will return [batch_size, NUM_IMAGE_PER_SAMPLE, 3, H, W]
                'image' will return [batch_size, 3, H, W] and the index of the camera [0 - 5]
                    CAM_FRONT_LEFT: 0
                    CAM_FRONT: 1
                    CAM_FRONT_RIGHT: 2
                    CAM_BACK_LEFT: 3
                    CAM_BACK.jpeg: 4
                    CAM_BACK_RIGHT: 5
            transform (Transform): The function to process the image
        """

        self.image_folder = image_folder
        self.scene_index = scene_index
        self.transform = transform

        assert first_dim in ['sample', 'image']
        self.first_dim = first_dim

    def __len__(self):
        if self.first_dim == 'sample':
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE
        elif self.first_dim == 'image':
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE * 1#NUM_IMAGE_PER_SAMPLE
    
    def __getitem__(self, index):
        if self.first_dim == 'sample':
            scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
            sample_id = index % NUM_SAMPLE_PER_SCENE
            sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}') 

            images = []
            for image_name in image_names:
                image_path = os.path.join(sample_path, image_name)
                image = Image.open(image_path)
                images.append(self.transform(image))
            image_tensor = torch.stack(images)
            
            return image_tensor, image_tensor

        elif self.first_dim == 'image':
            scene_id = self.scene_index[index // (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)]
            sample_id = (index % (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)) // NUM_IMAGE_PER_SAMPLE
            image_name = 'CAM_FRONT.jpeg'#image_names[index % NUM_IMAGE_PER_SAMPLE]

            image_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}', image_name) 
            
            image = Image.open(image_path)
#            print('image opened ',index)

            return data_transforms(image), data_transforms(image)
#            return transform(image), transform(image)


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj
