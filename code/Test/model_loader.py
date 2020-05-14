"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# import your model class
# import ...
from model import *
import cv2
import numpy as np

from shapely.geometry import Polygon

# Put your transform function here, we will use it for our dataloader
def get_transform(): 
    # return torchvision.transforms.Compose([
    # 
    # 
    # ])
    #transform = torchvision.transforms.ToTensor()
    return torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class ModelLoader():
    # Fill the information for your team
    team_name = 'Autowalas'
    team_member = ["Mohith Damarapati", "Vikas Patidar", "Alfred Ajay Aureate Rajakumar"]
    contact_email = 'md4289@nyu.edu'
    round_number = 3

    def __init__(self,model_file='weights.pth'):
        # You should 
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        # 
        self.roadmap_model = UNet_Encoder_Decoder()
        self.objdet_model = UNet_Encoder_Decoder()
        self.depth_encoder = ResnetEncoder(50)
        self.depth_decoder = DepthDecoder(num_ch_enc=self.depth_encoder.num_ch_enc)

        checkpoint = torch.load(model_file)

        self.roadmap_model.load_state_dict(checkpoint['roadmap'])
        self.objdet_model.load_state_dict(checkpoint['objdet'])
        self.depth_encoder.load_state_dict(checkpoint['depth-encoder'])
        self.depth_decoder.load_state_dict(checkpoint['depth-decoder'])

        self.roadmap_model.cuda()
        self.objdet_model.cuda()
        self.depth_encoder.cuda()
        self.depth_decoder.cuda()

    def get_bounding_boxes_util(self, object_map, th=0.5):
        object_map = object_map.squeeze(dim=1)
        object_map = (object_map >= th).float()
        contours, _ = cv2.findContours((object_map[0].cpu().detach().numpy()).astype(np.uint8),cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if (area<800):
              continue
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            print(rect)
            pointsx = box[:,0]
            pointsy = box[:,1]
            pointsx = list(map(lambda a: (a-400)*1.0/10, pointsx))
            pointsy = list(map(lambda a: (a-400)*1.0/10, pointsy))
            x_tensor  = torch.tensor(pointsx, dtype = torch.float32)
            y_tensor = torch.tensor(pointsy, dtype = torch.float32)
            box = torch.stack((x_tensor,y_tensor))
            boxes.append(box)
        if (len(boxes) == 0):
          return torch.empty((1,1,2,4))
        outputs = torch.stack(boxes)
        return outputs.unsqueeze(0)

    def get_depth(self, sample):

        transform_depth_before = transforms.Compose([
                            transforms.Resize(256,320),
                        ])

        transform_depth_after = transforms.Compose([
                            transforms.Resize(256,306),
                            transforms.Normalize(mean=[0.485], std=[0.229]),
                        ])

        input_image_pytorch = transforms.depth_before.apply(sample)
        with torch.no_grad():
            features = self.depth_encoder(input_image_pytorch)
            outputs = self.depth_decoder(features)

        depth = outputs[("disp", 0)]
        depth= transform_depth_after.apply(depth)
        return depth
        

    def get_bounding_boxes(self,samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        depths = []
        for i in range(6):
            depths.append(self.get_depth(sample[:,i,:,:,:]))
        depths = torch.cat(depths.unsqueeze(1), dim=1)

        sample_with_depth = torch.cat(samples, depths, dim=2)

        object_map = self.objdet_model(samples_with_depth)

        x = self.get_bounding_boxes_util(object_map, 0.5)

        #print(x.size())

        return x

    def get_binary_road_map(self,samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 
        depths = []
        for i in range(6):
            depths.append(get_depth(sample[:,i,:,:,:]))
        depths = torch.cat(depths.unsqueeze(1), dim=1)

        sample_with_depth = torch.cat(samples, depths, dim=2)
        
        return (self.roadmap_model(samples_with_depth) >= 0.5).float()
