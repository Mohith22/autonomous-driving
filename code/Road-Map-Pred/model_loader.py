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

# Put your transform function here, we will use it for our dataloader
def get_transform(): 
    # return torchvision.transforms.Compose([
    # 
    # 
    # ])
    #transform = torchvision.transforms.ToTensor()
    return torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

class ModelLoader():
    # Fill the information for your team
    team_name = 'Connectionists'
    team_member = ["Mohith Damarapati", "Vikas Patidar", "Alfred Ajay Aureate Rajakumar"]
    contact_email = 'md4289@nyu.edu'

    def __init__(model_file):
        # You should 
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        # 
        self.roadmap_model = Resnet_Encoder_Decoder()
        self.objdet_model = Resnet_Encoder_Decoder()
        checkpoint = torch.load(model_file)

        self.roadmap_model.load_state_dict(checkpoint['roadmap'])
        self.objdet_model.load_state_dict(checkpoint['objdet'])

        self.roadmap_model.cuda()
        self.objdet_model.cuda()

    def get_bounding_boxes_util(object_map, th=0.5):
        object_map = object_map.squeeze(dim=1)
        object_map = (object_map >= th).float()
        _, contours, _ = cv2.findContours((object_map.detach().numpy()).astype(np.uint8),cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []

        for c in cnts:
            rect = cv2.boundingRect(c)
            x,y,w,h = rect
            if w*h < 200:
                continue
            pointsx = [x, x+w, x, x+w]
            pointsy = [y, y, y+h, y+h]
            pointsx = map(lambda a: (a-400)*1.0/10, pointsx)
            pointsy = map(lambda a: (a-400)*1.0/10, pointsy)
            x_tensor  = torch.tensor(pointsx, dtype = torch.float32)
            y_tensor = torch.tensor(pointsy, dtype = torch.float32)
            box = torch.stack((x_tensor,y_tensor))
            boxes.append(box)

        outputs = torch.stack(boxes)
        return outputs

    def get_bounding_boxes(samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        object_map = self.objdet_model(samples)
        return get_bounding_boxes_util(object_map, 0.5)

    def get_binary_road_map(samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 
        
        return (self.roadmap_model(samples) >= 0.5).float()
