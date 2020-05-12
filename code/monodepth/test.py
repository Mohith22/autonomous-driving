import os
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import argparse
import torch
from torchvision import transforms

import networks

parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--idx', required = True, type=int)
args = parser.parse_args()

model_name = "mono_model/models/weights_8"
model_file = "res-net-50-logs"
data_input_path = "../DLProject/data"
data_output_path = "../DLProject/data_depth_8"

image_names = [
    'CAM_FRONT_LEFT.jpeg',
    'CAM_FRONT.jpeg',
    'CAM_FRONT_RIGHT.jpeg',
    'CAM_BACK_LEFT.jpeg',
    'CAM_BACK.jpeg',
    'CAM_BACK_RIGHT.jpeg',
    ]
encoder_path = os.path.join(model_file, model_name, "encoder.pth")
depth_decoder_path = os.path.join(model_file, model_name, "depth.pth")

encoder = networks.ResnetEncoder(50, False)
depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)

loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
depth_decoder.load_state_dict(loaded_dict)

encoder.eval()
depth_decoder.eval()

def save_image(scene_id, sample_id, img_path):
  sample_path = "/scene_" + str(scene_id) + "/sample_" + str(sample_id)
  img_path = '/' + img_path
  image_path = data_input_path + sample_path + img_path
  depth_image_path = data_output_path + sample_path

  input_image = pil.open(image_path).convert('RGB')
  original_width, original_height = input_image.size
  print(original_width, original_height)
  feed_height = loaded_dict_enc['height']
  feed_width = loaded_dict_enc['width']
  print(feed_height, feed_width)
  input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)

  input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)
  with torch.no_grad():
    features = encoder(input_image_pytorch)
    outputs = depth_decoder(features)

  disp = outputs[("disp", 0)]
  disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)
  # Saving colormapped depth image
  disp_resized_np = disp_resized.squeeze().cpu().numpy()
  #print(disp_resized_np.shape)
  if not os.path.exists(depth_image_path):
    os.makedirs(depth_image_path)
  #cv2.imwrite(depth_image_path + img_path, disp_resized_np)
  vmax = np.percentile(disp_resized_np, 95)
  plt.imsave(depth_image_path + img_path, arr = disp_resized_np, vmax = vmax, cmap = 'gray')

for scene_id in range(args.idx, args.idx+1):
  for sample_id in range(126):
    for img_path in image_names:
      save_image(scene_id, sample_id, img_path)
