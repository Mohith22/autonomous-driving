# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset


class NYUDLDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(NYUDLDataset, self).__init__(*args, **kwargs)

        self.intrinsics = {'CAM_FRONT_LEFT': [[0.7181685027124183, 0.0, 0.500960762369281, 0.0], [0.0, 0.8584357883984375, 0.5118594453613281, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 
                          'CAM_FRONT': [[0.7210918637009804, 0.0, 0.5078705761846405, 0.0], [0.0, 0.8619301183300782, 0.5120937291210937, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 
                          'CAM_FRONT_RIGHT': [[0.7192903106781046, 0.0, 0.5056776937908497, 0.0], [0.0, 0.8597766994824219, 0.5091691258007812, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 
                          'CAM_BACK_LEFT': [[0.7200021624836601, 0.0, 0.5002429094035947, 0.0], [0.0, 0.86062758484375, 0.5095453828027344, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 
                          'CAM_BACK': [[0.7213481897222223, 0.0, 0.5036395417075163, 0.0], [0.0, 0.8622365080273438, 0.5158898733105469, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 
                          'CAM_BACK_RIGHT': [[0.7202927750898693, 0.0, 0.4964567662009804, 0.0], [0.0, 0.8609749577246094, 0.5132993482421875, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]}


        self.side_map = {"0": "CAM_BACK_LEFT", "1": "CAM_BACK_RIGHT", "2": "CAM_BACK", "3": "CAM_FRONT_LEFT", "4": "CAM_FRONT_RIGHT", "5": "CAM_FRONT"}

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index, side):

        image_path = os.path.join(self.data_path, "scene_"+str(folder), "sample_"+str(frame_index), self.side_map[side] + ".jpeg")
        '''
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        '''
        return image_path
