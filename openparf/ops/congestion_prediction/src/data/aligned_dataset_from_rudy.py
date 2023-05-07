### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
import os
from .base_dataset import BaseDataset, get_params, get_transform, normalize
from .image_folder import make_dataset
from PIL import Image
import torch
import numpy as np
import pandas as pd


class AlignedDataset(BaseDataset):
    def initialize(self, opt, horizontal_utilization_map, vertical_utilization_map, pin_density_map):
        self.opt = opt
        self.root = opt.dataroot
        self.horizontal_utilization_map = horizontal_utilization_map    
        self.vertical_utilization_map = vertical_utilization_map
        self.pin_density_map = pin_density_map


     
    def __getitem__(self, index):        

        fcellpin=self.pin_density_map    #(1,168,480)
        fdemandh=self.horizontal_utilization_map
        fdemandv=self.vertical_utilization_map
        fdata=torch.stack((fcellpin,fdemandh,fdemandv),dim=1)
        hcong=torch.zeros_like(fcellpin)
        vcong=torch.zeros_like(fcellpin)
        ldata=torch.stack((hcong,vcong),dim=1)
        
        #fill 4x480
        fill=torch.zeros(1,3,8,480)
        fdata=torch.cat((fdata,fill),dim=2)
        fill=torch.zeros(1,1,168,480)
        ldata=torch.cat((ldata,fill),dim=1)

        A_tensor=fdata[0,:,:,:].float()
        B_tensor = inst_tensor = feat_tensor = 0
        B_tensor=ldata[0,:,:,:].float()
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_tensor=ldata[0,:,:,:].float()
                       

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor }

        return input_dict

    def __len__(self):
        return 1

    def name(self):
        return 'AlignedDataset'
    
