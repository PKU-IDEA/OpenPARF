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

def rgbmap(tensor_val, file_name):
    pic = Image.new("RGB",(144,480))
    list_val = (tensor_val.reshape(144,480)).tolist()
    list_max = max(map(max,list_val))
    for i in range (0,144):
        for j in range (0,480):		
            pic.putpixel([i,j],(255,255- int(255*list_val[i][j]/list_max),255- int(255*list_val[i][j]/list_max)))
    pic.save(file_name)

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

      
        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dir_all=os.path.join(opt.dataroot, opt.phase)
        self.path_all=[]
        for file in os.listdir(self.dir_all):
            f_path=os.path.join(self.dir_all,file)
            if not os.path.isdir(f_path):
                if os.path.splitext(f_path)[1]=='.csv':
                    self.path_all.append(f_path)
        self.path_all=sorted(self.path_all)
        self.dataset_size=len(self.path_all)
        
           
            
        
        
      
    def __getitem__(self, index):        




        filename=self.path_all[index]
        train_data=pd.read_csv(filename)
        train_data=train_data[0:67200]
        
        fcell=torch.from_numpy((np.array(train_data.Cell)/32).reshape(1,140,480))
        fcellpin=torch.from_numpy((np.array(train_data.CellPin)/64).reshape(1,140,480))
        flocal=torch.from_numpy((np.array(train_data.LocalNet)/16).reshape(1,140,480))
        ffanin=torch.from_numpy((np.array(train_data.FanIn)/64).reshape(1,140,480))
        ffanout=torch.from_numpy((np.array(train_data.FanOut)/64).reshape(1,140,480))
        fff=torch.from_numpy(np.array(train_data.FF).reshape(1,140,480))
        flut=torch.from_numpy(np.array(train_data.LUT).reshape(1,140,480))
        fdemandh=torch.from_numpy(((np.array(train_data.demandW)+np.array(train_data.demandE))/400).reshape(1,140,480))
        fdemandv=torch.from_numpy(((np.array(train_data.DemandN)+np.array(train_data.demandS))/400).reshape(1,140,480))
        fdata=torch.stack((fcell,fcellpin,flocal,ffanin,ffanout,fff,flut,fdemandh,fdemandv),dim=1)
        hcong=torch.from_numpy((np.array(train_data.Hcong)/400).reshape(1,140,480))
        vcong=torch.from_numpy((np.array(train_data.Vcong)/400).reshape(1,140,480))

        ldata=torch.stack((hcong,vcong),dim=1)
        #fill 4x480
        fill=torch.zeros(1,9,4,480)
        fdata=torch.cat((fdata,fill),dim=2)
        fill=torch.zeros(1,2,4,480)
        ldata=torch.cat((ldata,fill),dim=2)
        fill=torch.zeros(1,1,144,480)
        ldata=torch.cat((ldata,fill),dim=1)

        A_tensor=fdata[0,:,:,:].float()
        B_tensor = inst_tensor = feat_tensor = 0
        B_tensor=ldata[0,:,:,:].float()
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_tensor=ldata[0,:,:,:].float()
        A_path=filename



        
        ### input B (real images)
#        if self.opt.isTrain or self.opt.use_encoded_image:
#            B_path = self.B_paths[index]   
#            B = Image.open(B_path).convert('RGB')
#            transform_B = get_transform(self.opt, params)      
#            B_tensor = self.c[index,:,:,:]
#
#        ### if using instance maps        
#        if not self.opt.no_instance:
#            inst_path = self.inst_paths[index]
#            inst = Image.open(inst_path)
#            inst_tensor = transform_A(inst)
#
#            if self.opt.load_features:
#                feat_path = self.feat_paths[index]            
#                feat = Image.open(feat_path).convert('RGB')
#                norm = normalize()
#                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return self.dataset_size // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
    
