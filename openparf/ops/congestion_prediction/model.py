### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from torch.autograd import Variable
from .src.options.test_options import TestOptions        
from .src.data.data_loader import CreateDataLoader   
from .src.models.models import create_model          
from .src.util import util                          
import torch
import time

def model(
        horizontal_utilization_map,
        vertical_utilization_map,
        pin_density_map):


    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip


    data_loader = CreateDataLoader(opt, horizontal_utilization_map, vertical_utilization_map, pin_density_map)
    dataset = data_loader.load_data()

    # test
    model = create_model(opt)

    for i, data in enumerate(dataset):
        data['image']=torch.zeros(1)
        # T=time.time()
        if i >= opt.how_many:
            break
        if opt.data_type == 16:
            data['label'] = data['label'].half()
            data['inst']  = data['inst'].half()
        elif opt.data_type == 8:
            data['label'] = data['label'].uint8()
            data['inst']  = data['inst'].uint8()
        if opt.export_onnx:
            print ("Exporting to ONNX: ", opt.export_onnx)
            assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
            torch.onnx.export(model, [data['label'], data['inst']],
                          opt.export_onnx, verbose=True)
            exit(0)     
        generated = model.inference(data['label'], data['inst'], data['image'])
        
        # print(time.time()-T)
    return generated
