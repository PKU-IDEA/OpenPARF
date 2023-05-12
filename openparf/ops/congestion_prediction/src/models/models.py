### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
from .pix2pixHD_model import InferenceModel

def create_model(opt):
    model = InferenceModel()

    model.initialize(opt)

    return model
