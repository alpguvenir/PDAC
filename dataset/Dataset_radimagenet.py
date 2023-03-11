import json
import os
import yaml
import random

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
import numpy as np
from matplotlib import pyplot as plt

import nibabel as nib
from PIL import Image, ImageEnhance
from skimage import color
from skimage import io
import cv2 

from typing import Any, Type

class Dataset_radimagenet(torch.utils.data.Dataset):

    def __init__(self, ct_scans: list[str], ct_labels: list[str], transforms: dict = None, train_mode = False, scan_type: Type[Any] = np.float32, label_type: Type[Any] = np.int64):

        self.ct_scans = ct_scans
        self.ct_labels = ct_labels

        self.transforms = transforms
        self.train_mode = train_mode

        self.scan_type = scan_type
        self.label_type = label_type

        with open('../parameters.yml') as params:
            params_dict = yaml.safe_load(params)
        
        self.params_dict = params_dict


    def __len__(self) -> int:
        return len(self.ct_scans)


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ct_scan, ct_label = (
            self.ct_scans[index],
            self.ct_labels[index]   
        )

        np.seterr(all='raise')

        height = self.transforms['Resize']['height']
        width = self.transforms['Resize']['width']

        # Open image by PIL, convert to numpy
        ct_instance = Image.open(ct_scan)
        ct_instance = np.asarray(ct_instance)
        

        # H, W, Layers -> 224 x 224 x 3
        ct_instance_resized = cv2.resize(ct_instance, dsize=(height, width), interpolation=cv2.INTER_CUBIC)


        lower_bound = np.amin(ct_instance_resized)
        upper_bound = np.amax(ct_instance_resized)

        ct_instance_resized_normalized = ct_instance_resized

        if not(upper_bound - lower_bound == 0):
            ct_instance_resized_normalized = (ct_instance_resized - (lower_bound)) / ((upper_bound) - (lower_bound))
            #plt.imshow(ct_instance_resized_normalized, cmap='gray')
            #plt.savefig("/home/guevenira/attention_CT/PDAC/debugging/rad.png")
            #plt.close()
            #print(ct_scan)
            #exit()
        else:
            ct_instance_resized_normalized = ct_instance_resized

        
        
        ct_instance_tensor = torch.tensor(ct_instance_resized_normalized.copy(), dtype=torch.float)
            
        return ct_instance_tensor, torch.tensor(ct_label)