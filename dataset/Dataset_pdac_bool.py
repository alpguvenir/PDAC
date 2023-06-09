# type: ignore
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name
# pylint: disable=unnecessary-lambda-assignment
# pylint: disable=no-member
# pylint: disable=trailing-whitespace
# pylint: disable=line-too-long
# pylint: disable=wrong-import-position
# pylint: disable=wrong-import-order
# pylint: disable=pointless-string-statement
# pylint: disable=unused-import
# pylint: disable=unspecified-encoding
# pylint: disable=consider-using-enumerate
# pylint: disable=superfluous-parens
# pylint: disable=consider-using-f-string
# pylint: disable=fixme

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

class Dataset_pdac_bool(torch.utils.data.Dataset):

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

        
        amin = self.transforms['Clip']['amin']
        amax = self.transforms['Clip']['amax']

        #lower_bound = self.transforms['Normalize']['bounds'][0]
        #upper_bound = self.transforms['Normalize']['bounds'][1]
        lower_bound = amin
        upper_bound = amax
        
        height = self.transforms['Resize']['height']
        width = self.transforms['Resize']['width']
        
        crop_height_begin = self.transforms['Crop-Height']['begin']
        crop_height_end = self.transforms['Crop-Height']['end']
        crop_width_begin = self.transforms['Crop-Width']['begin']
        crop_width_end = self.transforms['Crop-Width']['end']

        max_number_of_layers = self.transforms['Max-Layers']['max']
        uniform_number_of_layers = self.transforms['Uniform-Layers']['uniform']
        zeropad_number_of_layers = self.transforms['Zero-Pad-Layers']['zeropad']

        limit_max_number_of_layers = self.transforms['limit-max-number-of-layers']['bool']
        set_uniform_number_of_layers = self.transforms['uniform-number-of-layers']['bool']
        zero_pad_number_of_layers = self.transforms['zero-pad-number-of-layers']['bool']
        #assert limit_max_number_of_layers != set_uniform_number_of_layers, "Either there should be a maximum threshold or a uniform number of layers"


        # Open image by nibabel
        ct_instance = nib.load(ct_scan).get_fdata()

        # H, W, Layers -> 512 x 512 x L
        ct_instance_shape = ct_instance.shape
        ct_instance_layer_number = ct_instance_shape[2]

        ########################################################

        ct_instance_tensor = []

        if limit_max_number_of_layers:

            # If CT has more layers than the max number of layers threshold
            if(ct_instance_layer_number > max_number_of_layers):      

                # Executed in the exact order they are specified.
                # Each image would be clipped, normalized, rotated, resized, cropped

                divider = 0
                for ct_instance_layer_index in range(max_number_of_layers):

                    divider += ct_instance_layer_number / max_number_of_layers
                    ct_instance_layer_index = int(divider) - 1
                    #print("ct_instance_layer_index", ct_instance_layer_index)

                    ct_instance_layer = ct_instance[:,:,ct_instance_layer_index]
                    #plt.imshow(ct_instance_layer, cmap='gray')
                    #plt.savefig("/home/guevenira/attention_CT/development/src/data_slice.png")
                    #plt.close()
                    
                    ct_instance_layer_clipped = np.clip(ct_instance_layer, amin, amax)
                    #plt.imshow(ct_instance_layer_clipped, cmap='gray')
                    #plt.savefig("/home/guevenira/attention_CT/development/src/data_slice_clip.png")
                    #plt.close()
                    

                    ct_instance_layer_clipped_normalized = (ct_instance_layer_clipped - (lower_bound)) / ((upper_bound) - (lower_bound))
                    #plt.imshow(ct_instance_layer_clippep_normalized, cmap='gray')
                    #plt.savefig("/home/guevenira/attention_CT/development/src/data_slice_clip_normalize.png")
                    #plt.close()


                    if "/home/guevenira/Data/shared/PDAC_CT/CTs/" in ct_scan:
                        # With PDAC
                        ct_instance_layer_clipped_normalized_rotated = np.rot90(ct_instance_layer_clipped_normalized)
                        #plt.imshow(ct_instance_layer_clipped_normalized_rotated, cmap='gray')
                        #plt.savefig("/home/guevenira/attention_CT/PDAC/debugging/x1.png")
                        #plt.close()

                    elif "/home/guevenira/Data/shared/NormalPancreas/normal_selected" in ct_scan:
                        ct_instance_layer_clipped_normalized_rotated = np.rot90(ct_instance_layer_clipped_normalized, 3)
                        ct_instance_layer_clipped_normalized_rotated = cv2.flip(ct_instance_layer_clipped_normalized_rotated, 1)
                        #plt.imshow(ct_instance_layer_clipped_normalized_rotated, cmap='gray')
                        #plt.savefig("/home/guevenira/attention_CT/PDAC/debugging/x1.png")
                        #plt.close()
                    else:
                        raise ValueError('This should not happened!!!!')
                    
                    
                    
                    ct_instance_layer_clipped_normalized_rotated_resized = cv2.resize(ct_instance_layer_clipped_normalized_rotated, dsize=(height, width), interpolation=cv2.INTER_CUBIC)

                    ct_instance_layer_clipped_normalized_rotated_resized_cropped = ct_instance_layer_clipped_normalized_rotated_resized[crop_height_begin:crop_height_end, crop_width_begin:crop_width_end]
                    #plt.imshow(ct_instance_layer_clipped_normalized_rotated_resized_cropped, cmap='gray')
                    #plt.savefig("/home/guevenira/attention_CT/PDAC/debugging/1before.png")
                    #plt.close()
                    
                    #plt.imshow(ct_instance_layer_clipped_normalized_rotated_resized_cropped, cmap='gray')
                    #plt.savefig("/home/guevenira/attention_CT/PDAC/debugging/1after.png")
                    #plt.close()
                    #exit()            
                    
                    if ct_instance_tensor == []:
                        ct_instance_tensor = torch.tensor(ct_instance_layer_clipped_normalized_rotated_resized_cropped.copy(), dtype=torch.float)
                        ct_instance_tensor = torch.unsqueeze(ct_instance_tensor, 0)
                    else:
                        ct_instance_tensor_new_layer = torch.tensor(ct_instance_layer_clipped_normalized_rotated_resized_cropped.copy(), dtype=torch.float)
                        ct_instance_tensor_new_layer = torch.unsqueeze(ct_instance_tensor_new_layer, 0)
                        ct_instance_tensor = torch.cat((ct_instance_tensor, ct_instance_tensor_new_layer), 0)
            
            # CT has layer number less than the max specified
            else:
                for ct_instance_layer_index in range(ct_instance_layer_number):

                    ct_instance_layer = ct_instance[:,:,ct_instance_layer_index]                    
                    ct_instance_layer_clipped = np.clip(ct_instance_layer, amin, amax)
                    ct_instance_layer_clipped_normalized = (ct_instance_layer_clipped - (lower_bound)) / ((upper_bound) - (lower_bound))
                    
                    if "/home/guevenira/Data/shared/PDAC_CT/CTs/" in ct_scan:
                        # With PDAC
                        ct_instance_layer_clipped_normalized_rotated = np.rot90(ct_instance_layer_clipped_normalized)
                        #plt.imshow(ct_instance_layer_clipped_normalized_rotated, cmap='gray')
                        #plt.savefig("/home/guevenira/attention_CT/PDAC/debugging/x1.png")
                        #plt.close()

                    elif "/home/guevenira/Data/shared/NormalPancreas/normal_selected" in ct_scan:
                        ct_instance_layer_clipped_normalized_rotated = np.rot90(ct_instance_layer_clipped_normalized, 3)
                        ct_instance_layer_clipped_normalized_rotated = cv2.flip(ct_instance_layer_clipped_normalized_rotated, 1)
                        #plt.imshow(ct_instance_layer_clipped_normalized_rotated, cmap='gray')
                        #plt.savefig("/home/guevenira/attention_CT/PDAC/debugging/x1.png")
                        #plt.close()
                    else:
                        raise ValueError('This should not happened!!!!')
                    
                    ct_instance_layer_clipped_normalized_rotated_resized = cv2.resize(ct_instance_layer_clipped_normalized_rotated, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
                    ct_instance_layer_clipped_normalized_rotated_resized_cropped = ct_instance_layer_clipped_normalized_rotated_resized[crop_height_begin:crop_height_end, crop_width_begin:crop_width_end]

                    if ct_instance_tensor == []:
                        ct_instance_tensor = torch.tensor(ct_instance_layer_clipped_normalized_rotated_resized_cropped.copy(), dtype=torch.float)
                        ct_instance_tensor = torch.unsqueeze(ct_instance_tensor, 0)
                    else:
                        ct_instance_tensor_new_layer = torch.tensor(ct_instance_layer_clipped_normalized_rotated_resized_cropped.copy(), dtype=torch.float)
                        ct_instance_tensor_new_layer = torch.unsqueeze(ct_instance_tensor_new_layer, 0)
                        ct_instance_tensor = torch.cat((ct_instance_tensor, ct_instance_tensor_new_layer), 0)

            ct_instance_tensor = torch.unsqueeze(ct_instance_tensor, 0)            
            return ct_instance_tensor, torch.tensor(ct_label)