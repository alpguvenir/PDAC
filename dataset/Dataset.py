import json
import os
import yaml
import random

import torch
import torchvision.transforms.functional as F
import numpy as np
from matplotlib import pyplot as plt

import nibabel as nib
from PIL import Image, ImageEnhance
from skimage import color
from skimage import io
import cv2 

from typing import Any, Type

class Dataset(torch.utils.data.Dataset):

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


    def augmentation(self,  ct_instance_layer_clipped_normalized_rotated_resized_cropped,
                            horizontal_flip_prob, horizontal_flip_threshold, 
                            translation_prob, translation_threshold, M, 
                            gaussian_prob, gaussian_threshold, noise, amin, amax, 
                            brightness_contrast_sharpness_prob, brightness_contrast_sharpness_threshold):

        if self.train_mode and self.params_dict.get("data.augmentation"):

            if horizontal_flip_prob > horizontal_flip_threshold:
                ct_instance_layer_clipped_normalized_rotated_resized_cropped = cv2.flip(ct_instance_layer_clipped_normalized_rotated_resized_cropped, 1)
            
            if translation_prob > translation_threshold:
                print("translation")
                ct_instance_layer_clipped_normalized_rotated_resized_cropped = cv2.warpPerspective(ct_instance_layer_clipped_normalized_rotated_resized_cropped, M, 
                                                                                                (ct_instance_layer_clipped_normalized_rotated_resized_cropped.shape[0], ct_instance_layer_clipped_normalized_rotated_resized_cropped.shape[1]))

            # Dont do both, too much augmentation
            if gaussian_prob > gaussian_threshold:
                print("gaussiam")
                ct_instance_layer_clipped_normalized_rotated_resized_cropped += noise
                ct_instance_layer_clipped_normalized_rotated_resized_cropped = np.clip(ct_instance_layer_clipped_normalized_rotated_resized_cropped, amin, amax)
                
            elif brightness_contrast_sharpness_prob > brightness_contrast_sharpness_threshold:
                print("brightness")
                # B 0.9, C 1.3
                ct_instance_layer_clipped_normalized_rotated_resized_cropped = torch.from_numpy(ct_instance_layer_clipped_normalized_rotated_resized_cropped).unsqueeze(0)
                ct_instance_layer_clipped_normalized_rotated_resized_cropped = F.adjust_brightness(ct_instance_layer_clipped_normalized_rotated_resized_cropped, 0.9)
                ct_instance_layer_clipped_normalized_rotated_resized_cropped = F.adjust_contrast(ct_instance_layer_clipped_normalized_rotated_resized_cropped, 1.3)
                ct_instance_layer_clipped_normalized_rotated_resized_cropped = F.adjust_sharpness(ct_instance_layer_clipped_normalized_rotated_resized_cropped, 1.5)
                ct_instance_layer_clipped_normalized_rotated_resized_cropped = ct_instance_layer_clipped_normalized_rotated_resized_cropped.squeeze(0)

                ct_instance_layer_clipped_normalized_rotated_resized_cropped = ct_instance_layer_clipped_normalized_rotated_resized_cropped.detach().numpy()
        
        return ct_instance_layer_clipped_normalized_rotated_resized_cropped


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ct_scan, ct_label = (
            self.ct_scans[index],
            self.ct_labels[index]   
        )

        amin = self.transforms['Clip']['amin']
        amax = self.transforms['Clip']['amax']

        lower_bound = self.transforms['Normalize']['bounds'][0]
        upper_bound = self.transforms['Normalize']['bounds'][1]
        
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


        horizontal_flip_prob = random.random()
        horizontal_flip_threshold = self.params_dict.get("data.augmentation.horizontal_flip_threshold")

        translation_prob = random.random()
        translation_threshold = self.params_dict.get("data.augmentation.translation_threshold")

        gaussian_prob = random.random()
        gaussian_threshold= self.params_dict.get("data.augmentation.gaussian_threshold")

        brightness_contrast_sharpness_prob = random.random()
        brightness_contrast_sharpness_threshold = self.params_dict.get("data.augmentation.brightness_contrast_sharpness_threshold")

        # Transformation matrix for translation
        horizontal_shift = random.randint(-5, 5)
        vertical_shift = random.randint(-5, 5)

        M = np.float32([[1, 0, horizontal_shift],     # Horizontal shift, - to left and + to right
                        [0, 1, vertical_shift],   # Vertical shift, + to bottom and - to top
                        [0, 0, 1]])

        mean = 0
        variance = 0.05
        noise = np.random.normal(mean, variance, [crop_height_end - crop_height_begin, crop_width_end - crop_width_begin]) 

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
                    
                    ct_instance_layer_clipped_normalized_rotated = np.rot90(ct_instance_layer_clipped_normalized)
                    #plt.imshow(ct_instance_layer_clipped_normalized_rotated, cmap='gray')
                    #plt.savefig("/home/guevenira/attention_CT/PDAC/debugging/x1.png")
                    #plt.close()
                    
                    ct_instance_layer_clipped_normalized_rotated_resized = cv2.resize(ct_instance_layer_clipped_normalized_rotated, dsize=(height, width), interpolation=cv2.INTER_CUBIC)

                    ct_instance_layer_clipped_normalized_rotated_resized_cropped = ct_instance_layer_clipped_normalized_rotated_resized[crop_height_begin:crop_height_end, crop_width_begin:crop_width_end]

                    
                    ct_instance_layer_clipped_normalized_rotated_resized_cropped = self.augmentation(       ct_instance_layer_clipped_normalized_rotated_resized_cropped,
                                                                                                            horizontal_flip_prob, horizontal_flip_threshold, 
                                                                                                            translation_prob, translation_threshold, M, 
                                                                                                            gaussian_prob, gaussian_threshold, noise, amin, amax, 
                                                                                                            brightness_contrast_sharpness_prob, brightness_contrast_sharpness_threshold)
                    
                    
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
                    ct_instance_layer_clipped_normalized_rotated = np.rot90(ct_instance_layer_clipped_normalized)
                    ct_instance_layer_clipped_normalized_rotated_resized = cv2.resize(ct_instance_layer_clipped_normalized_rotated, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
                    ct_instance_layer_clipped_normalized_rotated_resized_cropped = ct_instance_layer_clipped_normalized_rotated_resized[crop_height_begin:crop_height_end, crop_width_begin:crop_width_end]


                    ct_instance_layer_clipped_normalized_rotated_resized_cropped = self.augmentation(       ct_instance_layer_clipped_normalized_rotated_resized_cropped,
                                                                                                            horizontal_flip_prob, horizontal_flip_threshold, 
                                                                                                            translation_prob, translation_threshold, M, 
                                                                                                            gaussian_prob, gaussian_threshold, noise, amin, amax, 
                                                                                                            brightness_contrast_sharpness_prob, brightness_contrast_sharpness_threshold)


                    if ct_instance_tensor == []:
                        ct_instance_tensor = torch.tensor(ct_instance_layer_clipped_normalized_rotated_resized_cropped.copy(), dtype=torch.float)
                        ct_instance_tensor = torch.unsqueeze(ct_instance_tensor, 0)
                    else:
                        ct_instance_tensor_new_layer = torch.tensor(ct_instance_layer_clipped_normalized_rotated_resized_cropped.copy(), dtype=torch.float)
                        ct_instance_tensor_new_layer = torch.unsqueeze(ct_instance_tensor_new_layer, 0)
                        ct_instance_tensor = torch.cat((ct_instance_tensor, ct_instance_tensor_new_layer), 0)

            ct_instance_tensor = torch.unsqueeze(ct_instance_tensor, 0)            
            return ct_instance_tensor, torch.tensor(ct_label)


        # Setting all CTs to same number of layers
        elif set_uniform_number_of_layers:

            divider = 0
            for ct_instance_layer_index in range(uniform_number_of_layers):
                
                divider += ct_instance_layer_number / uniform_number_of_layers
                ct_instance_layer_index = int(divider) - 1

                ct_instance_layer = ct_instance[:,:,ct_instance_layer_index]
                ct_instance_layer_clipped = np.clip(ct_instance_layer, amin, amax)                    
                ct_instance_layer_clipped_normalized = (ct_instance_layer_clipped - (lower_bound)) / ((upper_bound) - (lower_bound))
                ct_instance_layer_clipped_normalized_rotated = np.rot90(ct_instance_layer_clipped_normalized)
                ct_instance_layer_clipped_normalized_rotated_resized = cv2.resize(ct_instance_layer_clipped_normalized_rotated, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
                ct_instance_layer_clipped_normalized_rotated_resized_cropped = ct_instance_layer_clipped_normalized_rotated_resized[crop_height_begin:crop_height_end, crop_width_begin:crop_width_end]


                ct_instance_layer_clipped_normalized_rotated_resized_cropped = self.augmentation(       ct_instance_layer_clipped_normalized_rotated_resized_cropped,
                                                                                                        horizontal_flip_prob, horizontal_flip_threshold, 
                                                                                                        translation_prob, translation_threshold, M, 
                                                                                                        gaussian_prob, gaussian_threshold, noise, amin, amax, 
                                                                                                        brightness_contrast_sharpness_prob, brightness_contrast_sharpness_threshold)


                if ct_instance_tensor == []:
                    ct_instance_tensor = torch.tensor(ct_instance_layer_clipped_normalized_rotated_resized_cropped.copy(), dtype=torch.float)
                    ct_instance_tensor = torch.unsqueeze(ct_instance_tensor, 0)
                else:
                    ct_instance_tensor_new_layer = torch.tensor(ct_instance_layer_clipped_normalized_rotated_resized_cropped.copy(), dtype=torch.float)
                    ct_instance_tensor_new_layer = torch.unsqueeze(ct_instance_tensor_new_layer, 0)
                    ct_instance_tensor = torch.cat((ct_instance_tensor, ct_instance_tensor_new_layer), 0)
        
            ct_instance_tensor = torch.unsqueeze(ct_instance_tensor, 0)
            return ct_instance_tensor, torch.tensor(ct_label)

        elif zero_pad_number_of_layers:

            if(ct_instance_layer_number > zeropad_number_of_layers):      
                divider = 0
                for ct_instance_layer_index in range(max_number_of_layers):

                    divider += ct_instance_layer_number / max_number_of_layers
                    ct_instance_layer_index = int(divider) - 1
                    ct_instance_layer = ct_instance[:,:,ct_instance_layer_index]
                    ct_instance_layer_clipped = np.clip(ct_instance_layer, amin, amax)
                    ct_instance_layer_clipped_normalized = (ct_instance_layer_clipped - (lower_bound)) / ((upper_bound) - (lower_bound))
                    ct_instance_layer_clipped_normalized_rotated = np.rot90(ct_instance_layer_clipped_normalized)
                    ct_instance_layer_clipped_normalized_rotated_resized = cv2.resize(ct_instance_layer_clipped_normalized_rotated, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
                    ct_instance_layer_clipped_normalized_rotated_resized_cropped = ct_instance_layer_clipped_normalized_rotated_resized[crop_height_begin:crop_height_end, crop_width_begin:crop_width_end]

                    
                    ct_instance_layer_clipped_normalized_rotated_resized_cropped = self.augmentation(       ct_instance_layer_clipped_normalized_rotated_resized_cropped,
                                                                                                            horizontal_flip_prob, horizontal_flip_threshold, 
                                                                                                            translation_prob, translation_threshold, M, 
                                                                                                            gaussian_prob, gaussian_threshold, noise, amin, amax, 
                                                                                                            brightness_contrast_sharpness_prob, brightness_contrast_sharpness_threshold)
                    

                    if ct_instance_tensor == []:
                        ct_instance_tensor = torch.tensor(ct_instance_layer_clipped_normalized_rotated_resized_cropped.copy(), dtype=torch.float)
                        ct_instance_tensor = torch.unsqueeze(ct_instance_tensor, 0)
                    else:
                        ct_instance_tensor_new_layer = torch.tensor(ct_instance_layer_clipped_normalized_rotated_resized_cropped.copy(), dtype=torch.float)
                        ct_instance_tensor_new_layer = torch.unsqueeze(ct_instance_tensor_new_layer, 0)
                        ct_instance_tensor = torch.cat((ct_instance_tensor, ct_instance_tensor_new_layer), 0)
            else:
                for ct_instance_layer_index in range(ct_instance_layer_number):

                    ct_instance_layer = ct_instance[:,:,ct_instance_layer_index]                    
                    ct_instance_layer_clipped = np.clip(ct_instance_layer, amin, amax)
                    ct_instance_layer_clipped_normalized = (ct_instance_layer_clipped - (lower_bound)) / ((upper_bound) - (lower_bound))
                    ct_instance_layer_clipped_normalized_rotated = np.rot90(ct_instance_layer_clipped_normalized)
                    ct_instance_layer_clipped_normalized_rotated_resized = cv2.resize(ct_instance_layer_clipped_normalized_rotated, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
                    ct_instance_layer_clipped_normalized_rotated_resized_cropped = ct_instance_layer_clipped_normalized_rotated_resized[crop_height_begin:crop_height_end, crop_width_begin:crop_width_end]

                    
                    ct_instance_layer_clipped_normalized_rotated_resized_cropped = self.augmentation(       ct_instance_layer_clipped_normalized_rotated_resized_cropped,
                                                                                                            horizontal_flip_prob, horizontal_flip_threshold, 
                                                                                                            translation_prob, translation_threshold, M, 
                                                                                                            gaussian_prob, gaussian_threshold, noise, amin, amax, 
                                                                                                            brightness_contrast_sharpness_prob, brightness_contrast_sharpness_threshold)
                    

                    if ct_instance_tensor == []:
                        ct_instance_tensor = torch.tensor(ct_instance_layer_clipped_normalized_rotated_resized_cropped.copy(), dtype=torch.float)
                        ct_instance_tensor = torch.unsqueeze(ct_instance_tensor, 0)
                    else:
                        ct_instance_tensor_new_layer = torch.tensor(ct_instance_layer_clipped_normalized_rotated_resized_cropped.copy(), dtype=torch.float)
                        ct_instance_tensor_new_layer = torch.unsqueeze(ct_instance_tensor_new_layer, 0)
                        ct_instance_tensor = torch.cat((ct_instance_tensor, ct_instance_tensor_new_layer), 0)

                # Calculate how many layers missing
                pad = torch.zeros(zeropad_number_of_layers - ct_instance_layer_number, crop_height_end - crop_height_begin, crop_width_end - crop_width_begin)
                ct_instance_tensor = torch.cat((pad, ct_instance_tensor), 0)
                
            ct_instance_tensor = torch.unsqueeze(ct_instance_tensor, 0)            
            return ct_instance_tensor, torch.tensor(ct_label)