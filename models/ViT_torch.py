import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights
from torchvision import models

# https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py

class ViViT_torch(nn.Module):
    def __init__(self):
        super().__init__()

        hidden_size1 = 1

        weights = ViT_B_16_Weights.DEFAULT
        feature_extractor = models.vit_b_16(weights=weights)   


        self.feature_extractor = feature_extractor
        

    def forward(self, x):
        features = self.feature_extractor(x).unsqueeze(
            1
        )  # assuming only 1 CT at a time
        
        print(features.shape)

        return features
