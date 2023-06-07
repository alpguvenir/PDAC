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

import torch.nn as nn
from torchvision import models
from functools import partial

class ViT_b16_MultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        get_params = lambda m: sum(p.numel() for p in m.parameters())

        hidden_size1 = 256
        
        feature_extractor = models.vit_b_16(weights=models.ViT_B_16_Weights) 

        feature_extractor.conv_proj = nn.Sequential( nn.Conv2d(1, 3, 1), feature_extractor.conv_proj )
        feature_extractor.heads = nn.Linear(768, hidden_size1)

        self.feature_extractor = feature_extractor
        
        # Number of heads 8, embedding dimension 256
        self.att = nn.MultiheadAttention(hidden_size1, 8)
        
        # Classifier returning with only 1 unit, binary
        self.classifier = nn.Linear(hidden_size1, 1)

        print(f"Feature extractor has {get_params(self.feature_extractor)} params")
        print(f"Attention has {get_params(self.att)} params")
        print(f"Classifier has {get_params(self.classifier)} params")


    def forward(self, x):

        # [batch_size, channels, height, width]        
        features = self.feature_extractor(x).unsqueeze(
            1
        )  # assuming only 1 CT at a time
        
        #print("features.shape", features.shape)

        query = features.mean(0, keepdims=True)

        #print("query.shape", query.shape)
        
        features, att_map = self.att(query, features, features)

        #print("features.shape", features.shape)

        #print("att_map.shape", att_map.shape)
        #print(torch.sum(att_map))

        out = self.classifier(features.squeeze(0))
        # print(out.shape)
        
        return out, att_map
        #return out