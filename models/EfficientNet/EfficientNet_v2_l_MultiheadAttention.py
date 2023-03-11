import torch
import torch.nn as nn
from torchvision import models

class EfficientNet_v2_l_MultiheadAttention(nn.Module):

    def __init__(self):
        super().__init__()

        get_params = lambda m: sum(p.numel() for p in m.parameters())

        hidden_size1 = 1024

        feature_extractor = models.efficientnet.efficientnet_v2_l(weights=models.efficientnet.EfficientNet_V2_L_Weights)

        feature_extractor.features[0][0] = nn.Sequential( nn.Conv2d(1,3,1), feature_extractor.features[0][0])

        feature_extractor.classifier[1] = nn.Linear(1280, hidden_size1)
        
        self.feature_extractor = feature_extractor
        self.att = nn.MultiheadAttention(hidden_size1, 16)
        self.classifier = nn.Linear(hidden_size1, 1)

        print(f"Feature extractor has {get_params(self.feature_extractor)} params")
        print(f"Attention has {get_params(self.att)} params")
        print(f"Classifier has {get_params(self.classifier)} params")

    def forward(self, x):
        features = self.feature_extractor(x).unsqueeze(
            1
        )  # assuming only 1 CT at a time

        query = features.mean(0, keepdims=True)
        # print(features.shape)
        # print(query.shape)
        
        features, att_map = self.att(query, features, features)
        out = self.classifier(features.squeeze(0))
        # print(out.shape)
        
        #return out, att_map
        return out