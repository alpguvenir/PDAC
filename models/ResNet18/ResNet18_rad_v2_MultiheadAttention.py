import torch
import torch.nn as nn
from torchvision import models

from models.ResNet18.ResNet18_rad_v2 import ResNet18_rad_v2

class ResNet18_rad_v2_MultiheadAttention(nn.Module):

    def __init__(self):
        super().__init__()

        get_params = lambda m: sum(p.numel() for p in m.parameters())

        hidden_size1 = 256

        #feature_extractor = models.resnet.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        #feature_extractor = models.resnet.resnet18(weights=None)
        
        feature_extractor = ResNet18_rad_v2()        
        #feature_extractor.load_state_dict(torch.load('/home/guevenira/attention_CT/PDAC/results-train/ResNet18/radimagenet-1677429635/model24.pth'))
        #feature_extractor.load_state_dict(torch.load('/home/guevenira/attention_CT/PDAC/results-train/ResNet18/radimagenet-1677456606/model16.pth'))

        feature_extractor.feature_extractor.fc = (
            nn.Linear(512, hidden_size1) if hidden_size1 != 512 else nn.Identity()
        )

        self.feature_extractor = feature_extractor
        self.att = nn.MultiheadAttention(hidden_size1, 8)
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