import torch.nn as nn
from torchvision import models
from functools import partial

class ResNet152_MultiheadAttention_v3(nn.Module):
    def __init__(self):
        super().__init__()

        get_params = lambda m: sum(p.numel() for p in m.parameters())

        hidden_size1 = 1000
        
        feature_extractor = models.resnet.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        #feature_extractor = models.resnet.resnet152(weights=None)

        # Convert a input 1 channel image to output 3 channel image -> Input [151, 1, 256, 256]
        # ResNet conv1 is normally Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # So we convert it to 
        # (0): Conv2d(1, 3, kernel_size=(1, 1), stride=(1, 1))
        # (1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        feature_extractor.conv1 = nn.Sequential( nn.Conv2d(1, 3, 1), feature_extractor.conv1 )

        self.feature_extractor = feature_extractor
        
        # Number of heads 8, embedding dimension 256
        self.att = nn.MultiheadAttention(hidden_size1, 10)
        
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
        #[batch size, 1, 2048]
        
        query = features.mean(0, keepdims=True)
        #print(features.shape)                           # [110, 1, 2048
        #print(query.shape)                              # [1, 1, 2048]
        
        #print("query.shape", query.shape)
        
        features, att_map = self.att(query, features, features)

        #print("features.shape", features.shape)

        #print("att_map.shape", att_map.shape)
        #print(torch.sum(att_map))

        out = self.classifier(features.squeeze(0))
        # print(out.shape)
        return out