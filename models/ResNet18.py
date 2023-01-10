import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights
from torchvision import models

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        hidden_size1 = 1

        weights = ResNet18_Weights.DEFAULT
        feature_extractor = models.resnet.resnet18(weights=weights)   

        # Fully connected layer returning (hidden_size1) 256 units
        feature_extractor.fc =  (
                                    nn.Linear(512, hidden_size1)
                                )
        
        # To convert a input 1 channel image to output 3 channel image
        feature_extractor.conv1 = nn.Sequential ( 
                                                    nn.Conv2d(1, 3, 1), feature_extractor.conv1 
                                                )

        self.feature_extractor = feature_extractor
        

    def forward(self, x):
        features = self.feature_extractor(x).unsqueeze(
            1
        )  # assuming only 1 CT at a time
        
        #print(features.shape)
        features = torch.mean(features, 0)
        return features
