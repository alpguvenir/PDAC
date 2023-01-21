import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights
from torchvision import models

class ResNet18_Linear(nn.Module):
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

        # Change here according to the number of layers
        self.classifier = nn.Linear(200, 1)

        self.feature_extractor = feature_extractor
        

    def forward(self, x):
        features = self.feature_extractor(x).unsqueeze(
            1
        )  # assuming only 1 CT at a time

        features = features.permute(2, 1, 0)
        out = self.classifier(features.squeeze(0))

        return out