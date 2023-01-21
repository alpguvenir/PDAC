import torch
import torch.nn as nn
from torchvision import models

class ResNet50_sum(nn.Module):
    def __init__(self):
        super().__init__()

        get_params = lambda m: sum(p.numel() for p in m.parameters())

        hidden_size1 = 1
        
        feature_extractor = models.resnet.resnet50(pretrained=True)
        
        # Fully connected layer returning (hidden_size1) 256 units
        # fc contains 1000 nodes at the end, so override it to keep the same number of nodes as in_features = 2048
        
        feature_extractor.fc =  (
                                    nn.Linear(2048, hidden_size1) if hidden_size1 != 512 else nn.Identity()
                                )

        # Convert a input 1 channel image to output 3 channel image -> Input [151, 1, 256, 256]
        # ResNet conv1 is normally Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # So we convert it to 
        # (0): Conv2d(1, 3, kernel_size=(1, 1), stride=(1, 1))
        # (1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        feature_extractor.conv1 = nn.Sequential( nn.Conv2d(1, 3, 1), feature_extractor.conv1 )

        self.feature_extractor = feature_extractor
        

    def forward(self, x):

        # [batch_size, channels, height, width]
        
        features = self.feature_extractor(x).unsqueeze(
            1
        )  # assuming only 1 CT at a time

        #print(features.shape)
        features = torch.sum(features, 0)
        return features