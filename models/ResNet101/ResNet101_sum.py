import torch
import torch.nn as nn
from functools import partial
from torchvision import models

class ResNet101_sum(nn.Module):
    def __init__(self):
        super().__init__()

        get_params = lambda m: sum(p.numel() for p in m.parameters())

        hidden_size1 = 1
        
        #feature_extractor = models.resnet.resnet18(weights=models.ResNet18_Weights)
        feature_extractor = models.resnet.resnet18(norm_layer=partial(nn.BatchNorm2d, track_running_stats=False))

        # Fully connected layer returning (hidden_size1) 256 units
        # fc contains 1000 nodes at the end, so override it to keep the same number of nodes as in_features = 2048
        
        feature_extractor.fc =  (
                                    nn.Linear(512, hidden_size1) if hidden_size1 != 512 else nn.Identity()
                                )

        # Convert a input 1 channel image to output 3 channel image -> Input [151, 1, 256, 256]
        # ResNet conv1 is normally Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # So we convert it to 
        # (0): Conv2d(1, 3, kernel_size=(1, 1), stride=(1, 1))
        # (1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        feature_extractor.conv1 = nn.Sequential( nn.Conv2d(1, 3, 1), feature_extractor.conv1 )

        self.feature_extractor = feature_extractor

        """
        for name, module in feature_extractor.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                #module = nn.BatchNorm2d(module.num_features, module.eps, module.momentum, module.affine, False)
                
                module.__init__(module.num_features, module.eps,
                                            module.momentum, module.affine,
                                            track_running_stats=False)
                #module.track_running_stats = False

        #print(feature_extractor)
        #exit()
        """

        """
        feature_extractor.bn1 = (
            nn.GroupNorm(1, 64, 0.00001, True)
        )
        print(feature_extractor)
        """


    def forward(self, x):

        # [batch_size, channels, height, width]
        
        features = self.feature_extractor(x).unsqueeze(
            1
        )  # assuming only 1 CT at a time

        #print(features.shape)
        features = torch.sum(features, 0)
        return features