import torch
import torch.nn as nn
from torchvision import models

class ResNet18_rad_v2(nn.Module):

    def __init__(self):
        super().__init__()

        get_params = lambda m: sum(p.numel() for p in m.parameters())

        hidden_size1 = 28

        #feature_extractor = models.resnet.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feature_extractor = models.resnet.resnet18(weights=None)

        feature_extractor.fc = (
            nn.Linear(512, hidden_size1) if hidden_size1 != 512 else nn.Identity()
        )

        feature_extractor.conv1 = nn.Sequential(
            nn.Conv2d(1, 3, 1), feature_extractor.conv1
        )

        self.feature_extractor = feature_extractor

        print(f"Feature extractor has {get_params(self.feature_extractor)} params")

    def forward(self, x):
        features = self.feature_extractor(x)

        # dont need it for crossentropyloss
        # The input is expected to contain the unnormalized logits for each class
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
        #features = nn.functional.softmax(features, dim=1)

        return features