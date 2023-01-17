import torch
import torch.nn as nn
from torchvision import models

class ResNet18_3D(nn.Module):

    def __init__(self):
        super().__init__()

        # https://pytorch.org/vision/main/models/generated/torchvision.models.video.r3d_18.html
        # https://github.com/pytorch/vision/issues/4083
        # img = torch.randn(1, 3, 192, 112, 112).to(torch.float)

        get_params = lambda m: sum(p.numel() for p in m.parameters())


        feature_extractor = models.video.r3d_18(pretrained=True)
        
        feature_extractor.stem = nn.Sequential(
            nn.Conv3d(1, 3, 1), feature_extractor.stem
        )
        
        hidden_size1 = 256
        feature_extractor.fc = (
            nn.Linear(512, hidden_size1) if hidden_size1 != 512 else nn.Identity()
        )

        self.feature_extractor = feature_extractor
        self.classifier = nn.Linear(hidden_size1, 1)
        
        print(f"Feature extractor has {get_params(self.feature_extractor)} params")
        print(f"Classifier has {get_params(self.classifier)} params")

    def forward(self, x):

        #print(self.feature_extractor.stem(x).shape)

        features = self.feature_extractor(x).unsqueeze(
            1
        )  # assuming only 1 CT at a time

        """        
        query = features.mean(0, keepdims=True)
        # print(features.shape)
        # print(query.shape)
        
        features, att_map = self.att(query, features, features)
        """
        
        out = self.classifier(features.squeeze(0))
        # print(out.shape)
        

        return out