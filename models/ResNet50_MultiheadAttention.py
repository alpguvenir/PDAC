import torch.nn as nn
from torchvision import models

class ResNet50_MultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        get_params = lambda m: sum(p.numel() for p in m.parameters())

        hidden_size1 = 2048
        
        feature_extractor = models.resnet.resnet50(pretrained=True)
        
        # Fully connected layer returning (hidden_size1) 256 units
        feature_extractor.fc =  (
                                    nn.Linear(hidden_size1, hidden_size1) if hidden_size1 != 512 else nn.Identity()
                                )
        
        # To convert a input 1 channel image to output 3 channel image
        feature_extractor.conv1 = nn.Sequential( nn.Conv2d(1, 3, 1), feature_extractor.conv1 )

        self.feature_extractor = feature_extractor
        
        # Number of heads 8, embedding dimension 256
        self.att = nn.MultiheadAttention(hidden_size1, 16)
        
        # Classifier returning with only 1 unit, binary
        self.classifier = nn.Linear(hidden_size1, 1)

        print(f"Feature extractor has {get_params(self.feature_extractor)} params")
        print(f"Attention has {get_params(self.att)} params")
        print(f"Classifier has {get_params(self.classifier)} params")

    def forward(self, x):
        features = self.feature_extractor(x).unsqueeze(
            1
        )  # assuming only 1 CT at a time

        #print("features.shape", features.shape)

        query = features.mean(0, keepdims=True)

        #print("query.shape", query.shape)
        
        features, att_map = self.att(query, features, features)

        #print("features.shape", features.shape)

        #print("att_map.shape", att_map.shape)

        out = self.classifier(features.squeeze(0))
        # print(out.shape)
        return out