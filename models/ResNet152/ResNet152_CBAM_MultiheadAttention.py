import torch
import torch.nn as nn
from torchvision import models
from functools import partial

class ResNet152_CBAM_MultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        get_params = lambda m: sum(p.numel() for p in m.parameters())

        hidden_size1 = 2048
        
        feature_extractor = models.resnet.resnet152(weights=models.ResNet152_Weights.DEFAULT)

        # Fully connected layer returning (hidden_size1) 256 units
        # fc contains 1000 nodes at the end, so override it to keep the same number of nodes as in_features = 2048
        
        feature_extractor.fc =  (
                                    nn.Linear(hidden_size1, hidden_size1) if hidden_size1 != 512 else nn.Identity()
                                )

        
        """
        feature_extractor.fc = nn.Sequential(
                                    nn.Dropout(0.2),
                                    nn.Linear(hidden_size1, hidden_size1)
                                )
        """

        # Convert a input 1 channel image to output 3 channel image -> Input [151, 1, 256, 256]
        # ResNet conv1 is normally Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # So we convert it to 
        # (0): Conv2d(1, 3, kernel_size=(1, 1), stride=(1, 1))
        # (1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
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

        # [batch_size, channels, height, width]
        
        """
        features = self.feature_extractor(x).unsqueeze(
            1
        )  # assuming only 1 CT at a time
        print("features", features.shape)
        """

        """
        conv1
        bn1
        relu
        maxpool
        layer1
        layer2
        layer3
        layer4
        avgpool
        fc
        """
        # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
        # Check _forward_impl()
        
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.relu(x)        
        x = self.feature_extractor.maxpool(x)

        x = self.feature_extractor.layer1(x)
        x = self.feature_extractor.layer2(x)
        x = self.feature_extractor.layer3(x)
        x = self.feature_extractor.layer4(x)

        x = self.feature_extractor.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.feature_extractor.fc(x)
        features = (x).unsqueeze(1)

        #print("features.shape", features.shape)

        query = features.mean(0, keepdims=True)

        #print("query.shape", query.shape)
        
        features, att_map = self.att(query, features, features)

        #print("features.shape", features.shape)

        #print("att_map.shape", att_map.shape)
        #print(torch.sum(att_map))

        out = self.classifier(features.squeeze(0))
        # print(out.shape)
        return out