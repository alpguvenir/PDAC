import torch.nn as nn
from torchvision import models
from functools import partial

class ResNet152_linearquery_MultiheadAttention(nn.Module):
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

        #self.att_linear = nn.Linear(110, 1)

        
        # Number of heads 8, embedding dimension 256
        self.att = nn.MultiheadAttention(hidden_size1, 16)
        
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
        
        """Option 1 ORIGINAL"""
        query = features.mean(0, keepdims=True)
        #print(features.shape)                           # [110, 1, 2048
        #print(query.shape)                              # [1, 1, 2048]
        """Option 1"""


        """Option 2"""
        #temp_features = features.squeeze(1)              # [110, 2048]
        #temp_features = temp_features.permute(1, 0)      # [2048, 110]

        #query = self.att_linear(temp_features)           # [2048, 1]
        #query = query.permute(1, 0)                      # [1, 2048]
        #query = query.unsqueeze(0)                       # [1, 1, 2048]
        """Option 2"""
        
        #print("query.shape", query.shape)
        
        features, att_map = self.att(query, features, features)

        #print("features.shape", features.shape)

        #print("att_map.shape", att_map.shape)
        #print(torch.sum(att_map))

        out = self.classifier(features.squeeze(0))
        # print(out.shape)
        return out