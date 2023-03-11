import torch
import torch.nn as nn
from torchvision import models


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResNet152_inter_layer_MultiheadAttention_v2(nn.Module):

    def __init__(self):
        super().__init__()

        get_params = lambda m: sum(p.numel() for p in m.parameters())

        hidden_size1 = 2048

        feature_extractor = models.resnet.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        #feature_extractor = models.resnet.resnet152(weights=None)

        feature_extractor.fc = (
            nn.Linear(hidden_size1, hidden_size1) if hidden_size1 != 512 else nn.Identity()
        )

        feature_extractor.conv1 = nn.Sequential(
            nn.Conv2d(1, 3, 1), feature_extractor.conv1
        )
        
        self.feature_extractor = feature_extractor

        # CBAM after layer 1
        self.ca1 = ChannelAttention(256)
        self.sa1 = SpatialAttention()

        # CBAM after layer 2
        #self.ca2 = ChannelAttention(512)
        #self.sa2 = SpatialAttention()


        self.att = nn.MultiheadAttention(hidden_size1, 16)
        self.classifier = nn.Linear(hidden_size1, 1)

        print(f"Feature extractor has {get_params(self.feature_extractor)} params")
        print(f"Attention has {get_params(self.att)} params")
        print(f"Classifier has {get_params(self.classifier)} params")

    def forward(self, x):
        x = self.feature_extractor.conv1(x)     # conv1 torch.Size([110, 64, 112, 112])        
        x = self.feature_extractor.bn1(x)       # bn1 torch.Size([110, 64, 112, 112])
        x = self.feature_extractor.relu(x)      # relu torch.Size([110, 64, 112, 112])
        x = self.feature_extractor.maxpool(x)   # maxpool torch.Size([110, 64, 56, 56])

        x = self.feature_extractor.layer1(x)    # layer1 torch.Size([114, 64, 64, 64])

        # CBAM after layer 1
        x = self.ca1(x) * x
        x = self.sa1(x) * x

        x = self.feature_extractor.layer2(x)    # layer2 torch.Size([114, 128, 32, 32])

        # CBAM after layer 2
        #x = self.ca2(x) * x
        #x = self.sa2(x) * x

        x = self.feature_extractor.layer3(x)    # layer3 torch.Size([114, 256, 16, 16])
        x = self.feature_extractor.layer4(x)    # layer4 torch.Size([114, 512, 8, 8])

        x = self.feature_extractor.avgpool(x)   # avgpool torch.Size([114, 512, 1, 1])
        x = torch.flatten(x, 1)                 # flatten torch.Size([114, 512])
        x = self.feature_extractor.fc(x)        # fc torch.Size([114, 256])

        features = (x).unsqueeze(
            1
        ) # assuming only 1 CT at a time

    
        query = features.mean(0, keepdims=True)
        # print(features.shape)
        # print(query.shape)
        
        features, att_map = self.att(query, features, features)
        out = self.classifier(features.squeeze(0))
        # print(out.shape)
        
        #return out, att_map
        return out