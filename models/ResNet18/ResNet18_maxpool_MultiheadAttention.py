
import torch
import torch.nn as nn
from torchvision import models
from functools import partial

class MaxpoolSpatialAttention(nn.Module):
    def __init__(self):
        super(MaxpoolSpatialAttention, self).__init__()

        # self.conv1 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv1 = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1 = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResNet18_maxpool_MultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        get_params = lambda m: sum(p.numel() for p in m.parameters())

        hidden_size1 = 256
        
        feature_extractor = models.resnet.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Convert a input 1 channel image to output 3 channel image -> Input [151, 1, 256, 256]
        # ResNet conv1 is normally Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # So we convert it to 
        # (0): Conv2d(1, 3, kernel_size=(1, 1), stride=(1, 1))
        # (1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        feature_extractor.conv1 = nn.Sequential( nn.Conv2d(1, 3, 1), feature_extractor.conv1 )


        # Fully connected layer returning (hidden_size1) 256 units
        # fc contains 1000 nodes at the end, so override it to keep the same number of nodes as in_features = 2048
        feature_extractor.fc =  (
                                    nn.Linear(512, hidden_size1) if hidden_size1 != 512 else nn.Identity()
                                )


        self.feature_extractor = feature_extractor


        #self.conv2 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.maxpool_spatial = MaxpoolSpatialAttention()

        # Number of heads 16, embedding dimension 3136
        #self.kernel_att_after_maxpool = nn.MultiheadAttention(3136, 56)
        #self.kernel_att_after_layer1 = nn.MultiheadAttention(4096, 64)
        #self.kernel_att_after_layer2 = nn.MultiheadAttention(784, 28)
        #self.kernel_att_after_layer3 = nn.MultiheadAttention(196, 14)

        
        #self.att_linear = nn.Linear(110, 1)

        # Number of heads 8, embedding dimension 256
        self.att = nn.MultiheadAttention(hidden_size1, 8)
        
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
        """
        conv1 torch.Size([114, 64, 112, 112])
        maxpool torch.Size([114, 64, 56, 56])
        layer1 torch.Size([114, 256, 56, 56])
        cbam_layer1 torch.Size([114, 256, 56, 56])
        """

        print(x.shape)
        
        x = self.feature_extractor.conv1(x)     # print("conv1", x.shape) 
                                                # conv1 torch.Size([110, 64, 112, 112])
        
        x = self.feature_extractor.bn1(x)       # print("bn1", x.shape)
                                                # bn1 torch.Size([110, 64, 112, 112])

        x = self.feature_extractor.relu(x)      # print("relu", x.shape)
                                                # relu torch.Size([110, 64, 112, 112])
      
        """ OPTION 1 ORIGINAL """
        x = self.feature_extractor.maxpool(x)  # print("maxpool", x.shape)
                                                # maxpool torch.Size([110, 64, 56, 56])
        """ OPTION 1 """


        """ OPTION 2 """
        #x = self.conv2(x)                      # [110, 64, 56, 56]
        #x = self.maxpool_spatial(x) * x
        """ OPTION 2 """


        """ OPTION 3 """
        # op 1
        #x = self.conv2(x)                                       # [110, 64, 56, 56]
        # op 2
        #x = self.feature_extractor.maxpool(x)                  # [110, 64, 56, 56]

        """
        layer_avg = x.mean(0, keepdims=True)                    # [1, 64, 56, 56]
        
        kernel_features = torch.flatten(layer_avg, 2)           # [1, 64, 3136]
        kernel_features = kernel_features.permute(1, 0, 2)      # [64, 1, 3136]
        kernel_query = kernel_features.mean(0, keepdims=True)   # [1, 1, 3136]

        kernel_features, kernel_att_map = self.kernel_att_after_maxpool(kernel_query, kernel_features, kernel_features)
        
        #print(kernel_features.shape)                           # [1, 1, 3136]
        #print(kernel_att_map.shape)                            # [1, 1, 64]

        kernel_att_map = kernel_att_map.permute(2, 0 ,1)        # [64, 1, 1]
        kernel_att_map = kernel_att_map.unsqueeze(0)            # [1, 64, 1, 1]

        layer_avg_att = layer_avg * kernel_att_map              # [1, 64, 56, 56]
        
        x = x * layer_avg_att                                   # [110, 64, 56, 56]
        """
        """ OPTION 3 """


        x = self.feature_extractor.layer1(x)    #print("layer1", x.shape)   - [110, 64, 64, 64]
        # Kernal attention pooling between layer 1 and layer 2
        """
        layer_avg = x.mean(0, keepdims=True)                    # [1, 64, 64, 64]
        
        kernel_features = torch.flatten(layer_avg, 2)           # [1, 64, 4096]
        kernel_features = kernel_features.permute(1, 0, 2)      # [64, 1, 4096]
        kernel_query = kernel_features.mean(0, keepdims=True)   # [1, 1, 4096]

        kernel_features, kernel_att_map = self.kernel_att_after_layer1(kernel_query, kernel_features, kernel_features)
        
        kernel_att_map = kernel_att_map.permute(2, 0 ,1)        # [64, 1, 1]
        kernel_att_map = kernel_att_map.unsqueeze(0)            # [1, 64, 1, 1]

        layer_avg_att = layer_avg * kernel_att_map              # [1, 64, 64, 64]
        
        x = x * layer_avg_att                                   # [110, 64, 64, 64]
        """


        x = self.feature_extractor.layer2(x)    #print("layer2", x.shape)   - [110, 512, 28, 28]
        # Kernal attention pooling between layer 2 and layer 3
        """
        layer_avg = x.mean(0, keepdims=True)                    # [1, 512, 28, 28]
        
        kernel_features = torch.flatten(layer_avg, 2)           # [1, 512, 784]
        kernel_features = kernel_features.permute(1, 0, 2)      # [512, 1, 784]
        kernel_query = kernel_features.mean(0, keepdims=True)   # [1, 1, 784]

        kernel_features, kernel_att_map = self.kernel_att_after_layer2(kernel_query, kernel_features, kernel_features)
        
        kernel_att_map = kernel_att_map.permute(2, 0 ,1)        # [512, 1, 1]
        kernel_att_map = kernel_att_map.unsqueeze(0)            # [1, 512, 1, 1]

        layer_avg_att = layer_avg * kernel_att_map              # [1, 512, 28, 28]
        
        x = x * layer_avg_att                                   # [110, 512, 28, 28]
        """


        x = self.feature_extractor.layer3(x)    #print("layer3", x.shape)   - [110, 1024, 14, 14]
        # Kernal attention pooling between layer 3 and layer 4
        """
        layer_avg = x.mean(0, keepdims=True)                    # [1, 1024, 14, 14]
        
        kernel_features = torch.flatten(layer_avg, 2)           # [1, 1024, 196]
        kernel_features = kernel_features.permute(1, 0, 2)      # [1024, 1, 196]
        kernel_query = kernel_features.mean(0, keepdims=True)   # [1, 1, 196]

        kernel_features, kernel_att_map = self.kernel_att_after_layer3(kernel_query, kernel_features, kernel_features)
        
        kernel_att_map = kernel_att_map.permute(2, 0 ,1)        # [1024, 1, 1]
        kernel_att_map = kernel_att_map.unsqueeze(0)            # [1, 1024, 1, 1]

        layer_avg_att = layer_avg * kernel_att_map              # [1, 1024, 14, 14]
        
        x = x * layer_avg_att                                   # [110, 1024, 14, 14]
        """

        x = self.feature_extractor.layer4(x)    #print("layer4", x.shape)   - [110, 2048, 7, 7]

        x = self.feature_extractor.avgpool(x)   #print("avg", x.shape)      - [110, 2048, 1, 1]
        x = torch.flatten(x, 1)                 #                           - [110, 2048]
        x = self.feature_extractor.fc(x)        #                           - [110, 2048]
        features = (x).unsqueeze(1)

        #print("features.shape", features.shape)
        
        """Option 1 ORIGINAL"""
        query = features.mean(0, keepdims=True)
        #print(features.shape)                           # [110, 1, 2048
        #print(query.shape)                              # [1, 1, 2048]
        """Option 1"""


        """Option 2"""
        """
        #temp_features = features.squeeze(1)              # [110, 2048]
        #temp_features = temp_features.permute(1, 0)      # [2048, 110]

        #query = self.att_linear(temp_features)           # [2048, 1]
        #query = query.permute(1, 0)                      # [1, 2048]
        #query = query.unsqueeze(0)                       # [1, 1, 2048]
        """
        """Option 2"""

        #print("query.shape", query.shape)
        
        features, att_map = self.att(query, features, features)

        #print("features.shape", features.shape)

        #print("att_map.shape", att_map.shape)
        
        #print(torch.sum(att_map))

        out = self.classifier(features.squeeze(0))
        # print(out.shape)
        return out