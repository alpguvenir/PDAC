import torch
import torch.nn as nn
from torchvision import models
from functools import partial

class Channel_Attention(nn.Module):
    '''Channel Attention in CBAM.
    '''

    def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max']):
        '''Param init and architecture building.
        '''

        super(Channel_Attention, self).__init__()
        self.pool_types = pool_types

        self.shared_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=channel_in, out_features=channel_in//reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel_in//reduction_ratio, out_features=channel_in)
        )


    def forward(self, x):
        '''Forward Propagation.
        '''

        channel_attentions = []

        for pool_types in self.pool_types:
            if pool_types == 'avg':
                pool_init = nn.AvgPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                avg_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(avg_pool))
            elif pool_types == 'max':
                pool_init = nn.MaxPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                max_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(max_pool))

        pooling_sums = torch.stack(channel_attentions, dim=0).sum(dim=0)
        scaled = nn.Sigmoid()(pooling_sums).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scaled #return the element-wise multiplication between the input and the result.


class ChannelPool(nn.Module):
    '''Merge all the channels in a feature map into two separate channels where the first channel is produced by taking the max values from all channels, while the
       second one is produced by taking the mean from every channel.
    '''
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class Spatial_Attention(nn.Module):
    '''Spatial Attention in CBAM.
    '''

    def __init__(self, kernel_size=7):
        '''Spatial Attention Architecture.
        '''

        super(Spatial_Attention, self).__init__()

        self.compress = ChannelPool()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, dilation=1, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.01, affine=True)
        )


    def forward(self, x):
        '''Forward Propagation.
        '''
        x_compress = self.compress(x)
        x_output = self.spatial_attention(x_compress)
        scaled = nn.Sigmoid()(x_output)
        return x * scaled


class CBAM(nn.Module):
    '''CBAM architecture.
    '''
    def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max'], spatial=True):
        '''Param init and arch build.
        '''
        super(CBAM, self).__init__()
        self.spatial = spatial

        self.channel_attention = Channel_Attention(channel_in=channel_in, reduction_ratio=reduction_ratio, pool_types=pool_types)

        if self.spatial:
            self.spatial_attention = Spatial_Attention(kernel_size=7)


    def forward(self, x):
        '''Forward Propagation.
        '''
        x_out = self.channel_attention(x)
        if self.spatial:
            x_out = self.spatial_attention(x_out)

        return x_out




class ResNet152_inter_layer_CBAM_MultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        get_params = lambda m: sum(p.numel() for p in m.parameters())

        hidden_size1 = 2048
        
        feature_extractor = models.resnet.resnet152(weights=models.ResNet152_Weights.DEFAULT)

        # Convert a input 1 channel image to output 3 channel image -> Input [151, 1, 256, 256]
        # ResNet conv1 is normally Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # So we convert it to 
        # (0): Conv2d(1, 3, kernel_size=(1, 1), stride=(1, 1))
        # (1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        feature_extractor.conv1 = nn.Sequential( nn.Conv2d(1, 3, 1), feature_extractor.conv1 )


        # Fully connected layer returning (hidden_size1) 256 units
        # fc contains 1000 nodes at the end, so override it to keep the same number of nodes as in_features = 2048
        
        feature_extractor.fc =  (
                                    nn.Linear(hidden_size1, hidden_size1) if hidden_size1 != 512 else nn.Identity()
                                )


        self.cbam_layer1 = CBAM(channel_in=256)
        self.cbam_layer2 = CBAM(channel_in=512)
        self.cbam_layer3 = CBAM(channel_in=1024)
        self.cbam_layer4 = CBAM(channel_in=2048)

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
        """
        conv1 torch.Size([114, 64, 112, 112])
        maxpool torch.Size([114, 64, 56, 56])
        layer1 torch.Size([114, 256, 56, 56])
        cbam_layer1 torch.Size([114, 256, 56, 56])
        """
        
        x = self.feature_extractor.conv1(x)
        # print("conv1", x.shape) 
        # conv1 torch.Size([110, 64, 112, 112])
        
        x = self.feature_extractor.bn1(x)
        # print("bn1", x.shape)
        # bn1 torch.Size([110, 64, 112, 112])

        x = self.feature_extractor.relu(x)  
        # print("relu", x.shape)
        # relu torch.Size([110, 64, 112, 112])
      
        x = self.feature_extractor.maxpool(x)
        # print("maxpool", x.shape)
        # maxpool torch.Size([110, 64, 56, 56])

        x = self.feature_extractor.layer1(x)
        #print("layer1", x.shape)
        x = self.cbam_layer1(x)

        x = self.feature_extractor.layer2(x)
        #print("layer2", x.shape)
        x = self.cbam_layer2(x)

        x = self.feature_extractor.layer3(x)
        #print("layer3", x.shape)
        #x = self.cbam_layer3(x)

        x = self.feature_extractor.layer4(x)
        #print("layer4", x.shape)
        #x = self.cbam_layer4(x)

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