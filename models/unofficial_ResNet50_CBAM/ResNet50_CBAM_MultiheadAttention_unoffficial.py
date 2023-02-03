# https://github.com/Ugenteraan/ResNet-50-CBAM-PyTorch/blob/main/models/resnet50.py

# https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py -> check this also
# https://github.com/elbuco1/CBAM
# https://keras.io/examples/vision/3D_image_classification/

import torch
import torch.nn as nn
from .CBAM import CBAM

class BottleNeck(nn.Module):
    '''Bottleneck modules
    '''

    def __init__(self, in_channels, out_channels, expansion=4, stride=1, use_cbam=True):
        '''Param init.
        '''
        super(BottleNeck, self).__init__()

        self.use_cbam = use_cbam
        #only the first conv will be affected by the given stride parameter. The rest have default stride value (which is 1).
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels*expansion)
        self.relu = nn.ReLU(inplace=True)

        #since the input has to be same size with the output during the identity mapping, whenever the stride or the number of output channels are
        #more than 1 and expansion*out_channels respectively, the input, x, has to be downsampled to the same level as well.
        self.identity_connection = nn.Sequential()
        if stride != 1 or in_channels != expansion*out_channels:
            self.identity_connection = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=expansion*out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels*expansion)
            )

        if self.use_cbam:
            self.cbam = CBAM(channel_in=out_channels*expansion)


    def forward(self, x):
        '''Forward Propagation.
        '''

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.use_cbam:
            out = self.cbam(out)

        out += self.identity_connection(x) #identity connection/skip connection
        out = self.relu(out)

        return out


class ResNet50_CBAM_MultiheadAttention_unoffficial(nn.Module):
    '''ResNet-50 Architecture.
    '''

    # def __init__(self, use_cbam=True, image_depth=3, num_classes=6):
    def __init__(self, use_cbam=True, image_depth=1, num_classes=1):
        '''Params init and build arch.
        '''
        super(ResNet50_CBAM_MultiheadAttention_unoffficial, self).__init__()


        hidden_size1 = 2048

        # Number of heads 8, embedding dimension 256
        self.att = nn.MultiheadAttention(hidden_size1, 16)
        
        # Classifier returning with only 1 unit, binary
        self.classifier = nn.Linear(hidden_size1, 1)




        self.in_channels = 64
        self.expansion = 4
        self.num_blocks = [3, 3, 3, 2]

        self.conv_block1 = nn.Sequential(nn.Conv2d(kernel_size=7, stride=2, in_channels=image_depth, out_channels=self.in_channels, padding=3, bias=False),
                                            nn.BatchNorm2d(self.in_channels),
                                            nn.ReLU(inplace=True),
                                            nn.MaxPool2d(stride=2, kernel_size=3, padding=1))

        self.layer1 = self.make_layer(out_channels=64, num_blocks=self.num_blocks[0], stride=1, use_cbam=use_cbam)
        self.layer2 = self.make_layer(out_channels=128, num_blocks=self.num_blocks[1], stride=2, use_cbam=use_cbam)
        self.layer3 = self.make_layer(out_channels=256, num_blocks=self.num_blocks[2], stride=2, use_cbam=use_cbam)
        self.layer4 = self.make_layer(out_channels=512, num_blocks=self.num_blocks[3], stride=2, use_cbam=use_cbam)
        self.avgpool = nn.AvgPool2d(7)
        # Original 
        # self.linear = nn.Linear(512*self.expansion, num_classes)
        self.linear = nn.Linear(512*self.expansion, hidden_size1)


    def make_layer(self, out_channels, num_blocks, stride, use_cbam):
        '''To construct the bottleneck layers.
        '''
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BottleNeck(in_channels=self.in_channels, out_channels=out_channels, stride=stride, expansion=self.expansion, use_cbam=use_cbam))
            self.in_channels = out_channels * self.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        '''Forward propagation of ResNet-50.
        '''

        x = self.conv_block1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_conv = self.layer4(x)
        x = self.avgpool(x_conv)
        x = nn.Flatten()(x) #flatten the feature maps.
        x = self.linear(x)

        features = (x).unsqueeze(1)

        query = features.mean(0, keepdims=True)
        
        features, att_map = self.att(query, features, features)

        out = self.classifier(features.squeeze(0))

        return out