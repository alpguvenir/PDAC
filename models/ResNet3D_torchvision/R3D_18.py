# type: ignore
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name
# pylint: disable=unnecessary-lambda-assignment
# pylint: disable=no-member
# pylint: disable=trailing-whitespace
# pylint: disable=line-too-long
# pylint: disable=wrong-import-position
# pylint: disable=wrong-import-order
# pylint: disable=pointless-string-statement
# pylint: disable=unused-import
# pylint: disable=unspecified-encoding
# pylint: disable=consider-using-enumerate
# pylint: disable=superfluous-parens
# pylint: disable=consider-using-f-string
# pylint: disable=fixme
# pylint: disable=relative-beyond-top-level

import torch
import torch.nn as nn
from torchvision import models

from .ResNet import r3d_18, R3D_18_Weights

class R3D_18(nn.Module):

    def __init__(self):
        super().__init__()

        # https://pytorch.org/vision/main/models/video_mvit.html
        # https://github.com/pytorch/vision/issues/4083
        # img = torch.randn(1, 3, 192, 112, 112).to(torch.float)
        #### (B, T, C, H, W)

        get_params = lambda m: sum(p.numel() for p in m.parameters())
        
        hidden_size1 = 256

        feature_extractor = r3d_18(weights=R3D_18_Weights.DEFAULT)
        
        feature_extractor.stem = nn.Sequential(
            nn.Conv3d(1, 3, 1), feature_extractor.stem
        )

        feature_extractor.fc = (
            nn.Linear(512, hidden_size1)
        )  

        self.feature_extractor = feature_extractor
        #self.att = nn.MultiheadAttention(hidden_size1, 8)
        self.classifier = nn.Linear(hidden_size1, 1)

        print(f"Feature extractor has {get_params(self.feature_extractor)} params")
        #print(f"Attention has {get_params(self.att)} params")
        print(f"Classifier has {get_params(self.classifier)} params")

    def forward(self, x):

        """
        torch.Size([1, 1, 114, 256, 256])       # input
        torch.Size([1, 64, 114, 128, 128])      # after stem
        torch.Size([1, 64, 114, 128, 128])      # after layer 1
        torch.Size([1, 128, 57, 64, 64])        # after layer 2
        torch.Size([1, 256, 29, 32, 32])        # after layer 3
        torch.Size([1, 512, 15, 16, 16])        # after layer 4
        torch.Size([1, 512, 1, 1, 1])           # after avgpool
        torch.Size([1, 512])                    # after flatten
        torch.Size([1, 256])                    # after fc
        """

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
    




    """
    (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Sequential(
        (0): Conv3DSimple(256, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
        (1): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv2): Sequential(
        (0): Conv3DSimple(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        (1): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv3d(256, 512, kernel_size=(1, 1, 1), stride=(2, 2, 2), bias=False)
        (1): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Sequential(
        (0): Conv3DSimple(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        (1): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv2): Sequential(
        (0): Conv3DSimple(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        (1): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (relu): ReLU(inplace=True)
    )
  )
  (avgpool): AdaptiveAvgPool3d(output_size=(1, 1, 1))
  (fc): Linear(in_features=512, out_features=400, bias=True)
)
    """