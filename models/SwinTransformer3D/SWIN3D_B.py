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
from .SwinTransformer import swin3d_b, Swin3D_B_Weights

class SWIN3D_B(nn.Module):

    def __init__(self):
        super().__init__()
        
        # https://pytorch.org/vision/main/models/video_mvit.html
        # https://github.com/pytorch/vision/issues/4083
        # img = torch.randn(1, 3, 192, 112, 112).to(torch.float)
        #### (B, T, C, H, W)

        #1 1 100 224 224

        get_params = lambda m: sum(p.numel() for p in m.parameters())
        
        hidden_size1 = 256

        feature_extractor = swin3d_b(weights=Swin3D_B_Weights)

        feature_extractor.patch_embed = nn.Sequential(
            nn.Conv3d(1,3,1), feature_extractor.patch_embed
        )

        # (786, 400)
        feature_extractor.head = (
            nn.Linear(1024, hidden_size1)
        )

        self.feature_extractor = feature_extractor
        self.classifier = nn.Linear(hidden_size1, 1)

        print(f"Feature extractor has {get_params(self.feature_extractor)} params")
        print(f"Classifier has {get_params(self.classifier)} params")

    def forward(self, x):

        features = self.feature_extractor(x).unsqueeze(
            1
        )  # assuming only 1 CT at a time

        out = self.classifier(features.squeeze(0))
        # print(out.shape)

        return out
