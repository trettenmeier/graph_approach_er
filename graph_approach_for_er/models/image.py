import torch
import torch.nn as nn
from torchvision import models


def get_model():
    return SiameseNetwork()


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Using a pre-trained ResNet
        self.feature_extractor = models.resnet18(weights='IMAGENET1K_V1')
        self.feature_extractor.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, img1, img2):
        feat1 = self.feature_extractor(img1)
        feat2 = self.feature_extractor(img2)

        distance = torch.abs(feat1 - feat2)

        output = self.fc(distance)
        return output
