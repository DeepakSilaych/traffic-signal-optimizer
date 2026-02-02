import torch
import torch.nn as nn
import torch.nn.functional as F


class REsnext(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        self.resx1_main = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=2),
            nn.BatchNorm2d(32)
        )
        self.resx1_shortcut = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32)
        )

        self.resx2_main = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32)
        )

        self.regression_head = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=1000, kernel_size=1, stride=1),
            nn.BatchNorm2d(1000),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1000, out_channels=400, kernel_size=1, stride=1),
            nn.BatchNorm2d(400),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=400, out_channels=1, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x = self.stem(x)
        x_main1 = self.resx1_main(x)
        x_shortcut1 = self.resx1_shortcut(x)
        x = F.relu(x_main1 + x_shortcut1, inplace=True)
        x_main2 = self.resx2_main(x)
        x = F.relu(x_main2 + x, inplace=True)
        x = self.regression_head(x)
        return x
