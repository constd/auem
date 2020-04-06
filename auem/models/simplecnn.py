import torch
from torch import nn
from torch.nn import functional as F


class SimpleCNNBase(torch.nn.Module):
    def __init__(self, input_size, num_classes=10):
        super(SimpleCNNBase, self).__init__()
        # (x, 1, 128, 2206)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=7, stride=2, bias=False)
        # (x, 8, (128 - 7 + 1)/2 = 61, (2206 - 7 + 1)/2 = 1100)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=(1, 2), bias=False)
        # (x, 16, (61 - 5 + 1)=57, (1100-5+1)/2 = 548 )
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=(1, 2), bias=False)
        # (x, 16, (57 - 3 + 1)=55, (548-3+1)/2 = 273 )
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=4, stride=(2, 2), bias=False)
        # (x, 16, (55 - 4 + 1) / 2=26, (273-4+1)/2 = 135 )
        self.bn4 = nn.BatchNorm2d(32)

        self.linear1 = nn.Linear(32 * 26 * 135, 128)
        self.linear2 = nn.Linear(128, num_classes)

    def get_embedding(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = out.view(out.size(0), -1)
        out = F.relu(self.linear1(out))
        return out

    def forward(self, x):
        out = self.get_embedding(x)
        out = self.linear2(out)
        return out
