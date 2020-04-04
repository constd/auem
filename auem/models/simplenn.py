from torch.nn import functional as F
from torch import nn


class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes=10):
        super(SimpleNN, self).__init__()
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = x.reshape(x.size(0), -1)
        out = F.leaky_relu(self.linear1(out))
        out = self.linear2(out)
        return out
