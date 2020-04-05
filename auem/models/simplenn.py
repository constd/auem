from torch import nn
from torch.nn import functional as F


class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes=10):
        super(SimpleNN, self).__init__()
        self.linear1 = nn.Linear(input_size, 1024)
        # self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, 128)
        self.linear6 = nn.Linear(128, num_classes)

    def get_embedding(self, x):
        out = x.reshape(x.size(0), -1)
        out = F.leaky_relu(self.linear1(out))
        # out = F.leaky_relu(self.linear2(out))
        out = F.leaky_relu(self.linear3(out))
        out = F.leaky_relu(self.linear4(out))
        out = F.leaky_relu(self.linear5(out))
        return out

    def forward(self, x):
        out = self.get_embedding(x)
        out = self.linear6(out)
        return nn.functional.softmax(out)
