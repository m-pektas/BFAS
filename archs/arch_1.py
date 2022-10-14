import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self, args: dict):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(110*110*6, args["linear1_out"])
        self.fc2 = nn.Linear(args["linear1_out"], 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def input_producer(self, bs=1, device="cpu"):
        x = torch.FloatTensor(bs, 3, 224, 224).to(device)
        return {"x":x}

if __name__ == "__main__":
    params = {
        "linear1_out" : 120,
        "linear2_out" : 84,
        "n_class" : 10
    }

    net = Net(params)
