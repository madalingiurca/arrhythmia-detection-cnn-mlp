from torch import nn


class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 5, kernel_size=13, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(5, 10, kernel_size=10, stride=1, padding=1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(10, 20, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=5)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(20, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2)
        )
        self.drop_out = nn.Dropout()
        self.layer4 = nn.Sequential(
            nn.Linear(800, 300),
            nn.Linear(300, 50),
            nn.Linear(50, 2)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        x = self.layer4(x)
        return x
