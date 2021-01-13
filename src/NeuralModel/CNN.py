# CNN Model with Pytorch Lighting Structure
import pytorch_lightning as pl
from torch import nn
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy


# noinspection PyAbstractClass
class CNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.learning_rate = 10e-3
        # self.learning_rate = 5e-3
        # self.learning_rate = 1e-3
        self.learning_rate = 5e-4
        # self.learning_rate = 1e-4
        self.batch_size = None

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 50, kernel_size=6, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=6, stride=6, padding=1)
            # nn.AvgPool1d(kernel_size=3, stride=3, padding=1)
        )
        # nn.init.xavier_normal_(self.layer1[0].weight)
        self.layer2 = nn.Sequential(
            nn.Conv1d(50, 70, kernel_size=6, padding=1),
            nn.ReLU(),
        )
        # nn.init.xavier_normal_(self.layer2[0].weight)
        self.layer3 = nn.Sequential(
            nn.Conv1d(70, 100, kernel_size=6, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=6, stride=6, padding=1)
            # nn.AvgPool1d(kernel_size=3, stride=3, padding=1)
        )
        # nn.init.xavier_normal_(self.layer3[0].weight)
        self.drop_out = nn.Dropout(0.55)
        self.layer4 = nn.Sequential(
            # nn.Linear(1050, 300),
            # nn.Linear(9350, 300),
            nn.Linear(400, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
            # nn.LogSoftmax()
        )
        # nn.init.xavier_normal_(self.layer4[0].weight)
        # nn.init.xavier_normal_(self.layer4[1].weight)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        try:
            x = self.layer4(x)
        except:
            print("\nneurons after reshape = {}".format(x.shape[1]))
            exit(2)
        # x = self.layer4(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.75)
        return [optimizer], [scheduler]
        # return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), 1, -1)
        y_pred = self.forward(x)
        loss = F.cross_entropy(y_pred, y)
        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy(y_pred, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), 1, -1)
        y_pred = self.forward(x)
        loss = F.cross_entropy(y_pred, y)
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy(y_pred, y))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), 1, -1)
        y_pred = self.forward(x)
        self.log('accuracy', accuracy(y_pred, y))



