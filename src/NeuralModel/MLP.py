from typing import List, Any

import pytorch_lightning as pl
from torch import nn
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy


class MLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # daca folosesc dropout si batchnorm nu mai face overfit si da rezultate mai ok
        self.learning_rate = 0.005
        self.batch_size = None

        self.fc1 = nn.Linear(187, 100)
        nn.Dropout(0.5)
        nn.BatchNorm1d(100)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(100, 50)
        nn.Dropout(0.2)
        nn.BatchNorm1d(50)
        ##DROPOUT
        ##adaugat zgomot sigma=0,
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc3 = nn.Linear(50, 25)
        nn.Dropout(0.2)
        nn.BatchNorm1d(25)
        torch.nn.init.kaiming_normal_(self.fc3.weight)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc4 = nn.Linear(25, 2)
        torch.nn.init.kaiming_normal_(self.fc4.weight)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        # x=torch.tanh(self.fc1(x))
        # x=torch.tanh(self.fc2(x))
        # x=torch.tanh(self.fc3(x))
        # x=torch.tanh(self.fc4(x))
        # x=F.relu(self.fc1(x))
        # x=F.relu(self.fc2(x))
        # x=F.relu(self.fc3(x))
        # x=F.relu(self.fc4(x))
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.2)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        output = self.forward(x)
        loss = F.cross_entropy(output, y)
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, outputs: List[Any]) -> None:
        super().training_epoch_end(outputs)
        opt = self.optimizers()
        for param_group in self.optimizers().param_groups:
            print("Current learning rate is: {}".format(param_group['lr']))

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = F.cross_entropy(output, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        self.log('accuracy', accuracy(output, y))
