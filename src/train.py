import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.NeuralModel import model
import time

NO_EPOCHS = 400
NO_BATCH_SIZE = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
train_data = pd.read_csv('data/ptbdb_train.csv').to_numpy()

# train_data_x = torch.from_numpy(train_data[:4000, :-1]).to(device)
# train_data_y = torch.from_numpy(train_data[:4000, -1]).to(device)

train_data_x = torch.utils.data.DataLoader(torch.from_numpy(train_data[:4096, :-1]).to(device), batch_size=NO_BATCH_SIZE)
train_data_y = torch.utils.data.DataLoader(torch.from_numpy(train_data[:4096, -1]).to(device), batch_size=NO_BATCH_SIZE)

# train_data_x = torch.from_numpy(train_data[:1100, :-1]).to(device)
# train_data_y = torch.from_numpy(train_data[:1100, -1]).to(device)

test_x = torch.from_numpy(train_data[4051:4070, :-1]).to(device)
test_y = torch.from_numpy(train_data[4051:4070, -1]).to(device)
del train_data

CNN = model.ConvNN()
CNN.to(device)
CSLoss = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(CNN.parameters(), lr=0.001)

running_loss = 0.0

for epoch in range(1, NO_EPOCHS + 1):
    start = time.time()
    step = 0
    for x, y in zip(train_data_x, train_data_y):
        # print(x, x.shape)
        x = x.reshape([NO_BATCH_SIZE, 1, -1]).float()
        y = y.long()
        optimizer.zero_grad()
        y_pred = CNN(x).to(device)
        loss = CSLoss(y_pred, y)
        # print(loss)
        loss.backward()
        # print(loss, 10 * "-" + "\n")
        optimizer.step()
        running_loss += loss.item()
        step += 1

    print(loss.item())
    print('[%d] loss: %.3f' % (epoch, running_loss / 2000))
    running_loss = 0.0

end = time.time()
print(end - start)
