import torch
# import torchvision
# from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # daca folosesc dropout si batchnorm nu mai face overfit si da rezultate mai ok
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


net = Net()
# print(net)

learningrate = 0.005

optimizer = optim.Adam(net.parameters(), lr=learningrate)
loss_function = nn.MSELoss()

data = np.load(r"C:\Users\PC\Desktop\etti\etti_an_4\p3\training_data.npy", allow_pickle=True)
test_data = []
test_percent = round((2 / 10) * len(data))
validation_percent = round((1 / 20) * len(data))
for i in range(test_percent):
    random_index = round(len(data) * random.random())
    test_data.append(data[random_index])
    data = np.delete(data, random_index, 0)

validation_data = []
for i in range(test_percent):
    random_index = round(len(data) * random.random())
    validation_data.append(data[random_index])
    data = np.delete(data, random_index, 0)

X_train = torch.Tensor([i[0] for i in data])
y_train = torch.Tensor([i[1] for i in data])

X_validation = torch.Tensor([i[0] for i in validation_data])
y_validation = torch.Tensor([i[1] for i in validation_data])

# for index in range(len(X_train)):
# X_train[index]=X_train[index] + (0.5**0.5)*torch.randn(187)

## daca bag zgomot se duce drc tot

BATCH_SIZE = 64
EPOCHS = 100
ss = 50
gm = 0.2
## cea mai buna pana acum, lr=0.005 , stepsize 50, gamma 0.2 batch size 64
index = []
mylist = []

acc_plot = []
# lmbda= lambda epoch: 0.95
scheduler = StepLR(optimizer, step_size=ss, gamma=gm)
for epoch in range(EPOCHS):
    for i in range(0, len(X_train), BATCH_SIZE):
        # print(i)
        batch_X = X_train[i:i + BATCH_SIZE]
        batch_y = y_train[i:i + BATCH_SIZE]
        net.zero_grad()
        output = net(batch_X)
        loss = loss_function(output, batch_y)
        loss.backward()
        optimizer.step()
    scheduler.step()
    # for param_group in optimizer.param_groups:
    # print("Current learning rate is: {}".format(param_group['lr']))
    print(loss)
    index.append(epoch)
    mylist.append(loss.item())
    correct = 0
    total = len(validation_data)
    for i in range(total):
        output = net(X_validation[i])
        class_1 = output[0].item()
        class_2 = output[1].item()
        if (class_1 > class_2):
            final_class_val = class_1
        else:
            final_class_val = class_2
        if (y_validation[i][0].item() > y_validation[i][1].item()):
            real_class = class_1
        else:
            real_class = class_2
        if (real_class == final_class_val):
            correct += 1
    acc_plot.append(correct / len(validation_data))

correct = 0
total = len(test_data)
X_test = torch.Tensor([i[0] for i in test_data])
y_test = torch.Tensor([i[1] for i in test_data])

for i in range(total):
    output = net(X_test[i])
    class_1 = output[0].item()
    class_2 = output[1].item()
    if (class_1 > class_2):
        final_class_val = class_1
    else:
        final_class_val = class_2
    if (y_test[i][0].item() > y_test[i][1].item()):
        real_class = class_1
    else:
        real_class = class_2
    if (real_class == final_class_val):
        correct += 1
print(correct / total)
# print(index)
plt.plot(index, mylist, label=str(correct / total))
plt.legend()
title = ''
# title=title+'learning_rate_'+str(learningrate) +'_batch_size_' + str(BATCH_SIZE) + '_epochs_'+ str(EPOCHS) + '_step_size_' + str(ss) + '_gamma_pe_10_' + str(gm*10)
# plt.savefig(title+'.png')
plt.figure()
plt.plot(index, acc_plot)
