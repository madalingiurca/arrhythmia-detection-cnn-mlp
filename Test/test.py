from src.ECGDataModule import ECGDataset
import matplotlib.pyplot as plt
import torch

datapath = "data/ptbdb.csv"
ndataset = ECGDataset(datapath, noisy=True)
dataset = ECGDataset(datapath, noisy=False)

auxiliary = (0.01 ** 0.5) * ((torch.randn(8000, 187) + 1) / 2)
plt.plot(auxiliary)
plt.show()
exit(0)
for i in range(3):
    x,y = dataset[i]
    nx, ny = ndataset[i]
    plt.plot(x)
    plt.show()
    plt.plot(nx)
    plt.show()
