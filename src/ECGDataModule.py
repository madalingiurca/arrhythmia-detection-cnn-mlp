import os
from typing import Union, List, Optional

import pandas as pd, numpy as np
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, random_split, Dataset

class ECGDataset(Dataset):
    def __init__(self, dataPath, noisy=False):
        dataset = np.loadtxt(dataPath, delimiter=",", dtype=np.float64)
        self.no_samples = dataset.shape[0]
        self.noisy = noisy
        if self.noisy == True:
            # add noise over x_data tensor
            self.x_data = torch.from_numpy(dataset[:, :-1]).float()
            auxiliary = (0.01 ** 0.5) * ((torch.randn(8000, 187) + 1) / 2)
            self.x_data += auxiliary
            self.x_data /= torch.max(self.x_data)
        else:
            self.x_data = torch.from_numpy(dataset[:, :-1]).float()

        self.y_data = torch.from_numpy(dataset[:, -1]).long()

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.no_samples


class ECGDataModule(LightningDataModule):
    def __init__(self, data_folder="data", batch_size=64, noisy=False):
        super().__init__()
        self.noisy = noisy
        self.batch_size = batch_size
        self.data_folder = data_folder

    def prepare_data(self, *args, **kwargs):
        df_normal = pd.read_csv(os.path.join(self.data_folder, 'ptbdb_normal.csv'), header=None, nrows=4000)
        df_abnormal = pd.read_csv(os.path.join(self.data_folder, 'ptbdb_abnormal.csv'), header=None, nrows=4000)
        df = pd.concat([df_normal, df_abnormal])
        df.to_csv('data/ptbdb.csv', header=False, index=False)
        dataset = ECGDataset("data/ptbdb.csv", noisy=self.noisy)
        lengths = [int(len(dataset) * 0.80), int(len(dataset) * 0.10), int(len(dataset) * 0.10)]
        self.trainDataset, self.testDataset, self.valDataset = random_split(dataset, lengths,
                                                                            generator=torch.Generator())

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.trainDataset, batch_size=self.batch_size)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.valDataset, batch_size=self.batch_size)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.testDataset, batch_size=self.batch_size)