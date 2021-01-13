from pytorch_lightning.trainer import Trainer

from ..src.ECGDataModule import ECGDataModule
from ..src.NeuralModel.CNN import CNN

if __name__ == "__main__":
    '''Loading the dataset'''
    dataModule = ECGDataModule(batch_size=100)

    # '''Model Training'''
    model = CNN()
    # '''tuning the trainer'''
    trainer = Trainer(max_epochs=100, gpus=1)

    # '''start training'''
    trainer.test(model, datamodule=dataModule)
