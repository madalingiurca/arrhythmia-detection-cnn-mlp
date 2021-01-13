from pytorch_lightning.trainer import Trainer, seed_everything

from ECGDataModule import ECGDataModule
from NeuralModel.CNN import CNN

if __name__ == "__main__":
    '''Loading the dataset'''
    seed_everything(77)
    dataModule = ECGDataModule(batch_size=64, noisy=True)

    # '''Model Training'''
    model = CNN()

    # '''tuning the trainer'''
    trainer = Trainer(gpus=1, max_epochs=100, deterministic=True, auto_lr_find=False)

    trainer.tune(model,datamodule=dataModule)

    # '''start training'''
    trainer.fit(model, datamodule=dataModule)
    trainer.test(model, datamodule=dataModule)
