import torch
import torchvision
import torch.utils.data.DataLoader as DataLoader
from torchvision import datasets, transforms

class data_proc():

    def __init__(self, datapath='./data/'):
        self.data_path = datapath

    def import_data(self, download_enable=False, batchsize=128, test_batchsize=128):
        train_dataset = datasets.MNIST(root=self.data_path, train=True, transform=transforms.Compose([
                                        transforms.ToTensor(), 
                                        transforms.Normalize((0.1307,), (0.3081,))]), download=download_enable)
        test_dataset = datasets.MNIST(root=self.data_path, train=False, transform=transforms.Compose([
                                        transforms.ToTensor(), 
                                        transforms.Normalize((0.1307,), (0.3081,))]))

        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=test_batchsize, shuffle=True)
        return train_loader, test_loader
