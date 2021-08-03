import torch
import torchvision
import torch.utils.data.DataLoader as DataLoader
from torchvision import datasets, transforms
from .dataset import data_proc

def main():
    data_importer = data_proc()
    trainloader, testloader = data_importer.import_data(download_enable=True)
    print('load successfully !')

if __name__ == '__main__':
    main()
