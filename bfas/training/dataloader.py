
import torch
import torchvision.transforms as T
from torchvision.datasets import MNIST, CIFAR10

class Database:
    
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def getDataset(self, dataset : str):
        if dataset == "mnist":
            return self.get_mnist()
        elif dataset == "cifar10":
            return self.get_cifar10()
        else:
            raise ValueError("Dataset is not supported.")
    

    
    def get_mnist(self):

        img_transform= T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Lambda(lambda x: torch.cat([x, x, x], 0)),
            T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])
        
        train_dataset = MNIST(root="data", train=True, transform=img_transform, download=True)
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=True)
        
        test_dataset = MNIST(root="data", train=False, transform=img_transform, download=True)
        test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=True)
        
        return iter(train_data_loader), iter(test_data_loader)
    
    def get_cifar10(self):
            
            img_transform= T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            ])
            
            train_dataset = CIFAR10(root="data", train=True, transform=img_transform, download=True)
            train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=True)
            
            test_dataset = CIFAR10(root="data", train=False, transform=img_transform, download=True)
            test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=True)
            
            return iter(train_data_loader), iter(test_data_loader)
    
    