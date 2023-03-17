

from .dataloader import Database
from torch import optim
from torch import nn
import numpy as np
from tqdm import tqdm
from ..utils.vision import tensor2im

class Trainer:
    def __init__(self, model, dataset, batch_size, device):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.batch_size = batch_size
        self.train_loader, self.val_loader = Database(batch_size).getDataset(dataset)
    
    def train(self, step):
        self.model.train()
        for _ in tqdm(range(step), desc="Training ..."):
            batch = next(self.train_loader)
            image = batch[0]
            label = batch[1]
            image = image.to(self.device)
            label = label.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
    
    def test(self, step):
        test_loss = []
        test_acc = []
        self.model.eval()

        for _ in tqdm(range(step), desc="Testing ..."):
            batch = next(iter(self.val_loader))
            image = batch[0].to(self.device)
            label = batch[1].to(self.device)
            output = self.model(image)
            loss = self.criterion(output, label)
            test_loss.append(loss.item())
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            acc = pred.eq(label.view_as(pred)).sum().item()/pred.shape[0]
            test_acc.append(acc)

        return np.mean(test_loss), np.mean(test_acc)