import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime


class TemperatureDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        X = self.data[idx:idx+self.seq_length]
        y = self.data[idx+self.seq_length]
        return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)


class ClimateTrainer:
    def __init__(self, model, lr, loss, optimizer, epoch, samples=5000, seed=1234):
        self.model = model
        self.num_samples = samples
        self.time_steps = 365 * int(self.num_samples/200)
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = 1
        self.lr = lr
        self.epoch = epoch
        self.seed = seed

        self.train_losses = []
        self.val_losses = []

        """load model, loss_fn and optimizer"""
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.loss_f = self._prepare_lossfunction()
        if torch.cuda.is_available():
            self.loss_f = self.loss_f.cuda()
        self.optimizer_f = self._prepare_optimizer()

        """make data"""
        np.random.seed(42)
        self.temperatures = 20 + 10 * np.sin(np.arange(self.time_steps) * (2 * np.pi / 365)) + np.random.normal(scale=2, size=(self.time_steps,))
        self.temperatures = np.expand_dims(self.temperatures, axis=1)

        """data normalization"""
        self.scaler = MinMaxScaler()
        self.temperatures_scaled = self.scaler.fit_transform(self.temperatures)

        self.seq_length = 30
        self.train_size = int(0.8 * len(self.temperatures_scaled))

        self.train_data = self.temperatures_scaled[:self.train_size]
        self.valid_data = self.temperatures_scaled[self.train_size:self.train_size + int(0.1 * len(self.temperatures_scaled))]
        self.test_data = self.temperatures_scaled[self.train_size + int(0.1 * len(self.temperatures_scaled)):]

        self.train_dataset = TemperatureDataset(self.train_data, self.seq_length)
        self.valid_dataset = TemperatureDataset(self.valid_data, self.seq_length)
        self.test_dataset = TemperatureDataset(self.test_data, self.seq_length)

        """load data"""
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size)
        self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

        """set the seed"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

    def _prepare_optimizer(self):
        if self.optimizer == 'SGD':
            return optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer == 'Adam':
            return optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSprop':
            return optim.RMSprop(self.model.parameters(), lr=self.lr)
        elif self.optimizer == 'AdamW':
            return optim.AdamW(self.model.parameters(), lr=self.lr)
        elif self.optimizer == 'Adagrad':
            return optim.Adagrad(self.model.parameters(), lr=self.lr)
        elif self.optimizer == 'Adadelta':
            return optim.Adadelta(self.model.parameters())


    def _prepare_lossfunction(self):
        if self.loss == 'MSE':
            return nn.MSELoss()
        elif self.loss == 'CE':
            return nn.CrossEntropyLoss()
        elif self.loss == 'BCE':
            return nn.BCELoss()
        elif self.loss == 'Logistic':
            return nn.BCEWithLogitsLoss()
        elif self.loss == 'MAE':
            return nn.L1Loss()


    def train(self):
        scaler = GradScaler()
        for epoch in range(self.epoch):
            print('=' * 135)
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print(f"Epoch {epoch + 1} / {self.epoch}")
            self.model.train()
            train_bar = tqdm(self.train_dataloader, desc="[Train]", ncols=90)
            train_losses = []
            for inputs, targets in train_bar:
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda()
                self.optimizer_f.zero_grad()
                with autocast(enabled=torch.cuda.is_available()):
                    outputs = self.model(inputs)
                    loss = self.loss_f(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer_f)
                scaler.update()
                train_losses.append(loss.item())
                train_bar.set_postfix(train_loss=sum(train_losses) / len(train_bar))
            train_bar.close()
            self.train_losses.append(sum(train_losses) / len(train_bar))

            self.model.eval()
            predictions = []
            targets_list = []
            valid_loss = 0
            valid_samples = 0
            valid_bar = tqdm(self.valid_dataloader, desc="[Valid]", ncols=100)
            with torch.no_grad():
                for inputs, targets in valid_bar:
                    if torch.cuda.is_available():
                        inputs, targets = inputs.cuda(), targets.cuda()
                    outputs = self.model(inputs)
                    loss = self.loss_f(outputs, targets)
                    valid_loss += loss.item()
                    valid_samples += targets.size(0)
                    predictions.append(outputs.cpu().numpy())
                    targets_list.append(targets.cpu().numpy())
                    valid_bar.set_postfix(val_loss=valid_loss / valid_samples)
            valid_bar.close()
            predictions = np.concatenate(predictions, axis=0)
            targets = np.concatenate(targets_list, axis=0)
            self.val_losses.append(valid_loss / valid_samples)

        self.plot_valid(predictions, targets)

    @staticmethod
    def plot_valid(predictions, targets):
        plt.figure(figsize=(10, 5))
        plt.plot(targets, label='True', color='blue')
        plt.plot(predictions, label='Predicted', color='red', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('True vs Predicted')
        plt.legend()
        plt.show()

    def test(self):
        self.model.eval()
        predictions = []
        targets_list = []
        total_test_loss = 0
        num_samples = 0
        test_bar = tqdm(self.test_dataloader, desc="Testing [Test]")
        with torch.no_grad():
            for inputs, targets in test_bar:
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                outputs = self.model(inputs)
                loss = self.loss_f(outputs, targets)
                total_test_loss += loss.item()
                num_samples += len(inputs)
                predictions.append(outputs.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
                test_bar.set_postfix(test_loss=total_test_loss / num_samples)
                test_bar.update()
        test_bar.close()
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        self.plot_valid(predictions, targets)
