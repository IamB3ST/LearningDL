import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import random
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime


class MNISTTrainer:
    def __init__(self, model, lr, loss, optimizer, batch_size, epoch, model_type, seed=1234):
        self.model = model
        self.model_type = model_type
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.lr = lr
        self.epoch = epoch
        self.seed = seed

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

        """load model, loss_fn and optimizer"""
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.loss_f = self._prepare_lossfunction()
        if torch.cuda.is_available():
            self.loss_f = self.loss_f.cuda()
        self.optimizer_f = self._prepare_optimizer()
        self.scaler = GradScaler()

        """download and allocate data"""
        self.ROOT = '.data'
        self.train_valid_data = torchvision.datasets.MNIST(root=self.ROOT, train=True, download=True)
        self.test_data = torchvision.datasets.MNIST(root=self.ROOT, train=False, download=False)
        self.VALID_RATE = 0.1
        self.train_data, self.valid_data = torch.utils.data.random_split(self.train_valid_data,
                                                               [int(len(self.train_valid_data) * (1 - self.VALID_RATE)),
                                                                int(len(self.train_valid_data) * self.VALID_RATE)])

        """data normalization"""
        self.mean = self.train_valid_data.data.float().mean() / 255.0
        self.std = self.train_valid_data.data.float().std() / 255.0


        """data augmentation"""
        self.train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation(5, fill=(0,)),
            torchvision.transforms.RandomCrop(28, padding=2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[self.mean], std=[self.std])
        ])
        self.test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[self.mean], std=[self.std])
        ])

        if self.model_type == 'classification':
            self.train_data.dataset.transform = self.train_transforms
        else:
            self.train_data.dataset.transform = self.test_transforms
        self.valid_data.dataset.transform = self.test_transforms
        self.test_data.transform = self.test_transforms



        """load data"""
        self.train_dataloader = torch.utils.data.DataLoader(self.train_data, shuffle=True, batch_size=self.batch_size)
        self.valid_dataloader = torch.utils.data.DataLoader(self.valid_data, batch_size=self.batch_size)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size)

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
                    if self.model_type == 'classification':
                        loss = self.loss_f(outputs, targets)
                    elif self.model_type == 'recovery':
                        bs = inputs.shape[0]
                        inputs = inputs.view(bs, 28 * 28)
                        loss = self.loss_f(outputs, inputs)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer_f)
                self.scaler.update()
                train_losses.append(loss.item())
                train_bar.set_postfix(train_loss=sum(train_losses) / len(train_bar))
            train_bar.close()
            self.train_losses.append(sum(train_losses) / len(train_bar))

            self.model.eval()
            valid_loss = 0
            valid_accuracy = 0
            valid_samples = 0
            valid_bar = tqdm(self.valid_dataloader, desc="[Valid]", ncols=100)
            with torch.no_grad():
                for inputs, targets in valid_bar:
                    if torch.cuda.is_available():
                        inputs, targets = inputs.cuda(), targets.cuda()
                    outputs = self.model(inputs)
                    if self.model_type == 'classification':
                        loss = self.loss_f(outputs, targets)
                    elif self.model_type == 'recovery':
                        bs = inputs.shape[0]
                        inputs = inputs.view(bs, 28 * 28)
                        loss = self.loss_f(outputs, inputs)
                    valid_loss += loss.item()
                    pred = outputs.argmax(1)
                    correct = (pred == targets).sum().item()
                    valid_accuracy += correct
                    valid_samples += targets.size(0)
                    valid_bar.set_postfix(val_loss=valid_loss / valid_samples,
                                          val_acc=valid_accuracy / valid_samples * 100)
            valid_bar.close()
            self.val_losses.append(valid_loss / valid_samples)
            self.val_accuracies.append(valid_accuracy / valid_samples)

        self.plot_training()

    def plot_training(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.epoch + 1), self.train_losses, label='Train Loss')
        plt.plot(range(1, self.epoch + 1), self.val_losses, label='Val Loss')
        plt.plot(range(1, self.epoch + 1), self.val_accuracies, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Metrics')
        plt.title('Training Progress')
        plt.legend()
        plt.show()

    def test(self):
        self.model.eval()
        total_test_loss = 0
        total_test_accuracy = 0
        num_samples = 0
        test_bar = tqdm(self.test_dataloader, desc="Testing [Test]")
        with torch.no_grad():
            for inputs, targets in test_bar:
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                outputs = self.model(inputs)
                if self.model_type == 'classification':
                    loss = self.loss_f(outputs, targets)
                elif self.model_type == 'recovery':
                    bs = inputs.shape[0]
                    inputs = inputs.view(bs, 28 * 28)
                    loss = self.loss_f(outputs, inputs)
                total_test_loss += loss.item()
                accuracy = (outputs.argmax(1) == targets).sum().item()
                total_test_accuracy += accuracy
                num_samples += len(inputs)
                test_bar.set_postfix(test_loss=total_test_loss / num_samples,
                                     test_acc=total_test_accuracy / num_samples * 100)
                test_bar.update()
        test_bar.close()