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


class fashion_mnist_trainer:
    def __init__(self, model, lr, batch_size, epoch, train_transforms, test_transforms, seed=1234):
        self.model = model
        self.lr = lr
        self.epoch = epoch
        self.loss_f = nn.CrossEntropyLoss()

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

        """load model, loss_fn and optimizer"""
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        if torch.cuda.is_available():
            self.loss_f = self.loss_f.cuda()
        self.scaler = GradScaler()

        """download and allocate data"""
        ROOT = '.data'
        train_valid_data = torchvision.datasets.FashionMNIST(root=ROOT, train=True, download=True, transform=train_transforms)
        test_data = torchvision.datasets.FashionMNIST(root=ROOT, train=False, download=False, transform=test_transforms)

        VALID_RATE = 0.1
        n_train = int(len(train_valid_data) * (1 - VALID_RATE))
        n_valid = len(train_valid_data) - n_train
        train_data, valid_data = torch.utils.data.random_split(train_valid_data, [n_train, n_valid])

        """load data"""
        self.train_dataloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
        self.valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
        self.test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

        """set the seed"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def train(self):
        valid_loss_min = np.Inf
        counter = 0
        lr = self.lr
        optimizer_f = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        for epoch in range(self.epoch):
            print('=' * 135)
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print(f"Epoch {epoch + 1} / {self.epoch}")

            if counter / 10 == 1:
                counter = 0
                lr = lr * 0.5
                print(f"lr is updated to {lr}")
                optimizer_f = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

            self.model.train()
            train_bar = tqdm(self.train_dataloader, desc="[Train]", ncols=90)
            train_losses = []
            for inputs, targets in train_bar:
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda()
                optimizer_f.zero_grad()
                with autocast(enabled=torch.cuda.is_available()):
                    outputs = self.model(inputs)
                    loss = self.loss_f(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer_f)
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
                    loss = self.loss_f(outputs, targets)

                    valid_loss += loss.item()
                    pred = outputs.argmax(1)
                    correct = (pred == targets).sum().item()
                    valid_accuracy += correct
                    valid_samples += targets.size(0)
                    val_loss = valid_loss / valid_samples
                    val_acc = valid_accuracy / valid_samples
                    valid_bar.set_postfix(val_loss=val_loss, val_acc=val_acc * 100)

            valid_bar.close()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            if val_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).'.format(valid_loss_min, val_loss))
                valid_loss_min = val_loss
                counter = 0
            else:
                counter += 1
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
                loss = self.loss_f(outputs, targets)

                total_test_loss += loss.item()
                accuracy = (outputs.argmax(1) == targets).sum().item()
                total_test_accuracy += accuracy
                num_samples += len(inputs)
                test_bar.set_postfix(test_loss=total_test_loss / num_samples,
                                     test_acc=total_test_accuracy / num_samples * 100)
                test_bar.update()
        test_bar.close()