"""Trainer."""

import torch

from utils import toDevice


class Trainer(object):

    def __init__(self,
                 model,
                 device,
                 epochs,
                 criterion,
                 optimizer,
                 start_epoch,
                 learning_rate,
                 training_generator,
                 validation_generator):
        """Trainer with generator BUilder."""
        super(Trainer, self).__init__()

        self.model = model
        self.device = device
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.start_epoch = start_epoch
        self.learning_rate = learning_rate
        self.training_generator = training_generator
        self.validation_generator = validation_generator

    def train(self):
        """Training process."""

        for epoch in range(self.epochs):
            self.model.to(self.device)

            # Training
            train_loss = 0.0
            self.model.train()

            for local_batch, (centers, lefts, rights) in enumerate(self.training_generator):
                # Transfer to GPU
                centers, lefts, rights = toDevice(centers, self.device), toDevice(
                    lefts, self.device), toDevice(rights, self.device)

                # Model computations
                self.optimizer.zero_grad()
                datas = [centers, lefts, rights]
                for data in datas:
                    imgs, angles = data
                    # print("training image: ", imgs.shape)
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, angles.unsqueeze(1))
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.data.item()

                if local_batch % 100 == 0:

                    print("Training Epoch: {} | Loss: {}".format(
                        epoch, train_loss / (local_batch + 1)))

            # Validation
            self.model.eval()
            valid_loss = 0
            with torch.set_grad_enabled(False):
                for local_batch, (centers, lefts, rights) in enumerate(self.validation_generator):
                    # Transfer to GPU
                    centers, lefts, rights = toDevice(centers, self.device), toDevice(
                        lefts, self.device), toDevice(rights, self.device)

                    # Model computations
                    self.optimizer.zero_grad()
                    datas = [centers, lefts, rights]
                    for data in datas:
                        imgs, angles = data
                        outputs = self.model(imgs)
                        loss = self.criterion(outputs, angles.unsqueeze(1))

                        valid_loss += loss.data.item()

                    if local_batch % 100 == 0:
                        print("Validation Loss: {}\n".format(
                            valid_loss / (local_batch + 1)))
