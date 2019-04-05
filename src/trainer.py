"""Trainer."""

import os
import torch

from utils import toDevice


class Trainer(object):
    """Trainer class."""

    def __init__(self,
                 ckptroot,
                 model,
                 device,
                 epochs,
                 criterion,
                 optimizer,
                 scheduler,
                 start_epoch,
                 trainloader,
                 validationloader):
        """Self-Driving car Trainer.

        Args:
            model: CNN model
            device: cuda or cpu
            epochs: epochs to training neural network
            criterion: nn.MSELoss()
            optimizer: optim.Adam()
            start_epoch: 0 or checkpoint['epoch']
            trainloader: training set loader
            validationloader: validation set loader

        """
        super(Trainer, self).__init__()

        self.model = model
        self.device = device
        self.epochs = epochs
        self.ckptroot = ckptroot
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.start_epoch = start_epoch
        self.trainloader = trainloader
        self.validationloader = validationloader

    def train(self):
        """Training process."""
        self.model.to(self.device)
        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            self.scheduler.step()

            # Training
            train_loss = 0.0
            self.model.train()

            for local_batch, (centers, lefts, rights) in enumerate(self.trainloader):
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

                    print("Training Epoch: {} | Loss: {}".format(epoch, train_loss / (local_batch + 1)))

            # Validation
            self.model.eval()
            valid_loss = 0
            with torch.set_grad_enabled(False):
                for local_batch, (centers, lefts, rights) in enumerate(self.validationloader):
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
                        print("Validation Loss: {}".format(valid_loss / (local_batch + 1)))

            print()

            # Save model
            if epoch % 5 == 0 or epoch == self.epochs + self.start_epoch - 1:

                print("==> Save checkpoint ...")

                state = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                }

                self.save_checkpoint(state)

    def save_checkpoint(self, state):
        """Save checkpoint."""
        if not os.path.exists(self.ckptroot):
            os.makedirs(self.ckptroot)

        torch.save(state, self.ckptroot + 'model-{}.h5'.format(state['epoch']))
