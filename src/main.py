"""Main pipeline of Self-driving car training."""

import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


import argparse

from torch.utils import data
from torch.utils.data import DataLoader

from model import NetworkLight
from RcCarDataset import Dataset
from trainer import Trainer


def parse_args():
    """Parse parameters."""
    parser = argparse.ArgumentParser()

    # directory
    parser.add_argument('--dataroot', type=str,
                        default="../track_2_data/", help='path to dataset')
    parser.add_argument('--ckptroot', type=str,
                        default="../model/", help='path to checkpoint')

    # hyperparameters settings
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-5, help='weight decay (L2 penalty)')

    # training settings
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of epochs to train')
    parser.add_argument('--start_epoch', type=int,
                        default=0, help='pre-trained epochs')

    parser.add_argument('--resume', type=bool, default=False,
                        help='whether re-training from ckpt')

    # parse the arguments
    args = parser.parse_args()

    return args


def main():
    """Main function."""
    args = parse_args()

    # Read from the log file
    samples = []
    with open(args.dataroot + "driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for line in reader:
            samples.append(line)

    # Divide the data into training set and validation set
    train_len = int(0.8 * len(samples))
    valid_len = len(samples) - train_len
    train_samples, validation_samples = data.random_split(
        samples, lengths=[train_len, valid_len])

    # Creating generator using the dataloader to parallasize the process
    transformations = transforms.Compose(
        [transforms.Lambda(lambda x: (x / 255.0) - 0.5)])

    params = {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 4
    }

    training_set = Dataset(train_samples, transformations)
    training_generator = DataLoader(training_set, **params)

    validation_set = Dataset(validation_samples, transformations)
    validation_generator = DataLoader(validation_set, **params)

    model = NetworkLight()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    criterion = nn.MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device is: ', device)

    trainer = Trainer(model,
                      device,
                      args.epochs,
                      criterion,
                      optimizer,
                      args.start_epoch,
                      args.lr,
                      training_generator,
                      validation_generator)
    trainer.train()

    # Define state and save the model wrt to state
    state = {
        'model': model.module if device == 'cuda' else model,
    }

    torch.save(state, 'model.h5')


if __name__ == '__main__':
    main()
