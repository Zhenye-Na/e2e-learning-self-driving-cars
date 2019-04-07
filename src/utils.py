"""Util functions."""

import os
import pandas as pd

import torch
from torch.utils import data
from RcCarDataset import TripletDataset

import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def toDevice(datas, device):
    """Enable cuda."""
    imgs, angles = datas
    return imgs.float().to(device), angles.float().to(device)


def load_data(data_dir, train_size):
    """Load training data and train-validation split."""
    # reads CSV file into a single dataframe variable
    data_df = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'),
                          names=['center', 'left', 'right', 'steering',
                                 'throttle', 'reverse', 'speed'])

    # Divide the data into training set and validation set
    train_len = int(train_size * data_df.shape[0])
    valid_len = data_df.shape[0] - train_len
    trainset, valset = data.random_split(
        data_df.values.tolist(), lengths=[train_len, valid_len])

    return trainset, valset


def data_loader(dataroot, trainset, valset, batch_size, shuffle, num_workers):
    """Self-Driving vehicles simulator dataset Loader.

    Args:
        trainset: training set
        valset: validation set
        batch_size: training set input batch size
        shuffle: whether shuffle during training process
        num_workers: number of workers in DataLoader

    Returns:
        trainloader (torch.utils.data.DataLoader): DataLoader for training set
        testloader (torch.utils.data.DataLoader): DataLoader for validation set
    """
    transformations = transforms.Compose(
        [transforms.Lambda(lambda x: (x / 127.5) - 1.0)])

    # Load training data and validation data
    training_set = TripletDataset(dataroot, trainset, transformations)
    trainloader = DataLoader(training_set,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers)

    validation_set = TripletDataset(dataroot, valset, transformations)
    valloader = DataLoader(validation_set,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers)

    return trainloader, valloader
