"""
Main pipeline of Self-driving car training.

@author: Zhenye Na - https://github.com/Zhenye-Na
@reference: "End to End Learning for Self-Driving Cars", arXiv:1604.07316
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import MultiStepLR

from model import NetworkNvidia, NetworkLight
from trainer import Trainer
from utils import load_data, data_loader


def parse_args():
    """Parse parameters."""
    parser = argparse.ArgumentParser(description='Main pipeline for self-driving vehicles simulation using machine learning.')

    # directory
    parser.add_argument('--dataroot',     type=str,   default="../dataset/", help='path to dataset')
    parser.add_argument('--ckptroot',     type=str,   default="./",          help='path to checkpoint')

    # hyperparameters settings
    parser.add_argument('--lr',           type=float, default=1e-4,          help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,          help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size',   type=int,   default=32,            help='training batch size')
    parser.add_argument('--num_workers',  type=int,   default=8,             help='# of workers used in dataloader')
    parser.add_argument('--train_size',   type=float, default=0.8,           help='train validation set split ratio')
    parser.add_argument('--shuffle',      type=bool,  default=True,          help='whether shuffle data during training')

    # training settings
    parser.add_argument('--epochs',       type=int,   default=60,            help='number of epochs to train')
    parser.add_argument('--start_epoch',  type=int,   default=0,             help='pre-trained epochs')
    parser.add_argument('--resume',       type=bool,  default=True,          help='whether re-training from ckpt')
    parser.add_argument('--model_name',   type=str,   default="nvidia",      help='model architecture to use [nvidia, light]')

    # parse the arguments
    args = parser.parse_args()

    return args


def main():
    """Main pipeline."""
    # parse command line arguments
    args = parse_args()

    # load trainig set and split
    trainset, valset = load_data(args.dataroot, args.train_size)

    print("==> Preparing dataset ...")
    trainloader, validationloader = data_loader(args.dataroot,
                                                trainset, valset,
                                                args.batch_size,
                                                args.shuffle,
                                                args.num_workers)

    # define model
    print("==> Initialize model ...")
    if args.model_name == "nvidia":
        model = NetworkNvidia()
    elif args.model_name == "light":
        model = NetworkLight()

    # define optimizer and criterion
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    # learning rate scheduler
    scheduler = MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)

    # resume
    if args.resume:
        print("==> Loading checkpoint ...")
        # use pre-trained model
        checkpoint = torch.load("model.h5",
                                map_location=lambda storage, loc: storage)

        print("==> Loading checkpoint model successfully ...")
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    # cuda or cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("==> Use accelerator: ", device)

    # training
    print("==> Start training ...")
    trainer = Trainer(args.ckptroot,
                      model,
                      device,
                      args.epochs,
                      criterion,
                      optimizer,
                      scheduler,
                      args.start_epoch,
                      trainloader,
                      validationloader)
    trainer.train()


if __name__ == '__main__':
    main()
