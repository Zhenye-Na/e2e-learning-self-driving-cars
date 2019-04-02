"""Main pipeline of Self-driving car training."""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR

from model import NetworkLight, NetworkNvidia
from trainer import Trainer
from utils import load_data, data_loader


def parse_args():
    """Parse parameters."""
    parser = argparse.ArgumentParser()

    # directory
    parser.add_argument('--dataroot',     type=str,   default="../data/", help='path to dataset')
    parser.add_argument('--ckptroot',     type=str,   default="./",       help='path to checkpoint')

    # hyperparameters settings
    parser.add_argument('--lr',           type=float, default=0.0001,     help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,       help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size',   type=int,   default=32,         help='training batch size')
    parser.add_argument('--num_workers',  type=int,   default=8,          help='number of workers in dataloader')
    parser.add_argument('--test_size',    type=float, default=0.8,        help='train validation set split ratio')
    parser.add_argument('--shuffle',      type=bool,  default=True,       help='whether shuffle data during training')
    # parser.add_argument('--p',            type=float, default=0.25,       help='probability of an element to be zeroed')

    # training settings
    parser.add_argument('--epochs',       type=int,   default=10,         help='number of epochs to train')
    parser.add_argument('--start_epoch',  type=int,   default=0,          help='pre-trained epochs')
    parser.add_argument('--resume',       type=bool,  default=False,      help='whether re-training from ckpt')

    # parse the arguments
    args = parser.parse_args()

    return args


def main():
    """Main pipeline."""
    args = parse_args()

    # load trainig set and split
    trainset, valset = load_data(args.dataroot, args.test_size)

    # ------------------ Oringinally ------------------

    # samples = []
    # with open(args.dataroot + "driving_log.csv") as csvfile:
    #     reader = csv.reader(csvfile)
    #     next(reader, None)
    #     for line in reader:
    #         samples.append(line)

    # # Divide the data into training set and validation set
    # train_len = int(0.8 * len(samples))
    # valid_len = len(samples) - train_len
    # train_samples, validation_samples = data.random_split(
    #     samples, lengths=[train_len, valid_len])

    # ------------------ Oringinally ------------------

    # Creating generator using the dataloader to parallasize the process
    # transformations = transforms.Compose(
    #     [transforms.Lambda(lambda x: (x / 127.5) - 1.0)])

    # # Load training data and validation data
    # training_set = TripletDataset(train_set, transformations)
    # training_generator = DataLoader(training_set,
    #                                 batch_size=args.batch_size,
    #                                 shuffle=args.shuffle,
    #                                 num_workers=args.num_workers)

    # validation_set = TripletDataset(val_set, transformations)
    # validation_generator = DataLoader(validation_set,
    #                                   batch_size=args.batch_size,
    #                                   shuffle=args.shuffle,
    #                                   num_workers=args.num_workers)

    # ------------------ Oringinally ------------------

    print("==> Preparing dataset ...")
    trainloader, validationloader = data_loader(args.dataroot,
                                                trainset, valset,
                                                args.batch_size,
                                                args.shuffle,
                                                args.num_workers)

    # Define model
    print("==> Initialize model ...")
    model = NetworkNvidia()

    # Define optimizer and criterion
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    if args.resume:
        print("==> Loading checkpoint ...")
        checkpoint = torch.load("model-10.h5",
                                map_location=lambda storage, loc: storage)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # learning rate scheduler
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
