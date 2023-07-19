import sys; sys.path.append("../")
import argparse
import yaml
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils.data import ADLDataset
from model import WISTAR


def setup_device():
    """
    Sets up the device to be used for training.

    Args:
        None

    Returns:
        device (torch.device): The device to be used for training
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device: {}".format(device))

    return device


def train_one_epoch(model, criterion, optimizer, epoch_index, tb_writer, train_loader, device):
    """
    Trains the WISTAR for one epoch.

    Args:
        model (torch.nn.Module): WISTAR model
        criterion (torch.nn.Module): loss function
        optimizer (torch.optim.Optimizer): optimizer
        epoch_index (int): current epoch index
        tb_writer (torch.utils.tensorboard.SummaryWriter): tensorboard writer
        train_loader (torch.utils.data.DataLoader): training data loader
        device (torch.device): device to be used for training 

    Returns:
        last_loss (float): average loss over the last 10 batches
    """

    # initliaze running and last loss
    running_loss = 0.0
    last_loss = 0.0

    # loop over the dataset multiple times (one epoch)
    for i, (input, lable) in enumerate(train_loader):

        # prepare input and label
        input = input.to(device)
        label = label.squeeze(0).to(device)

        # zero gradients for every batch
        optimizer.zero_grad()

        # make predictions for this batch
        output = model(input, device)

        # compute the loss and its gradients
        loss = criterion(output, label)
        loss.backward()

        # update the weights
        optimizer.step()

        # gather data and report
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10  # average loss over the last 10 batches
            print("  batch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss


def main(config):
    """
    Main function for training the WISTAR. This function sets up the device, creates the datasets and dataloaders,
        initializes the network, loss function and optimizer, and trains the network for the specified number of epochs.
        After each epoch, the validation loss is computed and the model's sate is saved if the validation loss is lower
        that the best validation loss so far.

    References:
        - https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

    Args:
        config (dict): configuration dictionary for training the WISTAR

    Returns:
        None
    """

    # setup device
    device = setup_device()

    # create datasets
    train_dataset = ADLDataset()
    val_dataset = ADLDataset()

    # create dataloaders for our datasets; shuffle for training, not for validation
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # initialize network, criterion and optimizer
    model = WISTAR(config["input_size"], config["hidden_size"], config["n_layers"], config["n_adl"]).to(device)
    criterion = torch.nn.NNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # create a timestamped subdirectory for this run
    timestamp = datetime.now().strftime("%d%m%Y-%H%M%S")
    writer = SummaryWriter(config["run_path"] + "WISTAR_{}".format(timestamp))

    # set number of epochs
    EPOCHS = config["epochs"]

    # track the best validation loss, and the current epoch number
    best_vloss = 1_000_000.0
    epoch_number = 0

    # loop over the dataset multiple times
    for epoch in range(EPOCHS):
        print("EPOCH {}".format(epoch_number + 1))

        # make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(model, criterion, optimizer, epoch_number, writer, train_loader, device)

        # initialize validation loss for this epoch
        running_vloss = 0.0

        # set the mode to evaluation mode, disabling dropout and using population statistics for batch normalization
        model.eval()

        # disable gradient computation and reduce memory consumption
        with torch.no_grad():
            for i, (vinput, vlabel) in enumerate(val_loader):
                vinput = vinput.to(device)
                vlabel = vlabel.squeeze(0).to(device)
                voutput = model(vinput, device)
                vloss = criterion(voutput, vlabel)
                running_vloss += vloss

        # compute the average validation loss for this epoch
        avg_vloss = running_vloss / (i + 1)
        print("LOSS train {} val {}".format(avg_loss, avg_vloss))

        # log the running loss averaged per batch for both training and validation
        writer.add_scalars("Training vs Validatino Loss",
                           { "Training" : avg_loss, "Validation" : avg_vloss },
                            epoch_number + 1)
        writer.flush()

        # track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = config["model_path"] + "WISTAR_{}_{}.pt".format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Train the WISTAR")
    parser.add_argument("-c", "--config", type=str, default="train_config.yaml", help="Path to the configuration file for training the WISTAR")
    args = parser.parse_args()

    # load configuration file
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    # train network
    main(config)