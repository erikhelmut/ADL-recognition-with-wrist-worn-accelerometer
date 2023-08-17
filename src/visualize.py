import argparse
import yaml

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.data import init_datasets, ADLDataset
from model import WISTAR


def main(config):
    """
    Visualizes the performance of WISTAR on the test set.

    Args:
        config (dict): configuration dictionary for visualizing the WISTAR

    Returns:
        None
    """

    # setup device
    device = "cpu"

    # load model
    model = WISTAR(config["input_size"], config["hidden_size"], config["n_layers"], config["n_adl"])
    model.load_state_dict(torch.load("../models/WISTAR_17082023-125201_41.pt"))
    model.eval()

    # load test dataset
    #_, _, test_dataset = init_datasets(config["data_path"], config["train_split"], config["val_split"], config["seed"])
    _, _, test_dataset = init_datasets(config["data_path"], 0, 0, config["seed"])

    # create dataloader for test dataset
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # create confusion matrix
    confusion = torch.zeros(config["n_adl"], config["n_adl"])

    with torch.no_grad():
        n_correct = 0
        n_samples = 0

        for i, (input, label) in enumerate(test_loader):
                
            # prepare input and label
            input = input
            label = label.squeeze(0)

            # make predictions for this batch
            output = model(input, device)

            # get predicted label
            predicted_idx = torch.argmax(output, dim=1)

            # update confusion matrix
            confusion[predicted_idx, label] += 1

            # update number of correct predictions and number of samples
            n_correct += (predicted_idx == label).item()
            n_samples += 1

        # compute accuracy
        acc = 100.0 * n_correct / n_samples
        print("Accuracy: {}%".format(acc))

        # normalize confusion matrix and set NaNs to 0
        confusion = confusion / confusion.sum(dim=0, keepdim=True)
        confusion[confusion != confusion] = 0

        # plot confusion matrix
        plt.imshow(confusion, cmap="viridis")
        plt.colorbar()
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.xticks(range(config["n_adl"]), test_dataset.adl, rotation=90)
        plt.yticks(range(config["n_adl"]), test_dataset.adl)
        plt.show()


if __name__ == "__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser(description="Visualize the WISTAR")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Path to the configuration file for visualizing the WISTAR")
    args = parser.parse_args()

    # load configuration file
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    # visualize the WISTAR
    main(config)