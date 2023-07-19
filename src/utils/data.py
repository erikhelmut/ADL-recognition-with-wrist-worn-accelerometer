import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ADLDataset(Dataset):

    def __init__(self, files, adl, transform=None):
        self.files = files
        self.adl = adl

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # read file
        file = self.files[idx]
        data = np.loadtxt(file)

        # convert data to tensor
        data = torch.from_numpy(data).float()

        # get label
        label = file.split(".")[-2].split("-")[-2]
        label = self.adl.index(label)

        return data, torch.tensor([label])


def init_datasets(root_dir, train_split=0.6, val_split=0.2, seed=0):
    """Initialize datasets for training, validation and testing."""
    assert train_split + val_split < 1.0, "train_split + val_split must be less than 1.0"

    # get all files
    files = []

    folders = os.listdir(root_dir)
    for folder in folders:
        if os.path.isdir(root_dir + folder):
            # check if folder ends with "_MODEL", if so, skip
            if folder.endswith("_MODEL"):
                continue
            for file in os.listdir(root_dir + folder):
                files.append(root_dir + folder + "/" + file)
    print("Found {} files".format(len(files)))

    adl = list(set([f.split(".")[-2].split("-")[-2] for f in files]))
    adl.sort()
    print("Found {} ADLs".format(len(adl)))

    # split files into train, val and test
    np.random.seed(seed)
    np.random.shuffle(files)

    train_files = files[:int(train_split * len(files))]
    val_files = files[int(train_split * len(files)):int((train_split + val_split) * len(files))]
    test_files = files[int((train_split + val_split) * len(files)):]
    print("Split files into {} train, {} val and {} test files.".format(len(train_files), len(val_files),
                                                                        len(test_files)))

    # create datasets
    train_dataset = ADLDataset(files=train_files, adl=adl)
    val_dataset = ADLDataset(files=val_files, adl=adl)
    test_dataset = ADLDataset(files=test_files, adl=adl)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = init_datasets("../../HMP_Dataset/", train_split=0.6, val_split=0.2,
                                                             seed=42)
