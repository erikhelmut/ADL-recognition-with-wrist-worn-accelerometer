import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ADLDataset(Dataset):

    def __init__(self, files, adl):
        """
        Dataset class for the WISTAR dataset. 
            This class is used to load the data from the files and convert it to tensors.
            Each file contains the data of one ADL. The data is loaded from the file and converted to a tensor
            of shape (n_samples, n_features). The label is extracted from the filename and converted to a tensor
            of shape (1).

        References:
            - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

        Args:
            files (list): list of files to be loaded
            adl (list): list of ADLs

        Returns:
            None
        """

        self.files = files
        self.adl = adl


    def __len__(self):
        """
        Returns the number of files in the dataset.
        
        Args:
            None

        Returns:
            len (int): total number of files in the dataset
        """

        return len(self.files)


    def __getitem__(self, idx):
        """
        Returns the data and label for a given index.

        Args:
            idx (int): index of the file to be loaded

        Returns:
            data (torch.tensor): data tensor of shape (n_samples, n_features)
            label (torch.tensor): label tensor of shape (1)
        """
        
        # read file
        file = self.files[idx]
        data = np.loadtxt(file)

        # convert data to tensor
        data = torch.from_numpy(data).float()

        # get label
        label = file.split(".")[-2].split("-")[-2]
        label = self.adl.index(label)

        return data, torch.tensor([label])


def init_datasets(root_dir, train_split=0.6, val_split=0.2, seed=42):
    """
    Initialize datasets for training, validation and testing.
    
    Args:
        root_dir (str): path to the root directory of the dataset
        train_split (float): fraction of the dataset to be used for training
        val_split (float): fraction of the dataset to be used for validation
        seed (int): random seed for reproducibility

    Returns:
        train_dataset (ADLDataset): dataset for training
        val_dataset (ADLDataset): dataset for validation
        test_dataset (ADLDataset): dataset for testing
    """
    # check if train_split + val_split < 1.0
    assert train_split + val_split < 1.0, "train_split + val_split must be less than 1.0"

    # get all files
    samples = dict()

    # get all folders
    folders = os.listdir(root_dir)

    num_samples = 0
    # loop over all folders
    for folder in folders:
        # check if folder is a directory
        if os.path.isdir(root_dir + folder):
            # check if folder ends with "_MODEL", if so, skip
            if folder.endswith("_MODEL"):
                continue
            #create empty list for files
            samples[folder] = []
            # loop over all files in folder and append to list
            for file in os.listdir(root_dir + folder):
                samples[folder].append(root_dir + folder + "/" + file)
                num_samples += 1

    print("Found {} files".format(num_samples))

    # get all ADLs and sort them
    # adl = list(set([f.split(".")[-2].split("-")[-2] for f in files])) --> old version
    adl = list(samples.keys())
    adl.sort()

    print("Found {} ADLs".format(len(adl)))
    # split files into train, val and test
    train_files = []
    val_files = []
    test_files = []

    for curr_adl in adl:
        # get all files for current adl
        files = samples[curr_adl]
        # shuffle files
        np.random.seed(seed)
        np.random.shuffle(files)
        # split files
        assert (len(files) >= 3), "Less than three files for ADL {}".format(curr_adl)
        # make shure that each dataset includes at least one sample of each folder
        train_files.append(files[0])
        val_files.append(files[1])
        test_files.append(files[2])
        # split remaining files
        files = files[3:]
        train_files += files[:int(train_split * len(files))]
        val_files += files[int(train_split * len(files)):int((train_split + val_split) * len(files))]
        test_files += files[int((train_split + val_split) * len(files)):]

    print("Split files into {} train, {} val and {} test files.\n".format(len(train_files), len(val_files),
                                                                        len(test_files)))

    # make all ADLs lowercase
    adl = [a.lower() for a in adl]

    # create datasets for train, val and test
    train_dataset = ADLDataset(files=train_files, adl=adl)
    val_dataset = ADLDataset(files=val_files, adl=adl)
    test_dataset = ADLDataset(files=test_files, adl=adl)

    return train_dataset, val_dataset, test_dataset