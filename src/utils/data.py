import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ADLDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)
        self.transform = transform

        self.adl = list(set([f.split(".")[0].split("-")[-2] for f in self.files]))
        self.adl.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        # read file
        file = self.files[idx]
        data = np.loadtxt(self.root_dir + file)

        # convert data to tensor
        data = torch.from_numpy(data).float()
        # add batch dimension to second axis
        #data = data.unsqueeze(1)


        # get label
        label = file.split(".")[0].split("-")[-2]
        label = self.adl.index(label)

        # create one hot vector
        one_hot = torch.zeros(len(self.adl))
        one_hot[label] = 1

        return data, torch.tensor([label])
    

if __name__ == "__main__":

    dataset = ADLDataset()