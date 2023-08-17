import torch
import torch.nn as nn


class WISTAR(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers, n_adl):
        """
        WISTAR - Wrist-worn Intelligent Sensing for Activity Recognition

        Args:
            input_size (int): number of expected features in the input x
            hidden_size (int): number of features in the hidden state h
            n_layers (int): number of recurrent layers
            n_adl (int): number of expected features in the output y (number of ADLs)

        Returns:
            y (torch.tensor): output tensor of the network
        """
        
        super(WISTAR, self).__init__()

        # define network parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_adl = n_adl

        # define network layers
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_adl)
        self.dropout = nn.Dropout(p=0.25)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, x, device):
        """
        Forward pass of the network.

        Args:
            x (torch.tensor): input tensor of the network

        Returns:
            y (torch.tensor): output tensor of the network
        """        

        # initial hidden- & cell state
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(device)

        # forward propagate LSTM
        h, _ = self.lstm(x, (h0, c0))

        # only take last time step
        h = h[:, -1, :]

        # pass through fully connected layers
        h = self.fc1(h)
        h = self.dropout(h)
        h = self.relu(h)
        h = self.fc2(h)

        # apply softmax to get probabilities
        y = self.softmax(h)

        return y