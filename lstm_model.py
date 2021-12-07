import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layer_size, output_size, dropout):
        super(LSTM, self).__init__()
      
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, layer_size, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    def forward(self, x, placeholder):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.layer_size, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.layer_size, x.size(0), self.hidden_size)
        # Seperate output 
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # Linearly transform output to match output size
        out = self.linear(out)
        
        return out   


# not used 
# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, layer_size, output_size, dropout):
#         super(RNN, self).__init__()
#         self.hidden_dim = hidden_size
#         self.layer_dim = layer_size
#         self.rnn = nn.RNN(input_size, hidden_size, layer_size, batch_first=True, nonlinearity='relu', dropout=dropout)

#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):

#         h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
#         out, hn = self.rnn(x, h0.detach())
#         out = self.fc(out) 
#         return out


