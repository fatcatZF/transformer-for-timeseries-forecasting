import math

import torch
import torch.nn as nn
import torch.nn.functional as F



"""Positional Encoding"""

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len , d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer("pe", pe)
        
    def forward(self, x):
        """
        args: [seq_len, n_batch, d_model]
        """
        x = x+self.pe[:x.size(0), :]
        return self.dropout(x)


"""Time series encoder"""

class TimeSeriesEncoder(nn.Module):
    def __init__(self, n_in, d_model, dim_feedforward, nhead, num_enlayers, dropout, 
                max_len):
        super(TimeSeriesEncoder, self).__init__()
        self.fc_in = nn.Linear(n_in, d_model)
        self.pe = PositionalEncoding(d_model, dropout, max_len)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, 
                                                       dim_feedforward=dim_feedforward,
                                                       nhead=nhead, dropout=dropout), 
                                            num_layers=num_enlayers)
    def forward(self, x):
        """
        args: x, shape: [seq_len, n_batch, n_in]
        """
        x = self.fc_in(x)
        x = self.pe(x)
        return self.encoder(x)
        

"""Time series decoder"""

class TimeSeriesDecoder(nn.Module):
    def __init__(self, n_in, d_model, dim_feedforward, nhead, num_delayers, dropout):
        super(TimeSeriesDecoder, self).__init__()
        self.fc_in = nn.Linear(n_in, d_model)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=d_model, 
                                                       dim_feedforward=dim_feedforward,
                                                       nhead=nhead, dropout=dropout), 
                                             num_layers=num_delayers)
        self.fc_out = nn.Linear(d_model, n_in)
    
    def forward(self, x, memory):
        """
        args:
          x: current time step, shape: [1,n_batch,n_in]
          memory: previous memory, shape: [seq_len, n_batch, d_model]
        """
        x = self.fc_in(x)
        x = self.decoder(tgt=x, memory=memory)
        return self.fc_out(x)


