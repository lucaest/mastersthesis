from typing import Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def PositionalEncoding(length, d_model):
    PE = torch.zeros((length, d_model))
    # position of element i
    pos = torch.arange(length).unsqueeze(1)
    # pos.encode even and odd indexed values
    PE[:, 0::2] = torch.sin(
        pos / torch.pow(10000, torch.arange(0, d_model, 2)/d_model))
    PE[:, 1::2] = torch.cos(
        pos / torch.pow(10000, torch.arange(1, d_model, 2)/d_model))
    return PE

class FFNN(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FFNN, self).__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.W2(F.relu(self.W1(x)))

class Encoder(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout, d_ff):
        super(Encoder, self).__init__()
        self.MHA = MultiHeadAttention(d_model, d_k, d_v, h)
        self.FFNN = FFNN(d_model, d_ff)
        self.Normalization1 = nn.LayerNorm(d_model, eps=1e-5)
        self.Normalization2 = nn.LayerNorm(d_model, eps=1e-5)
        self.Dropout1 = nn.Dropout(p=dropout)
        self.Dropout2 = nn.Dropout(p=dropout)

    def forward(self, src, mask: Optional = None):
        # Self attention
        src2 = self.MHA(src, src, src)
        src2 = self.Dropout1(src2)
        out1 = self.Normalization1(src2 + src)
        # FFNN
        src3 = self.FFNN(out1)
        src3 = self.Dropout2(src3)
        # Add residual connection and apply layer norm.
        out2 = self.Normalization2(src3 + out1)
        return out2

class Decoder(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout, d_ff):
        super(Decoder, self).__init__()
        self.MHA = MultiHeadAttention(d_model, d_k, d_v, h)
        self.MHA_Enc_Dec = MultiHeadAttention(d_model, d_k, d_v, h)
        self.FFNN = FFNN(d_model, d_ff)
        self.Normalization1 = nn.LayerNorm(d_model, eps=1e-5)
        self.Normalization2 = nn.LayerNorm(d_model, eps=1e-5)
        self.Normalization3 = nn.LayerNorm(d_model, eps=1e-5)
        self.Dropout1 = nn.Dropout(p=dropout)
        self.Dropout2 = nn.Dropout(p=dropout)
        self.Dropout3 = nn.Dropout(p=dropout)

    def forward(self, tgt, encoder_output, look_ahead_mask):
        # Self attention
        tgt2 = self.MHA(tgt, tgt, tgt, look_ahead_mask)
        tgt2 = self.Dropout1(tgt2)
        out1 = self.Normalization1(tgt2 + tgt)
        # Encoder-decoder attention
        tgt3 = self.MHA_Enc_Dec(out1, encoder_output, encoder_output)
        tgt3 = self.Dropout2(tgt3)
        out2 = self.Normalization2(tgt3 + out1)
        # FFNN
        tgt4 = self.FFNN(out2)
        tgt4 = self.Dropout3(tgt4)
        # Add residual connection and apply layer norm.
        out3 = self.Normalization3(tgt4 + out2)
        return out3

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.WQ = nn.Linear(d_model, self.h * d_k)
        self.WK = nn.Linear(d_model, self.h * d_k)
        self.WV = nn.Linear(d_model, self.h * d_v)
        self.WO = nn.Linear(self.h * d_v, d_model)
        self.Q_K = None

    def forward(self, query, key, value, mask: Optional = None):
        # Q, K and V
        Queries = torch.cat(self.WQ(query).chunk(self.h, dim=-1), dim=0)
        Keys    = torch.cat(self.WK(key).chunk(self.h, dim=-1), dim=0)
        Values  = torch.cat(self.WV(value).chunk(self.h, dim=-1), dim=0)
        # Scaled dot product
        self.Q_K = torch.bmm(Queries, Keys.transpose(1, 2)) / np.sqrt(query.shape[1])
        # Apply look-ahead mask on Decoder input
        if mask is not None:
            self.Q_K += mask 
        # Get attention values 
        Attention = torch.bmm(F.softmax(self.Q_K, dim=-1), Values)
        # Concat. heads
        AttentionHeads = torch.cat(Attention.chunk(self.h, dim=0), dim=-1)
        # Get single attention vector per input value
        SelfAttention = self.WO(AttentionHeads)
        return SelfAttention

class Transformer(nn.Module):
    def __init__(self, d_input, d_model, d_output, h, N, dropout, d_ff):
        super(Transformer, self).__init__()
        # adopt d_k=d_v from Vaswani et. al (2017, p. 5), queries and keys have same size
        d_k = int(d_model/h)
        d_v = d_k
        self.d_model = d_model
        self.EncoderLayers = nn.ModuleList([Encoder(d_model, d_k, d_v, h, dropout=dropout, d_ff=d_ff) for _ in range(N)])
        self.DecoderLayers = nn.ModuleList([Decoder(d_model, d_k, d_v, h, dropout=dropout, d_ff=d_ff) for _ in range(N)])
        self.SrcEmbedding = nn.Linear(d_input, d_model)
        self.TgtEmbedding = nn.Linear(d_input, d_model)
        self.Linear = nn.Linear(d_model, d_output)
        self.PositionalEncoding = PositionalEncoding
        self.reset_params()

    def forward(self, src, tgt, mask:Optional = None):
        # Embed Encoder input
        Enc = self.SrcEmbedding(src)
        # Pos. encoding for Encoder input
        Enc += self.PositionalEncoding(src.shape[1], self.d_model)
        # N number of Encoders
        for layer in self.EncoderLayers:
            Enc = layer(Enc)
        # Embed Decoder Input
        Dec = self.TgtEmbedding(tgt)
        # Positional encoding for Decoder input
        Dec += self.PositionalEncoding(tgt.shape[1], self.d_model)
        # Create look-ahead-mask for Decoder input
        look_ahead_mask = self.generate_square_subsequent_mask(tgt.shape[1])
        # N number of Decoders
        for layer in self.DecoderLayers:
            Dec = layer(Dec, Enc, look_ahead_mask)
        # Linearly transform output to match d_output 
        output = self.Linear(Dec)
        return output

    # Look-ahead-Mask for Decoder
    def generate_square_subsequent_mask(self, sz):
        # Generates an upper-triangular matrix of -inf, with zeros on diag.
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask 
    
    # Initialize model params. with Xavier Uniform Distribution
    def reset_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

