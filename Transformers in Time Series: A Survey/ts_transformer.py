import torch
import torch.nn as nn
import numpy as np

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, output_size, d_model, nhead, num_layers):
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_fc = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, 
                                           num_decoder_layers=num_layers)
        self.output_fc = nn.Linear(d_model, output_size)
        
    def forward(self, src):
        src = self.input_fc(src)
        src = self.pos_encoder(src)
        output = self.transformer(src, src)
        output = self.output_fc(output)
        return output.squeeze(0)
