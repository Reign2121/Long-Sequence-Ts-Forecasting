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


''' input_fc : 입력을 d_model 차원으로 매핑하는 fully connected layer
    pos_encoder : position encoding을 적용하는 모듈.
    transformer : Transformer 모델. nn.Transformer를 사용하여 구현함.
    output_fc : Transformer의 출력을 output_size 크기의 벡터로 매핑하는 fully connected layer
    
    forward 메소드는 다음과 같은 작업을 수행합니다:
    src : shape이 (sequence_length, batch_size, input_size) 인 인풋 시퀀스
    input_fc 를 사용하여 입력을 d_model 차원으로 매핑함.
    pos_encoder를 사용하여 position encoding을 적용함.
    transformer를 사용하여 시계열 데이터를 학습.
    output_fc를 사용하여 Transformer의 출력을 output_size 크기의 벡터로 매핑함.
    output을 반환하기 전에 squeeze 를 사용하여 차원을 축소함. '''
