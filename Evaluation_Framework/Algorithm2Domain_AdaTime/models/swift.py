#################################################################################################################################
######### Based on SWIFT by                                                                                                    ##
# Annapragada AV, Greenstein JL, Bose SN, Winters BD, Sarma SV, Winslow RL.                                                    ##
# SWIFT: A deep learning approach to prediction of hypoxemic events in critically-Ill patients using SpO2 waveform prediction. ##
# Althouse B, editor. PLoS Comput Biol. 2021 Dec 21;17(12):e1009712.                                                           ##
##################################################################################################################################
##################################################################################################################################
# translated from keras to pytorch and added some flexibility (more optional layers) #############################################
##################################################################################################################################

import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, configs):
        super(LSTM, self).__init__()

        self.input_size = configs.input_channels
        self.bidirectional = configs.bidirectional_lstm
        self.layers = configs.lstm_layers
        self.output_size = configs.features_len * configs.final_out_channels
        self.dropout = configs.dropout

        self.batch_norm = nn.BatchNorm1d(self.input_size)

        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        self.lstm1 = nn.LSTM(self.input_size, self.layers[0], batch_first=True, bidirectional=self.bidirectional)
        self.dropout1 = nn.Dropout(self.dropout)
        if self.bidirectional:
            i = 2
        else:
            i = 1

        for layer in range(1, len(self.layers)):
            lstm_layer = nn.LSTM(self.layers[layer - 1] * i, self.layers[layer], batch_first=True, bidirectional=self.bidirectional)
            self.lstm_layers.append(lstm_layer)
            self.dropout_layers.append(nn.Dropout(self.dropout))

        self.lstm_last = nn.LSTM(self.layers[-1] * i, self.output_size, batch_first=True, bidirectional=self.bidirectional)
        self.final_layer = nn.Linear(self.output_size * 2, self.output_size)


    def forward(self, x):
        # (batch_size, input_size, sequence_length) =>  (batch_size, sequence_length, input_size)
        x = x.transpose(2, 1)
        
        # Apply batch normalization
        # Note: BatchNorm1d expects (batch_size, features) or (batch_size, features, length)
        # We need to transpose for batch norm, then transpose back
        if x.dim() == 3:
            x = x.transpose(1, 2)  # (batch_size, input_size, seq_len)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)  # (batch_size, seq_len, input_size)
        else:
            x = self.batch_norm(x)
        
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        for i in range(len(self.lstm_layers)):
            lstm_out, _ = self.lstm_layers[i](lstm1_out)
            lstm_out = self.dropout_layers[i](lstm_out)
            lstm1_out = lstm_out

        lstm1_out, _ = self.lstm_last(lstm1_out)

        if self.bidirectional:
            last_output = self.final_layer(lstm1_out[:, -1, :])
        else:
            last_output = lstm1_out[:, -1, :]

        return last_output