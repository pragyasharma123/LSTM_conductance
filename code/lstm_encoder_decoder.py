# Author: Laura Kulowski

import numpy as np
import random
import os, errno
import sys
from tqdm import trange

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size, num_layers = 3):
        
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer with dropout
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.2)

    def forward(self, x_input):
        
        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence 
        '''
        
        lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))
        
        return lstm_out, self.hidden     
    
    def init_hidden(self, batch_size):
        
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state 
        '''
        
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''
    
    def __init__(self, input_size, hidden_size, num_layers = 3):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, dropout=0.2)
        self.linear = nn.Linear(hidden_size, input_size)           

    def forward(self, x_input, encoder_hidden_states):
        
        '''        
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence 
 
        '''
        
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))     
        
        return output, self.hidden

class lstm_seq2seq(nn.Module):
    ''' Train LSTM encoder-decoder and make predictions '''

    def __init__(self, input_size, hidden_size, device=torch.device("cuda")):
        '''
        : param input_size:  the number of expected features in the input X
        : param hidden_size: the number of features in the hidden state h
        : param device:      device to use (e.g., 'cuda' or 'cpu')
        '''
        super(lstm_seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device  # Assign the device

        self.encoder = lstm_encoder(input_size=input_size, hidden_size=hidden_size).to(device)
        self.decoder = lstm_decoder(input_size=input_size, hidden_size=hidden_size).to(device)


    '''
    def train_model(self, input_tensor, target_tensor, n_epochs, target_len, batch_size, training_prediction, teacher_forcing_ratio, learning_rate, dynamic_tf):
        # initialize array of losses 
        losses = np.full(n_epochs, np.nan)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # calculate number of batch iterations
        n_batches = int(input_tensor.shape[1] / batch_size)

        with trange(n_epochs) as tr:
            for it in tr:
            
                batch_loss = 0.0

                for b in range(n_batches):
                    # select data 
                    input_batch = input_tensor[:, b: b + batch_size, :].to(self.device)
                    target_batch = target_tensor[:, b: b + batch_size, :].to(self.device)

                    # outputs tensor
                    outputs = torch.zeros(target_len, batch_size, input_batch.shape[2], device=self.device)

                    # initialize hidden state
                    encoder_hidden = self.encoder.init_hidden(batch_size)
                    encoder_hidden = (encoder_hidden[0].to(self.device), encoder_hidden[1].to(self.device))

                    # zero the gradient
                    optimizer.zero_grad()

                    # encoder outputs
                    encoder_output, encoder_hidden = self.encoder(input_batch)

                    # decoder with teacher forcing
                    decoder_input = input_batch[-1, :, :]  # shape: (batch_size, input_size)
                    decoder_hidden = encoder_hidden

                    for t in range(target_len):
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        outputs[t] = decoder_output

                        # predict with teacher forcing
                        if random.random() < teacher_forcing_ratio:
                            decoder_input = target_batch[t, :, :]
                        else:
                            decoder_input = decoder_output

                    # compute the loss 
                    loss = criterion(outputs, target_batch)
                    batch_loss += loss.item()

                    # backpropagation
                    loss.backward()
                    optimizer.step()

                # loss for epoch 
                batch_loss /= n_batches
                losses[it] = batch_loss

                # dynamic teacher forcing
                if dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio = teacher_forcing_ratio - 0.02

                # progress bar
                tr.set_postfix(loss="{0:.3f}".format(batch_loss))

        return losses

    def predict(self, input_tensor, target_len):
        
        
        : param input_tensor:      input data (seq_len, input_size); PyTorch tensor 
        : param target_len:        number of target values to predict 
        : return np_outputs:       np.array containing predicted values; prediction done recursively 
        

        # encode input_tensor
        input_tensor = input_tensor.unsqueeze(1)     # add in batch size of 1
        encoder_output, encoder_hidden = self.encoder(input_tensor)

        # initialize tensor for predictions
        outputs = torch.zeros(target_len, input_tensor.shape[2])

        # decode input_tensor
        decoder_input = input_tensor[-1, :, :]
        decoder_hidden = encoder_hidden
        
        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output.squeeze(0)
            decoder_input = decoder_output
            
        np_outputs = outputs.detach().numpy()
        
        return np_outputs
'''

    def train_model_with_validation(
        self, input_tensor, target_tensor, val_input_tensor, val_target_tensor, n_epochs, target_len, batch_size,
        training_prediction='mixed_teacher_forcing', teacher_forcing_ratio=0.5, learning_rate=0.001, dynamic_tf=False
):
        losses, val_losses = [], []

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        n_batches = input_tensor.shape[1] // batch_size
        with trange(n_epochs) as tr:
            for epoch in tr:
                batch_loss = 0.0
                for b in range(n_batches):
                    # Training
                    input_batch = input_tensor[:, b: b + batch_size, :].to(self.device)
                    target_batch = target_tensor[:, b: b + batch_size, :].to(self.device)

                    outputs = torch.zeros(target_len, batch_size, input_batch.shape[2], device=self.device)
                    encoder_hidden = self.encoder.init_hidden(batch_size)
                    encoder_hidden = (encoder_hidden[0].to(self.device), encoder_hidden[1].to(self.device))

                    optimizer.zero_grad()
                    encoder_output, encoder_hidden = self.encoder(input_batch)
                    decoder_input = input_batch[-1, :, :]
                    decoder_hidden = encoder_hidden

                    for t in range(target_len):
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        outputs[t] = decoder_output
                        if random.random() < teacher_forcing_ratio:
                            decoder_input = target_batch[t, :, :]
                        else:
                            decoder_input = decoder_output

                    loss = criterion(outputs, target_batch)
                    batch_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                batch_loss /= n_batches
                losses.append(batch_loss)

            # Validation Loss
                with torch.no_grad():
                    val_outputs = torch.zeros(
                        target_len, val_input_tensor.size(1), val_input_tensor.size(2), device=self.device
                )
                    val_encoder_hidden = self.encoder.init_hidden(val_input_tensor.size(1))
                    val_encoder_hidden = (
                        val_encoder_hidden[0].to(self.device), val_encoder_hidden[1].to(self.device)
                )
                    val_encoder_output, val_encoder_hidden = self.encoder(val_input_tensor)
                    val_decoder_input = val_input_tensor[-1, :, :]
                    val_decoder_hidden = val_encoder_hidden

                    for t in range(target_len):
                        val_decoder_output, val_decoder_hidden = self.decoder(val_decoder_input, val_decoder_hidden)
                        val_outputs[t] = val_decoder_output
                        val_decoder_input = val_decoder_output

                    val_loss = criterion(val_outputs, val_target_tensor)
                    val_losses.append(val_loss.item())

                tr.set_postfix(train_loss=batch_loss, val_loss=val_loss.item())
                if dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio -= 0.02

        return losses, val_losses