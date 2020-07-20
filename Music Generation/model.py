import torch
from torch import nn
from torch.cuda import amp
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from collections import deque
import numpy as np
import math
from tensorflow.keras.utils import to_categorical



class Classical_Music_Generator:
    def generate_melody(self, mapping, seed, num_steps, temperature):
        melody = seed.split()
        seed = self._start_symbols + seed.split()
        seed = deque([mapping[symbol] for symbol in seed],maxlen=self.sequence_length)

        for _ in range(num_steps):
            onehot_seed = to_categorical(seed, num_classes=len(mapping))
            seed_tensor = torch.from_numpy(onehot_seed).type(torch.LongTensor)\
                                                        .to(self.device)
            if self.model_type == 'CNN':
                seed_tensor = seed_tensor.T
            probas = F.softmax(self.forward(seed_tensor)[0],dim=0)
            output_int = self._sample_with_temperature(probas, temperature)
            seed.append(output_int)

            output_symbol = [key for key,val in mapping.items() if val==output_int][0]
            if output_symbol == self.stop_symbol:
                break
            
            melody.append(output_symbol)
        
        return melody


    def _sample_with_temperature(self, probas, temperature):
        predictions = torch.log(probas) / temperature
        weights = F.softmax(predictions, dim=0).detach().cpu().numpy()
        choices = range(len(weights))
        index = np.random.choice(choices, 1, True, weights)
        return index


class Classical_Music_LSTM(nn.Module, Classical_Music_Generator):
    def __init__(self, embedding_size, hidden_size, output_size,
                 num_layers, dropout, device, sequence_length, birectional=True):

        super(Classical_Music_LSTM, self).__init__()

        self.model_type = 'LSTM'
        self.hidden_size = hidden_size
        self.device = device
        self.sequence_length = sequence_length

        self.embed = nn.Embedding(output_size, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, 
                            bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*(birectional*2), output_size)
        

        self.stop_symbol = '/'
        self._start_symbols = [self.stop_symbol] * sequence_length


    def forward(self, x):
        outputs = self.embed(x)
        outputs = F.relu(outputs)
        lstm_out, _ = self.lstm(outputs)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out[:,-1,:])
        return out


class Classical_Music_CNN(nn.Module, Classical_Music_Generator):
    def __init__(self, embedding_size, output_size, num_channels, kernel_size, 
                dropout, device, sequence_length):

        super(Classical_Music_CNN, self).__init__()

        self.model_type = 'CNN'
        self.device = device
        self.sequence_length = sequence_length
        
        self.embeddings = nn.Embedding(output_size, embedding_size)
        self.convs = nn.ModuleList([nn.Sequential(
                                    nn.Conv1d(in_channels=embedding_size, out_channels=num_channels, kernel_size=kernel),
                                    nn.ReLU(),
                                    nn.MaxPool1d(sequence_length - kernel+1)) for kernel in kernel_size])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_channels*len(kernel_size), output_size)

        self.stop_symbol = '/'
        self._start_symbols = [self.stop_symbol] * sequence_length


    def forward(self, x):
        embedded_notes = self.embeddings(x.T).permute(1,2,0)
        conv_outs = [conv(embedded_notes).squeeze(-1) for conv in self.convs]
        all_out = torch.cat(conv_outs, dim=1)
        final_feature_map = self.dropout(all_out)
        final_out = self.fc(final_feature_map)
        return final_out


class Classical_Music_Transformer(nn.Module, Classical_Music_Generator):
    def __init__(self, embedding_size, hidden_size, output_size,
                 num_layers, dropout, device, nhead, sequence_length):

        super(Classical_Music_Transformer, self).__init__()

        self.model_type = 'Transformer'
        self.hidden_size = hidden_size
        self.device = device
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size

        self.pos_encoder = PositionalEncoding(embedding_size, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_size, nhead, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Embedding(output_size, embedding_size)
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.decoder = nn.Linear(embedding_size*sequence_length, output_size)
        
        self.stop_symbol = '/'
        self._start_symbols = [self.stop_symbol] * sequence_length
        self.init_weights()


    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def forward(self, src):
        src = self.encoder(src) * torch.tensor(math.sqrt(self.embedding_size))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src).view(-1, self.embedding_size*self.sequence_length)
        output = self.decoder(output)
        return output

    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)