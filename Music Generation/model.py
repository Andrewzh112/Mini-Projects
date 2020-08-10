import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from collections import deque
import numpy as np
import math


class Classical_Music_Generator:
    def generate_melody(self, mapping, seed, num_steps,
                        temperature, dropout_rounds=6, topk=10):
        melody = seed.split()
        seed = self._start_symbols + seed.split()
        seed = deque([mapping[symbol]
                      for symbol in seed], maxlen=self.sequence_length)
        for _ in range(num_steps):
            seed_tensor = torch.tensor(seed).type(torch.LongTensor)\
                .to(self.device)
            if self.model_type == 'CNN':
                seed_tensor = seed_tensor.T
            if self.model_type == 'Transformer':
                # stochastic generation
                output = self._generate_stochastic(seed_tensor, dropout_rounds)
            else:
                output = self.forward(seed_tensor)[0]
            output, _ = torch.topk(output, topk)
            output_int = self._sample_with_temperature(output, temperature)
            seed.append(output_int)
            output_symbol = [key for key,
                             val in mapping.items() if val == output_int][0]
            if output_symbol == self.stop_symbol:
                break
            melody.append(output_symbol)
        return melody

    def _sample_with_temperature(self, logits, temperature):
        exp = logits / temperature
        weights = F.softmax(exp, dim=0).detach().cpu().numpy()
        choices = range(len(weights))
        index = np.random.choice(choices, 1, True, weights)
        return index

    def _generate_stochastic(self, seed_tensor, dropout_rounds):
        self.train()
        outputs, _, _ = self.forward(seed_tensor)
        output = outputs[0]
        for _ in range(dropout_rounds):
            additional_outputs, _, _ = self.forward(seed_tensor)
            output = torch.add(output, additional_outputs[0])
        output /= (dropout_rounds+1)
        return output


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
        out = self.fc(lstm_out[:, -1, :])
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
                                    nn.Conv1d(in_channels=embedding_size,
                                              out_channels=num_channels,
                                              kernel_size=kernel),
                                    nn.ReLU(),
                                    nn.AvgPool1d(sequence_length - kernel+1)) for kernel in kernel_size])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_channels*len(kernel_size), output_size)

        self.stop_symbol = '/'
        self._start_symbols = [self.stop_symbol] * sequence_length

    def forward(self, x):
        embedded_notes = self.embeddings(x.T).permute(1, 2, 0)
        conv_outs = [conv(embedded_notes).squeeze(-1) for conv in self.convs]
        all_out = torch.cat(conv_outs, dim=1)
        final_feature_map = self.dropout(all_out)
        final_out = self.fc(final_feature_map)
        return final_out


class Classical_Music_Transformer(nn.Module, Classical_Music_Generator):
    def __init__(self, embedding_size, hidden_size, output_size,
                 num_layers, dropout, device, nhead, sequence_length,
                 latent_size=8):

        super(Classical_Music_Transformer, self).__init__()

        self.model_type = 'Transformer'
        self.hidden_size = hidden_size
        self.device = device
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size

        self.pos_encoder = PositionalEncoding(embedding_size, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_size, nhead,
                                                 hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      num_layers)
        self.encoder = nn.Embedding(output_size, embedding_size)
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.decoder = nn.Linear(embedding_size, output_size)

        self.latent_size = latent_size
        self.next_node_projection = nn.Sequential(
            nn.Linear(embedding_size, embedding_size//2),
            nn.ReLU(),
            nn.Linear(embedding_size//2, latent_size*2)
        )
        self.expand = nn.Sequential(
            nn.Linear(latent_size, embedding_size//2),
            nn.ReLU(),
            nn.Linear(embedding_size//2, embedding_size)
        )
        self.stop_symbol = '/'
        self._start_symbols = [self.stop_symbol] * sequence_length
        self.init_weights()

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.next_node_projection[0].bias.data.zero_()
        self.next_node_projection[0].weight.data.uniform_(
            -initrange, initrange
        )
        self.next_node_projection[2].bias.data.zero_()
        self.next_node_projection[2].weight.data.uniform_(
            -initrange, initrange
        )
        self.expand[0].bias.data.zero_()
        self.expand[0].weight.data.uniform_(-initrange, initrange)
        self.expand[2].bias.data.zero_()
        self.expand[2].weight.data.uniform_(-initrange, initrange)

    def forward(self, src, next_notes=None):
        batch_size = src.size(0)
        if next_notes is None:
            sample = False
        else:
            sample = np.random.choice([True, False], p=[0.6, 0.4])
        src = self.encoder(src) * torch.tensor(math.sqrt(self.embedding_size))
        src = self.pos_encoder(src)
        encoded_notes = self.transformer_encoder(src)
        x, _ = torch.max(encoded_notes, dim=1)
        if sample:
            encoded_next_notes = self.encoder(next_notes)
            mu_logvar = self.next_node_projection(
                encoded_next_notes).view(-1, 2, self.latent_size)
            mu = mu_logvar[:, 0, :]
            logvar = mu_logvar[:, 1, :]
            latent = self.reparameterize(mu, logvar)
        else:
            mu = logvar = torch.zeros(batch_size, self.latent_size, device=self.device)
            latent = self.reparameterize(mu, logvar)
        variation = self.expand(latent)
        x = torch.add(x, variation)
        output = self.decoder(x)
        return output, mu, logvar


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
