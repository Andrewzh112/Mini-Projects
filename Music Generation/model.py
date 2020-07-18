import torch
from torch import nn
from torch.cuda import amp
from torch.nn import functional as F
from collections import deque
import numpy as np
from tensorflow.keras.utils import to_categorical



class Classical_Music(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size,
                 num_layers, dropout, device, sequence_length, birectional=True):

        super(Classical_Music, self).__init__()


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
    

    def generate_melody(self, mapping, seed, num_steps, temperature):
        melody = seed.split()
        seed = self._start_symbols + seed.split()
        seed = deque([mapping[symbol] for symbol in seed],maxlen=self.sequence_length)

        for _ in range(num_steps):
            onehot_seed = to_categorical(seed, num_classes=len(mapping))
            seed_tensor = torch.from_numpy(onehot_seed).type(torch.LongTensor)\
                                                        .to(self.device)
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