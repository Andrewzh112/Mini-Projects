import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import time
import json
import numpy as np
from tqdm import tqdm
import config
from model import Classical_Music
from preprocess import generate_sequences



def get_songs_datasets(sequence_length):
      inputs, targets = generate_sequences(sequence_length)
      dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
      return dataset


def get_song_loader(batch_size, sequence_length, shuffle=True):
      dataset = get_songs_datasets(sequence_length)
      song_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
      return song_loader


def train(output_dim, num_layers, embedding_dim, hidden_size, model_path,
          dropout, criterion, lr, epochs, sequence_length, batch_size, device):
      
      song_loader = get_song_loader(batch_size, sequence_length)
      model = Classical_Music(embedding_size=embedding_dim, hidden_size=hidden_size, 
                              output_size=output_dim, num_layers=num_layers, 
                              sequence_length=sequence_length, dropout=dropout, device=device).to(device)
      
      optimizer = torch.optim.Adam(model.parameters(), lr)

      for epoch in tqdm(range(1, epochs+1), total=epochs):
            batch_losses = []
            start = time.time()
            for inputs, targets in song_loader:
                  inputs, targets = inputs.to(device), targets.to(device)
                  optimizer.zero_grad()

                  outputs = model(inputs)
                  loss = criterion(outputs, targets)
                  batch_losses.append(loss.item())

                  loss.backward()
                  optimizer.step()
                  
            print(f'Epoch {epoch}/{epochs},\tLoss {np.mean(batch_losses)},\tDuration {time.time()-start}')
            torch.save(model, f'{model_path}')

            
if __name__ == '__main__':
      train(output_dim=config.OUTPUT_DIM, num_layers=config.NUM_LAYERS, model_path=config.MODEL_PATH,
          embedding_dim=config.EMBEDDING_DIM, hidden_size=config.HIDDEN_SIZE, 
          dropout=config.DROPOUT, criterion=config.CRITERION, lr=config.LR, epochs=config.EPOCHS,
          sequence_length=config.SEQUENCE_LENGTH, batch_size=config.BATCH_SIZE, device=config.DEVICE)