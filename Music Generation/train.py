import torch
from torch.utils.data import TensorDataset, DataLoader
import time
import numpy as np
from tqdm import tqdm
import config
from model import (Classical_Music_LSTM, Classical_Music_CNN,
                   Classical_Music_Transformer)
from preprocess import generate_sequences
from transformers.optimization import get_linear_schedule_with_warmup


def KL_Divergence(normal, pred_hidden):
    mu1, mu2 = normal.mean(), pred_hidden.mean()
    std1, std2 = normal.std(), pred_hidden.std()
    p = torch.distributions.Normal(mu1, std1)
    q = torch.distributions.Normal(mu2, std2)
    kld = torch.distributions.kl.kl_divergence(p, q)
    return kld.mean()


def cross_entropy_and_KL(y_pred, y_true, device=config.DEVICE,
                         pred_hidden=None, beta=1.):
    ce_loss = torch.nn.functional.cross_entropy(y_pred, y_true)
    divergence = 0
    if pred_hidden is not None:
        normal = torch.normal(
            0, 1e-1, size=pred_hidden.size(),
            device=device
        )
        # kl_divergence = torch.nn.KLDivLoss(reduction='batchmean')
        divergence = KL_Divergence(normal, pred_hidden).to(device)
    return ce_loss + beta*divergence


def get_songs_datasets(sequence_length):
    """returns pytorch tensor dataset"""
    inputs, targets = generate_sequences(sequence_length)
    dataset = TensorDataset(torch.from_numpy(inputs),
                            torch.from_numpy(targets))
    return dataset


def get_song_loader(batch_size, sequence_length, shuffle=True):
    """pytorch dataloader"""
    dataset = get_songs_datasets(sequence_length)
    song_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return song_loader


def train(output_dim, num_layers, embedding_dim, hidden_size,
          model_path, model_type, dropout, criterion, lr, epochs,
          sequence_length, num_channels, kernels, nhead, batch_size, device):
    model_types = ('Transformer', 'CNN', 'LSTM')
    assert model_type in model_types, f'model_type must be one of {", ".join(model_types)}'
    song_loader = get_song_loader(batch_size, sequence_length)
    if model_type == 'Transformer':
        model = Classical_Music_Transformer(embedding_dim, hidden_size,
                                            output_dim, num_layers, dropout,
                                            device, nhead,
                                            sequence_length).to(device)
    elif model_type == 'CNN':
        model = Classical_Music_CNN(embedding_dim, output_dim, num_channels,
                                    kernels, dropout, device,
                                    sequence_length).to(device)

    elif model_type == 'LSTM':
        model = Classical_Music_LSTM(embedding_dim, hidden_size, output_dim,
                                     num_layers, dropout, device,
                                     sequence_length).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    num_training_steps = len(song_loader) * epochs
    num_warmup_steps = 5
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    for epoch in tqdm(range(1, epochs+1), total=epochs):
        batch_losses = []
        start = time.time()
        for inputs, targets in song_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            if model_type == 'Transformer':
                output, latent = model(inputs, targets)
                loss = cross_entropy_and_KL(output, targets,
                                            pred_hidden=latent)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            batch_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f'Epoch {epoch}/{epochs},\tLoss {np.mean(batch_losses)}\
              ,\tDuration {time.time()-start}')
        torch.save(model, model_path)


if __name__ == '__main__':
    train(output_dim=config.OUTPUT_DIM, num_layers=config.NUM_LAYERS, model_type=config.MODEL_TYPE,
          model_path=config.MODEL_PATH, num_channels=config.NUM_CHANNELS, kernels=config.KERNELS,
          embedding_dim=config.EMBEDDING_DIM, hidden_size=config.HIDDEN_SIZE, nhead=config.NHEAD,
          dropout=config.DROPOUT, criterion=config.CRITERION, lr=config.LR, epochs=config.EPOCHS,
          sequence_length=config.SEQUENCE_LENGTH, batch_size=config.BATCH_SIZE, device=config.DEVICE)
