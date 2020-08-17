from src.loss import TripletLoss
from src.data import get_loaders, FaceData
from src.model import CoupleFaceNet
import src.config as config
from torch import optim
import torch
import os
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)


def train(batch_size, epochs, lr, device):
    model = CoupleFaceNet().to(device)
    criterion = TripletLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    dataset = FaceData()
    idx2class = dataset.idx2class
    model.train()
    for epoch in tqdm(range(epochs), total=epochs):
        # randomize every epoch
        train_loader = get_loaders(batch_size=batch_size, seed=epoch)
        batch_losses = 0
        progress_bar = tqdm(train_loader,
                            desc=f'Epoch {epoch + 1}/{epochs}',
                            leave=False,
                            disable=False)
        for batch in progress_bar:
            optimizer.zero_grad()
            targets = torch.cat(batch['target'], dim=0).to(device)
            images = torch.cat(
                [batch[idx2class[0]], batch[idx2class[1]]], dim=0
            ).to(device)
            latents = model(images)
            loss = criterion(latents, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            batch_losses += loss.item()
            progress_bar.set_postfix(
                {'Train Loss': '{:.3f}'.format(loss.item()/len(batch))}
            )

        if not os.path.exists('model_weights'):
            os.mkdir('model_weights') 
        torch.save(model.state_dict(), os.path.join('model_weights', 'face_reid.pt'))
        tqdm.write(f'\nEpoch {epoch + 1}/{epochs}')
        loss_train_avg = batch_losses / len(train_loader)
        tqdm.write(f'Training loss: {loss_train_avg}')


if __name__ == '__main__':
    train(config.BATCH_SIZE, config.EPOCHS, config.LR, config.DEVICE)
