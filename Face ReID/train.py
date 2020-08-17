from src.loss import TripletLoss
from src.data import get_loaders, FaceData
from src.model import CoupleFaceNet
import src.config as config
from torch import optim
import torch
from tqdm import tqdm


def train(batch_size, epochs, lr, device):
    model = CoupleFaceNet()
    criterion = TripletLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    progress = tqdm(range(epochs), total=epochs)
    dataset = FaceData()
    idx2class = dataset.idx2class
    for epoch in progress:
        # randomize every epoch
        train_loader = get_loaders(batch_size=batch_size, seed=epoch)
        batch_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            targets = torch.cat(batch['target'], dim=0).to(device)
            images = torch.cat(
                [batch[idx2class[0]], batch[idx2class[1]]], dim=0
            ).to(device)
            latents = model(images)
            loss = criterion(latents, targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        print(
            f'Epoch {epoch + 1}/epochs, Training Loss {sum(batch_losses) / len(batch_losses)}'
        )


if __name__ == '__main__':
    train(config.BATCH_SIZE, config.EPOCHS, config.LR, config.DEVICE)
