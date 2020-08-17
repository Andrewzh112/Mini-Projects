from torchvision import models
from torch import nn


class CoupleFaceNet(nn.Module):
    def __init__(self, hidden_size=256, latent_size=64):
        """
        Batch Mining Inspiration:
        https://arxiv.org/pdf/1703.07737.pdf
        """
        super(CoupleFaceNet, self).__init__()
        self.feature2latent = models.resnet50(pretrained=True)
        projector = nn.Sequential(
            nn.Linear(self.feature2latent.fc.in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size)
        )
        self.feature2latent.fc = projector

    def forward(self, images):
        latents = self.feature2latent(images)
        return latents
