from torchvision import models
from torch import nn


class CoupleFaceNet(nn.Module):
    def __init__(self, hidden_size=256, latent_size=64, pretrained=True):
        """
        Batch Mining Inspiration:
        https://arxiv.org/pdf/1703.07737.pdf
        """
        super(CoupleFaceNet, self).__init__()
        self.feature2latent = models.resnet50(pretrained=pretrained)
        projector = nn.Sequential(
            nn.Linear(self.feature2latent.fc.in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size)
        )
        self.feature2latent.fc = projector
        if not pretrained:
            self.reset_params()

    def forward(self, images):
        latents = self.feature2latent(images)
        return latents

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
