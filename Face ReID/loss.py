import torch
import torch.nn as nn
import src.config as config


class TripletLoss(nn.Module):
    # https://github.com/Cysu/open-reid
    # https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py
    def __init__(self, margin=2e-1, squared=False, device=config.DEVICE):
        super(TripletLoss, self).__init__()
        self.squared = squared
        self.margin = margin
        self.device = device
        self.epsilon = 1e-16

    def _pairwise_distances(self, embeddings):
        dot_product = embeddings @ embeddings.T
        squared_norm = torch.diagonal(dot_product)
        distances = squared_norm.unsqueeze(0) - 2.0 * \
            dot_product + squared_norm.unsqueeze(1)
        distances = distances.clamp(min=self.epsilon)
        if not self.squared:
            distances = torch.sqrt(distances)
        return distances

    def _get_anchor_triplet_mask(self, labels, img_type):
        assert img_type in ['positive', 'negative']
        indices_equal = torch.eye(labels.size(0)).byte().to(self.device)
        indices_not_equal = torch.logical_not(indices_equal)
        if img_type == 'positive':
            labels_eq = torch.eq(labels.unsqueeze(0),
                                 labels.unsqueeze(1))
        else:
            labels_eq = torch.ne(labels.unsqueeze(0),
                                 labels.unsqueeze(1))
        mask = torch.logical_and(indices_not_equal, labels_eq)
        return mask

    def forward(self, embeddings, labels):
        distances = self._pairwise_distances(embeddings)

        mask_positive = self._get_anchor_triplet_mask(labels, 'positive')
        anchor_positive_dist = (
            distances * mask_positive.float()
        )
        hardest_positive_dist, _ = anchor_positive_dist.max(
            dim=1, keepdim=True
        )

        mask_negative = self._get_anchor_triplet_mask(labels, 'negative')
        max_negative_dist, _ = distances.max(dim=1, keepdim=True)
        anchor_negative_dist = distances + \
            max_negative_dist * (1 - mask_negative.float())
        hardest_negative_dist, _ = anchor_negative_dist.min(
            dim=1, keepdim=True
        )

        triplet_loss = (hardest_positive_dist -
                        hardest_negative_dist + self.margin).clamp(min=0)
        triplet_loss = triplet_loss.mean()

        return triplet_loss
