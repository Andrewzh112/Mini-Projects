import torch


class TripletLoss:
    def __init__(self, margin=2e-1, squared=False):
        self.squared = squared
        self.margin = margin

    def _pairwise_distances(self, embeddings):
        epsilon = 1e-16
        dot_product = embeddings @ embeddings.T
        squared_norm = torch.diagonal(dot_product)
        distances = squared_norm.unsqueeze(0) - 2.0 * \
            dot_product + squared_norm.unsqueeze(1)
        distances.clamp_(min=epsilon)
        if not self.squared:
            mask = torch.eq(distances, 0).float()
            distances += mask * epsilon
            distances = torch.sqrt(distances)
            distances *= (1-mask)
        return distances

    def _get_anchor_triplet_mask(self, labels, img_type):
        assert img_type in ['positive', 'negative']
        indices_equal = torch.eye(labels.size(0)).byte()
        indices_not_equal = torch.logical_not(indices_equal)
        if img_type == 'positive':
            labels_eq = torch.eq(labels.unsqueeze(0),
                                 labels.unsqueeze(1))
        else:
            labels_eq = torch.ne(labels.unsqueeze(0),
                                 labels.unsqueeze(1))
        mask = torch.logical_and(indices_not_equal, labels_eq)
        return mask

    def __call__(self, embeddings, labels):
        distances = self._pairwise_distances(embeddings)

        mask_positive = self._get_anchor_triplet_mask(labels, 'positive')
        hardest_positive_dist = (
            distances * mask_positive.float()
        ).max(dim=1)[0]

        mask_negative = self._get_anchor_triplet_mask(labels, 'negative')
        max_negative_dist = distances.max(dim=1, keepdim=True)[0]
        distances = distances + max_negative_dist * (~mask_negative).float()
        hardest_negative_dist = distances.min(dim=1)[0]

        triplet_loss = (hardest_positive_dist -
                        hardest_negative_dist + self.margin).clamp(min=0)
        triplet_loss = triplet_loss.mean()

        return triplet_loss
