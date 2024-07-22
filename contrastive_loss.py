import torch
import torch.nn as nn
import torch.nn.functional as F


class CDCSourceAnchor(nn.Module):
    def __init__(self, temperature=0.1, scale_by_temperature=True):
        super(CDCSourceAnchor, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, feature_s, feature_t, label_s, label_t):
        """
        Input:
            feature_s: Features of the input source domain examples, shape (batch_size, hidden_dim)
            label_s: The ground truth of the input source domain examples, shape (batch_size)
            feature_t: Features of the input target domain examples, shape (pseudo_size, hidden_dim)
            label_t: The pseudo Labels of the input target domain examples, shape (pseudo_size)
        Output:
            Cross-domain contrastive loss value using source domain examples as anchors
        """
        device = (torch.device('cuda')
                  if feature_s.is_cuda and feature_t.is_cuda
                  else torch.device('cpu'))
        feature_s = F.normalize(feature_s, p=2, dim=1)
        feature_t = F.normalize(feature_t, p=2, dim=1)
        batch_size = feature_s.shape[0]
        pseudo_size = feature_t.shape[0]

        label_s = label_s.contiguous().view(-1, 1)  # (batch_size) -> (batch_size,1)
        label_t = label_t.contiguous().view(-1, 1)  # (pseudo_size) -> (pseudo_size,1)
        if label_s.shape[0] != batch_size or label_t.shape[0] != pseudo_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(label_s, label_t.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(feature_s, feature_t.T),
            self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        positives_mask = mask
        negatives_mask = 1. - mask

        num_positives_per_row = torch.sum(positives_mask, dim=1)
        denominator = torch.sum(
            exp_logits * negatives_mask, dim=1, keepdim=True) + torch.sum(
            exp_logits * positives_mask, dim=1, keepdim=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = (torch.sum(log_probs * positives_mask, dim=1)[num_positives_per_row > 0] /
                     num_positives_per_row[num_positives_per_row > 0])

        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature

        loss = loss.mean()

        return loss


class CDCTargetAnchor(nn.Module):
    def __init__(self, temperature=0.1, scale_by_temperature=True):
        super(CDCTargetAnchor, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, feature_s, feature_t, label_s, label_t):
        """
        Input:
            feature_s: Features of the input source domain examples, shape (batch_size, hidden_dim)
            label_s: The ground truth of the input source domain examples, shape (batch_size)
            feature_t: Features of the input target domain examples, shape (pseudo_size, hidden_dim)
            label_t: The pseudo Labels of the input target domain examples, shape (pseudo_size)
        Output:
            Cross-domain contrastive loss value using target domain examples as anchors
        """
        device = (torch.device('cuda')
                  if feature_s.is_cuda and feature_t.is_cuda
                  else torch.device('cpu'))
        feature_s = F.normalize(feature_s, p=2, dim=1)
        feature_t = F.normalize(feature_t, p=2, dim=1)
        batch_size = feature_s.shape[0]
        pseudo_size = feature_t.shape[0]

        label_s = label_s.contiguous().view(-1, 1)  # (batch_size) -> (batch_size,1)
        label_t = label_t.contiguous().view(-1, 1)  # (pseudo_size) -> (pseudo_size,1)
        if label_s.shape[0] != batch_size or label_t.shape[0] != pseudo_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(label_t, label_s.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(feature_t, feature_s.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        positives_mask = mask
        negatives_mask = 1. - mask
        num_positives_per_row = torch.sum(positives_mask, dim=1)
        denominator = torch.sum(
            exp_logits * negatives_mask, dim=1, keepdim=True) + torch.sum(
            exp_logits * positives_mask, dim=1, keepdim=True)
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = (torch.sum(log_probs * positives_mask, dim=1)[num_positives_per_row > 0] /
                     num_positives_per_row[num_positives_per_row > 0])
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature

        loss = loss.mean()

        return loss


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels):
        """
        Input:
            features: Features of the input example, shape (batch_size, hidden_dim)
            labels: The ground truth of each example, shape (batch_size)
            mask: The mask used for contrastive learning has a shape of (batch_size, batch_size). If the labels of examples i and j are the same, then mask_{i,j}=1
        Output:
            The value of the supervised contrastive loss within the domain
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).cuda()
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask

        num_positives_per_row = torch.sum(positives_mask, dim=1)
        denominator = torch.sum(
            exp_logits * negatives_mask, dim=1, keepdim=True) + torch.sum(
            exp_logits * positives_mask, dim=1, keepdim=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(
            log_probs * positives_mask, dim=1)[num_positives_per_row > 0] / num_positives_per_row[
                        num_positives_per_row > 0]
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature

        loss = loss.mean()

        return loss
