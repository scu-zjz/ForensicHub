import torch
import torchvision
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean', ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCELoss(reduction='none')

    def forward(self, logits, label):
        # compute loss
        logits = logits.float()  # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label)
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

class Loss_Function(nn.Module):
    def __init__(self):
        super(Loss_Function, self).__init__()
        #self.bceloss = nn.BCELoss()
        #self.focalloss = FocalLoss()
        self.ce = nn.CrossEntropyLoss()
    def forward(self, logits, label):
        #bce_loss = self.bceloss(logits,label)
        #focal_loss = self.focalloss(logits,label)
        ce_loss = self.ce(logits,label)
        return ce_loss