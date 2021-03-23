import torch
import torch.nn as nn
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.2, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target,device):
        self.alpha.to(device)
        pred = nn.Sigmoid()(pred)
        pred = pred.view(-1,1)
        target = target.view(-1,1)

        pred = torch.cat((1-pred,pred),dim=1)
        class_mask = torch.zeros(pred.shape[0],pred.shape[1]).to(device)
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

        probs = (pred * class_mask).sum(dim=1).view(-1,1)
        probs = probs.clamp(min=0.0001,max=1.0)
        log_p = probs.log()
        alpha = torch.ones(pred.shape[0],pred.shape[1]).to(device)
        alpha[:,0] = alpha[:,0] * (1-self.alpha)
        alpha[:,1] = alpha[:,1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1,1)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return 10*loss
