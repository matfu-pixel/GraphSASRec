import torch
import torch.nn.functional as F


class TwhinLoss(torch.nn.Module):
    """
    link-prediction loss with 
    """

    def __init__(self, reg_weight):
        super(TwhinLoss, self).__init__()
        
        self._reg_weight = reg_weight
    
    def forward(self, node1, node2):
        batch_size = node1.size(0)
        num_negatives = batch_size * batch_size - batch_size
        num_positives = batch_size
        neg_weight = float(num_positives) / num_negatives

        dot_products = torch.matmul(node1, node2.t())
        logits = torch.cat([dot_products.diag(), dot_products.flatten()[1:].view(batch_size - 1, batch_size + 1)[:,:-1].flatten()])
        labels = torch.zeros_like(logits)
        labels[:batch_size] = 1.
        weights = torch.full_like(logits, neg_weight)
        weights[:batch_size] = 1.

        l2_reg = torch.mean(node1 * node1) + torch.mean(node2 * node2)

        return F.binary_cross_entropy_with_logits(logits, labels, weights, reduction="sum") / 2 / batch_size, self._reg_weight * l2_reg


class TwhinLossWithParameters(torch.nn.Module):
    """
    link-prediction loss with
    """

    def __init__(self, reg_weight=0):
        super(TwhinLossWithParameters, self).__init__()
        
        self._reg_weight = reg_weight
        self._scale = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)
        self._bias = torch.nn.Parameter(torch.tensor(0.), requires_grad=True)
    
    def forward(self, node1, node2):
        node1 = node1 / torch.sqrt((node1 ** 2).sum(dim=-1))[:, None]
        node2 = node2 / torch.sqrt((node2 ** 2).sum(dim=-1))[:, None]

        batch_size = node1.size(0)
        num_negatives = batch_size * batch_size - batch_size
        num_positives = batch_size
        neg_weight = float(num_positives) / num_negatives

        dot_products = torch.matmul(node1, node2.t())
        logits = torch.cat([dot_products.diag(), dot_products.flatten()[1:].view(batch_size - 1, batch_size + 1)[:,:-1].flatten()])
        labels = torch.zeros_like(logits)
        labels[:batch_size] = 1.
        weights = torch.full_like(logits, neg_weight)
        weights[:batch_size] = 1.

        l2_reg = torch.mean(node1 * node1) + torch.mean(node2 * node2)

        logits = self._scale * logits + self._bias

        return F.binary_cross_entropy_with_logits(logits, labels, weights, reduction="sum") / 2 / batch_size, self._reg_weight * l2_reg
