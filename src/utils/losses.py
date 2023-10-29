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
