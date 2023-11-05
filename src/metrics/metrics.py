import torch
from metrics.base_metrics import BaseMetric, TopKMetric


class _MAP(TopKMetric):
    """
    Mean Average Precision
    input: tensor, tensor
    shape: (N,), (N,)
    """

    def __init__(self, k=None):
        super().__init__(k)

    def forward(self, y_true, y_score):
        assert self._k is not None, "@k is not set!"

        sort_true = y_true[torch.argsort(y_score, descending=True)][:self._k]
        sort_true[sort_true != 0] = 1 # all relevant objs in MAP is equivalent
        precisions = sort_true.cumsum(dim=0) / torch.arange(1, sort_true.shape[0] + 1)

        return (precisions * sort_true).sum().item() / max(self._eps, sort_true.sum().item())


class _MRR(BaseMetric):
    """
    Mean Reciprocal Rank
    input: tensor, tensor
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_score):
        sort_true = y_true[torch.argsort(y_score, descending=True, stable=True)]
        first_rel_ind = sort_true.argmax(dim=0).item()

        return 1 / (first_rel_ind + 1) if sort_true[first_rel_ind] > 0 else 0

        
class _nDCG(TopKMetric):
    """
    normalized Discounted Cumulative Gain
    input: tensor, tensor
    """

    def __init__(self, k=None):
        super().__init__(k)
    
    def forward(self, y_true, y_score):
        assert self._k is not None, "@k is not set!"

        sort_true = y_true[torch.argsort(y_score, descending=True)][:self._k]
        dcg = ((2**sort_true - 1) / torch.log2(torch.arange(2, sort_true.shape[0] + 2))).sum().item()
        idcg = ((2**(torch.sort(y_true, descending=True)[0][:self._k]) - 1) / torch.log2(torch.arange(2, sort_true.shape[0] + 2))).sum().item()
        return dcg / max(self._eps, idcg)
    

class _Recall(TopKMetric):
    """ 
    Recall at K
    """

    def __init__(self, k=None):
        super().__init__(k)

    def forward(self, y_true, y_score):
        assert self._k is not None, "@k is not set!"

        sort_true = y_true[torch.argsort(y_score, descending=True)][:self._k]
        sort_true[sort_true != 0] = 1 # all relevant objs in MAP is equivalent
        num_pos = min(self._k, (y_true != 0).sum().item())

        return sort_true.sum().item() / max(self._eps, num_pos)


def calc_aggregate(metrics, qrels, scored_docs):
    scored_docs_dict = dict()
    for it in scored_docs:
        if it[0] in scored_docs_dict:
            scored_docs_dict[it[0]][it[1]] = it[2]
        else:
            scored_docs_dict[it[0]] = dict({it[1]: it[2]})

    qrels_dict = dict()
    for it in qrels:
        if it[0] in qrels_dict:
            qrels_dict[it[0]][it[1]] = it[2]
        else:
            qrels_dict[it[0]] = dict({it[1]: it[2]})
    
    res_metrics = dict()
    for metric in metrics:
        res_metrics[str(metric)] = 0
    cnt = 0

    for user in qrels_dict.keys():
        if user in scored_docs_dict:
            cnt += 1

            y_score = torch.Tensor([scored_docs_dict[user][doc] for doc in scored_docs_dict[user].keys()])
            y_score_items = torch.Tensor([doc for doc in scored_docs_dict[user].keys()]) # doc = int

            y_true = torch.zeros_like(y_score)
            for doc in qrels_dict[user].keys():
                y_true[y_score_items == doc] = qrels_dict[user][doc]

            for metric in metrics:
                res_metrics[str(metric)] += metric(y_true, y_score)
        else:
            continue

    assert cnt > 0, "no queries"
    
    for metric in metrics:
        res_metrics[str(metric)] /= cnt
    
    return res_metrics
            

# objects
R = _Recall()
nDCG = _nDCG()
MRR = _MRR()
MAP = _MAP()