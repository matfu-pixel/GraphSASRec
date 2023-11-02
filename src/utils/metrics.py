import pandas as pd
import numpy as np


class MAP(torch.nn.Module):
    """
    Mean Average Precision
    input: list[tensor]
    """

    def __init__(self, k):
        super().__init__()
        self._k = k
    
    def forward(self, y_true, y_score): # positive_index
        sum_ap = 0
        cnt_ap = 0
        for user_true, user_score in zip(y_true, y_score):
            user_true = user_true.clone()
            user_true[user_true != 0] = 1
            sort_true = user_true[torch.argsort(user_score, descending=True)][:self._k]
            precision = sort_true.cumsum(dim=0) / torch.arange(1, sort_true.shape[0] + 1)

            sum_ap += (precision * sort_true).sum() / max(0.00001, sort_true.sum())
            cnt_ap += 1
        
        return sum_ap / cnt_ap


class MRR(torch.nn.Module):
    """
    Mean Reciprocal Rank
    input: list[tensor]
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_score):
        sum_mrr = 0
        cnt_mrr = 0
        for user_true, user_score in zip(y_true, y_score):
            sort_true = user_true[torch.argsort(user_score, descending=True)]
            ind = sort_true.argmax(dim=-1)
            if sort_true[ind] > 0:
                sum_mrr += 1 / (ind + 1)
            else:
                sum_mrr += 0

            cnt_mrr += 1
        
        return sum_mrr / cnt_mrr
            

class NDCG(torch.nn.Module):
    """
    normalized discountted cumulative gain
    input: list[tensor]
    """

    def __init__(self, k):
        super().__init__()
        self._k = k
    
    def forward(self, y_true, y_score):
        sum_ndcg = 0
        cnt_ndcg = 0
        for user_true, user_score in zip(y_true, y_score):
            user_true = user_true[:self._k]
            user_score = user_score[:self._k]
            sort_true = user_true[torch.argsort(user_score, descending=True)][:self._k]
            dcg = (sort_true / torch.log2(torch.arange(2, sort_true.shape[0] + 2))).sum()
            idcg = (torch.sort(user_true, descending=True)[0] / torch.log2(torch.arange(2, sort_true.shape[0] + 2))).sum()
            sum_ndcg += dcg / max(0.0001, idcg)

            cnt_ndcg += 1
        
        return sum_ndcg / cnt_ndcg


class DiversityK(torch.nn.Module):
    """
    user-wise diversity@K = number of unique genres
    """

    def __init__(self, K):
        super().__init__()
        self._K = K
    
    def forward(self, df_preds):
        user_wise_diversity_k = 0
        cnt = 0
        for genres in df_preds['preds']:
            if genres == -1:
                cnt += 1
            else:
                tmp = []
                for it in genres:
                    tmp += it[2]
                cnt += 1
                user_wise_diversity_k += len(set(tmp))
        return user_wise_diversity_k / cnt


class LongTailK(torch.nn.Module):
    """
    user-wise long tail@k = median of popularity of top-50
    """

    def __init__(self, K):
        super().__init__()
        self._K = K

    def forward(self, df_preds):
        user_wise_popularity_k = 0
        cnt = 0
        for popularity in df_preds['preds']:
            if popularity == -1:
                continue
            else:
                tmp = []
                for it in popularity:
                    tmp.append(it[1])
                cnt += 1
                user_wise_popularity_k += np.median(tmp)
        return user_wise_popularity_k / cnt
