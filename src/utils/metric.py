import pandas as pd
import numpy as np


class Metric:
    """
    base metric class
    """
    def __init__(self):
        pass

    def call(self, df_preds):
        pass


class RecallK(Metric):
    """
    user-wise recall@k
    """

    def __init__(self, K):
        super().__init__()
        self._K = K
    
    def __call__(self, df_preds):
        user_wise_recall_k = 0
        cnt = 0
        for it, positive in zip(df_preds['preds'], df_preds['positive']):
            if it == -1:
                preds = []
            else:
                preds = []
                for elem in it:
                    preds.append(elem[0])
            s = 0
            for movie in positive:
                if movie in preds:
                    s += 1
            user_wise_recall_k += s / min(len(positive), self._K)
            cnt += 1
        return user_wise_recall_k / cnt


class DiversityK(Metric):
    """
    user-wise diversity@K = number of unique genres
    """

    def __init__(self, K):
        super().__init__()
        self._K = K
    
    def __call__(self, df_preds):
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


class LongTailK(Metric):
    """
    user-wise long tail@k = median of popularity of top-50
    """

    def __init__(self, K):
        super().__init__()
        self._K = K

    def __call__(self, df_preds):
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
