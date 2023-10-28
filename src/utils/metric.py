import pandas as pd
import n

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
        for preds, positive in zip(df_preds['preds'], df_preds['positive']):
            s = 0
            if pd.isnull(preds):
                preds = []
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
        for genres in df_preds['genres']:
            if pd.isnull(genres):
                cnt += 1
            else:
                cnt += 1
                user_wise_diversity_k += len(set(genres))
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
        for popularity in df_preds['popularity']:
            if pd.isnull(popularity):
                continue
            else:
                cnt += 1
                user_wise_popularity_k += np.median(popularity)
        return user_wise_popularity / cnt
                
