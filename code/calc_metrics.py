import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.models.sasrec import SASRec, GraphSASRec
from src.utils.metrics import RecallK, DiversityK, LongTailK
from src.datasets.datasets import SASRecDataset


def get_top_50(events):
    tmp = []
    for movie, rating in zip(events['MovieID'], events['Rating']):
        tmp.append((movie, rating))
    tmp.sort(key=lambda x: x[1])
    return [it[0] for it in tmp[-50:]]


if __name__ == '__main__':
    args = {
        'device': 'cuda', 
        'hidden_units': 50, 
        'dropout_rate': 0.5,
        'num_blocks': 2,
        'num_heads': 1,
        'maxlen': 128,
        'batch_size': 2048,
        'num_epochs_sasrec': 100, 
        'num_epochs_twhin': 50
    }

    train = pd.read_pickle('train.pkl')
    movie_to_cnt = dict()
    movie_to_genre = dict()
    for movieID, genre in zip(train['MovieID'].values, train['Genre'].values):
        if movieID not in movie_to_cnt:
            movie_to_cnt[movieID] = 1
        else:
            movie_to_cnt[movieID] += 1

        if movieID not in movie_to_genre:
            movie_to_genre[movieID] = genre

    users_last_history = pd.read_pickle('user_last_history.pkl')
    val = pd.read_pickle('val.pkl')
    users_last_history['candidate'] = [[0] for _ in range(users_last_history.shape[0])]

    dataset = SASRecDataset(users_last_history)
    dataloader = DataLoader(dataset, batch_size=users_last_history.shape[0], collate_fn=SASRecDataset.collate_fn)

    model = SASRec(3952, args).to(args['device'])
    model.load_state_dict(torch.load('checkpoints/sasrec.pth'))
    logits = model.predict(next(iter(dataloader))['users'], torch.arange(1, 3952 + 1))

    # create df_preds
    preds = (torch.argsort(logits, dim=1)[:, -50:] + 1).to('cpu').tolist()
    print(logits[0].sort()[-50:])
    preds_new = []
    for row in preds:
        tmp = []
        for it in row:
            if it not in movie_to_cnt: # 196
                tmp.append((it, 0, []))
            else:
                tmp.append((it, movie_to_cnt[it], movie_to_genre[it]))
        preds_new.append(tmp)


    user2preds = pd.DataFrame({'UserID': users_last_history['UserID'].values, 'preds': preds_new})

    positive_pairs = val.groupby('UserID').apply(lambda x: [it for it in x['MovieID']]).reset_index()
    positive_pairs['positive'] = positive_pairs[0]
    positive_pairs = positive_pairs[['UserID', 'positive']]

    df_preds = positive_pairs.set_index('UserID').join(user2preds.set_index('UserID'), how='left', on='UserID').reset_index()
    df_preds = df_preds.fillna(-1)

    print(df_preds.head())

    metric_recall50 = RecallK(50)
    metric_diversity50 = DiversityK(50)
    metric_long_tail50 = LongTailK(50)

    print('User-wise recall@50:', round(metric_recall50(df_preds) * 100, 2), '%')
    print('User-wise diversity@50:', metric_diversity50(df_preds))
    print('User-wise long_tail@50:', metric_long_tail50(df_preds))
