import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class SASRecDataset(Dataset):
    """
    convert dataframe to torch dataset
    """

    def __init__(self, dataframe, col_history='history', col_candidate='candidate'):
        super().__init__()

        self._data = list()
        for user, candidate in tqdm(zip(dataframe[col_history], dataframe[col_candidate])):
            candidate_idx = candidate[0]
            user_idxs = []
            for event in user:
                user_idxs.append(event[0])
            self._data.append((torch.LongTensor(list(reversed(user_idxs))), candidate_idx))
        
        self._col_history = col_history
        self._col_candidate = col_candidate
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        return self._data[idx]

    @staticmethod
    def collate_fn(batch):
        """
        convert list of tuple to dict of tensors (add padding)
        """

        users = []
        candidates = []
        for it in batch:
            users.append(it[0])
            candidates.append(it[1])
        return {
            'users': torch.nn.utils.rnn.pad_sequence(users, batch_first=True),
            'candidates': torch.LongTensor(candidates)
        }


class TwhinDataset(Dataset):
    """
    dataset for twhin embeddings
    """

    def __init__(self, dataframe, col_user='UserID', col_type='Rating', col_item='MovieID'):
        super().__init__()

        self._data = list()
        for user, rating, candidate in tqdm(zip(dataframe[col_user], dataframe[col_type], dataframe[col_item])):
            self._data.append((user, rating, candidate))
        
    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    @staticmethod
    def collate_fn(batch):
        """
        convert list of tuple to dict of tensors (add padding)
        """

        users = []
        items = []
        types = []
        for it in batch:
            users.append(it[0])
            types.append(it[1])
            items.append(it[2])
        return {
            'users': torch.LongTensor(users),
            'types': torch.LongTensor(types),
            'items': torch.LongTensor(items)
        }
