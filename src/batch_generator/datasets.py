import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class MlDataset(Dataset):
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
