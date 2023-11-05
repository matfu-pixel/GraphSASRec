import torch
from torch.utils.data import Dataset, DataLoader


class TwhinDataset(Dataset):
    def __init__(self, input_file):
        with open(input_file, 'r') as f:
            self.inputs = [list(map(int, line.strip().split())) for line in f.readlines()]
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.LongTensor(self.inputs[idx])

def collate_fn(input_batch):
    input = torch.stack([input_batch[i] for i in range(len(input_batch))], dim=0)
    return {
        "node1": input[:, 0],
        "node2": input[:, 1]
    }

def get_train_dataloader(dataset_name, batch_size=32, max_length=200, train_neg_per_positive=256):
    dataset_dir = f"datasets/{dataset_name}"
    train_dataset = TwhinDataset(f"{dataset_dir}/train_graph/input.txt")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return train_loader

def get_val_dataloader(dataset_name, batch_size=32, max_length=200, train_neg_per_positive=256):
    dataset_dir = f"datasets/{dataset_name}"
    dataset = TwhinDataset(f"{dataset_dir}/val_graph/input.txt")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return dataloader
