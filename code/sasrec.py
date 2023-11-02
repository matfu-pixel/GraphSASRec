import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from src.utils.metrics import RecallK, DiversityK, LongTailK
from src.datasets.datasets import SASRecDataset, TwhinDataset
from src.models.sasrec import SASRec
from src.models.graph_encoders import TwhinGraphEncoder
from src.utils.losses import TwhinLoss

def train_epoch_sasrec(model, optimizer, epoch, train_dataloader, writer, loss_fn):
    model.train()
    last_loss = -1
    for idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        historys = batch['users']
        positives = batch['candidates']
        perm = torch.randperm(positives.shape[0])
        negatives = positives[perm]
        logits_pos, logits_neg = model(historys, positives, negatives)
        logits = torch.cat([logits_pos, logits_neg], dim=0)
        labels = torch.cat([torch.ones(logits_pos.shape[0]).to(args['device']), torch.zeros(logits_neg.shape[0]).to(args['device'])])
        loss = loss_fn(logits, labels)

        writer.add_scalar('Train/BCE', loss.item(), global_step=idx + epoch * len(train_dataloader))
        last_loss = loss.item() 

        loss.backward()
        optimizer.step()
    
    return last_loss

def val_epoch_sasrec(model, epoch, val_dataloader, train_dataloader, writer, loss_fn):
    model.eval()
    sum_loss = 0
    cnt = 0
    with torch.no_grad():
        for batch in val_dataloader:
            cnt += 1
            historys = batch['users']
            positives = batch['candidates']
            perm = torch.randperm(positives.shape[0])
            negatives = positives[perm]
            logits_pos, logits_neg = model(historys, positives, negatives)
            logits = torch.cat([logits_pos, logits_neg], dim=0)
            labels = torch.cat([torch.ones(logits_pos.shape[0]).to(args['device']), torch.zeros(logits_neg.shape[0]).to(args['device'])], dim=0)
            loss = loss_fn(logits, labels)
            sum_loss += loss.item()
        
        writer.add_scalar('Val/BCE', sum_loss / cnt, global_step=(epoch + 1) * len(train_dataloader))
    
    return sum_loss / cnt

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

    train = pd.read_pickle('train.csv')
    val = pd.read_pickle('val.csv')

    train_dataset_sasrec = SASRecDataset(train)
    val_dataset_sasrec = SASRecDataset(val)
    train_dataloader_sasrec = DataLoader(train_dataset_sasrec, batch_size=args['batch_size'], collate_fn=SASRecDataset.collate_fn, drop_last=True)
    val_dataloader_sasrec = DataLoader(val_dataset_sasrec, batch_size=args['batch_size'], collate_fn=SASRecDataset.collate_fn)
    print('Read data!')

    model = SASRec(3952, args).to(args['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98))
    writer = SummaryWriter(log_dir='logs/sasrec')

    for epoch in range(args['num_epochs_sasrec']):
        train_loss = train_epoch_sasrec(model, optimizer, epoch, train_dataloader_sasrec, writer, torch.nn.BCEWithLogitsLoss())
        print(f'Train bce loss on epoch {epoch + 1}: {train_loss}')
        val_loss = val_epoch_sasrec(model, epoch, val_dataloader_sasrec, train_dataloader_sasrec, writer, torch.nn.BCEWithLogitsLoss())
        print(f'Val bce loss on epoch {epoch + 1}: {val_loss}')

    torch.save(model.state_dict(), 'checkpoints/sasrec.pth')
