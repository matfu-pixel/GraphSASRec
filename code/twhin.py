import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from src.utils.metrics import RecallK, DiversityK, LongTailK
from src.datasets.datasets import SASRecDataset, TwhinDataset
from src.models.sasrec import SASRec
from src.models.graph_encoders import TwhinGraphEncoder
from src.utils.losses import TwhinLoss, TwhinLossWithParameters


def train_epoch_twhin(model, optimizer, epoch, train_dataloader, writer, loss_fn):
    model.train()
    last_loss = -1
    for idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        users = batch['users']
        types = batch['types']
        items = batch['items']
        
        users_output, items_output = model(users, items, types)
        twhin_loss, l2_reg = loss_fn(users_output, items_output)
        loss = twhin_loss # + l2_reg

        writer.add_scalar('Train/link_prediction', twhin_loss.item(), global_step=idx + epoch * len(train_dataloader))
        # writer.add_scalar('Train/total_loss', loss.item(), global_step=idx + epoch * len(train_dataloader))

        last_loss = twhin_loss.item() 

        loss.backward()
        optimizer.step()
    
    return last_loss

def val_epoch_twhin(model, epoch, val_dataloader, train_dataloader, writer, loss_fn):
    model.eval()
    sum_loss = 0
    cnt = 0
    with torch.no_grad():
        for batch in val_dataloader:
            cnt += 1
            users = batch['users']
            types = batch['types']
            items = batch['items']
            
            users_output, items_output = model(users, items, types)
            twhin_loss, l2_reg = loss_fn(users_output, items_output)

            sum_loss += twhin_loss.item()
        
        writer.add_scalar('Val/link_prediction', sum_loss / cnt, global_step=(epoch + 1) * len(train_dataloader))
    
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
        'num_epochs_twhin': 100
    }

    train = pd.read_pickle('train.pkl')
    val = pd.read_pickle('val.pkl')

    train_dataset_twhin = TwhinDataset(train)
    val_dataset_twhin = TwhinDataset(val)

    train_dataloader_twhin = DataLoader(train_dataset_twhin, batch_size=args['batch_size'], collate_fn=TwhinDataset.collate_fn, drop_last=True)
    val_dataloader_twhin = DataLoader(val_dataset_twhin, batch_size=args['batch_size'], collate_fn=TwhinDataset.collate_fn)
    print('Read data!')

    model = TwhinGraphEncoder(6040, 3952, 5, args).to(args['device'])
    loss_fn = TwhinLossWithParameters(reg_weight=0)

    optimizer = torch.optim.Adam([*model.parameters(), *loss_fn.parameters()], lr=0.001, betas=(0.9, 0.98))
    writer = SummaryWriter(log_dir='logs/twhin_scale_100ep_2')

    for epoch in range(args['num_epochs_twhin']):
        train_loss = train_epoch_twhin(model, optimizer, epoch, train_dataloader_twhin, writer, loss_fn)
        print(f'Train link-prediction loss on epoch {epoch + 1}: {train_loss}')
        val_loss = val_epoch_twhin(model, epoch, val_dataloader_twhin, train_dataloader_twhin, writer, loss_fn)
        print(f'Val link-prediction loss on epoch {epoch + 1}: {val_loss}')

    torch.save(model.state_dict(), 'checkpoints/twhin_scale_100ep_2.pth')