from argparse import ArgumentParser
import os

import torch
from utils.utils import load_config, get_device
from train_graphs.utils import build_model
from dataloaders.twhin_dataloader import get_train_dataloader, get_val_dataloader
from dataloaders.utils import get_num_items, get_num_users
from tqdm import tqdm
from train_graphs.eval_utils import evaluate
from heads.heads import TwhinLoss

from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter(log_dir='logs/twhin1')

checkpoints_dir = "checkpoints"
if not os.path.exists(checkpoints_dir):
    os.mkdir(checkpoints_dir)

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config/config_ml1m_twhin.py')
args = parser.parse_args()
config = load_config(args.config)

num_items = get_num_items(config.dataset_name) 
num_users = get_num_users(config.dataset_name)
device = get_device()
model = build_model(config)

train_dataloader = get_train_dataloader(config.dataset_name, batch_size=config.train_batch_size)
val_dataloader = get_val_dataloader(config.dataset_name, batch_size=config.eval_batch_size)

optimiser = torch.optim.Adam(model.parameters())
loss_fn = TwhinLoss(reg_weight=config.reg_weight)
batches_per_epoch = min(config.max_batches_per_epoch, len(train_dataloader))

best_metric = float("-inf")
best_model_name = None
step = 0
steps_not_improved = 0

model = model.to(device)

for epoch in range(config.max_epochs):
    model.train()
    batch_iter = iter(train_dataloader)
    pbar = tqdm(range(batches_per_epoch))
    loss_sum = 0
    for batch_idx in pbar:
        step += 1
        input_batch = next(batch_iter)
        users = input_batch["node1"]
        items = input_batch["node2"]
        node1, node2 = model(users, items)
        link_prediction, reg = loss_fn(node1, node2)

        loss_sum += link_prediction.item()
        loss = link_prediction + reg
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        pbar.set_description(f"Epoch {epoch} loss: {loss_sum / (batch_idx + 1)}")

    writer.add_scalar('Train/twhin-loss', loss_sum / batches_per_epoch, global_step=epoch + 1)

    evaluation_result = evaluate(model, val_dataloader, config.metrics, config.recommendation_limit, 
                                 loss_fn, device=device)
    
    writer.add_scalar('Eval/twhin-loss', evaluation_result['twhin-loss'], global_step=epoch + 1)
    writer.add_scalar('Eval/Recall@50', evaluation_result[str(config.val_metric)], global_step=epoch + 1)

    print(f"Epoch {epoch} evaluation result: {evaluation_result}")
    if evaluation_result[str(config.val_metric)] > best_metric:
        best_metric = evaluation_result[str(config.val_metric)]
        model_name = f"checkpoints/twhin-{config.dataset_name}-step:{step}-emb:{config.embedding_dim}-metric:{best_metric}.pt" 
        print(f"Saving new best model to {model_name}")
        if best_model_name is not None:
            os.remove(best_model_name)
        best_model_name = model_name
        steps_not_improved = 0
        torch.save(model.state_dict(), model_name)
    else:
        steps_not_improved += 1
        print(f"Validation metric did not improve for {steps_not_improved} steps")
        if steps_not_improved >= config.early_stopping_patience:
            print(f"Stopping training, best model was saved to {best_model_name}")
            break
