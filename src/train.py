from argparse import ArgumentParser
import os

import torch
from utils.utils import load_config, build_model, get_device
from dataloaders.gsasrec_dataloader import get_train_dataloader, get_val_dataloader
from dataloaders.utils import get_num_items
from tqdm import tqdm
from utils.eval_utils import evaluate
from torchinfo import summary

from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter(log_dir='logs/graph_sasrec1')

checkpoints_dir = "checkpoints"
if not os.path.exists(checkpoints_dir):
    os.mkdir(checkpoints_dir)

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config/config_ml1m_graph_sasrec.py')
args = parser.parse_args()
config = load_config(args.config)

num_items = get_num_items(config.dataset_name) 
device = get_device()
model = build_model(config)

train_dataloader = get_train_dataloader(config.dataset_name, batch_size=config.train_batch_size,
                                         max_length=config.sequence_length, train_neg_per_positive=config.negs_per_pos)
val_dataloader = get_val_dataloader(config.dataset_name, batch_size=config.eval_batch_size, max_length=config.sequence_length,
                                    train_neg_per_positive=config.negs_per_pos)

optimiser = torch.optim.Adam(model.parameters())
batches_per_epoch = min(config.max_batches_per_epoch, len(train_dataloader))

best_metric = float("-inf")
best_model_name = None
step = 0
steps_not_improved = 0

model = model.to(device)
# summary(model, (config.train_batch_size, config.sequence_length), batch_dim=None)

for epoch in range(config.max_epochs):
    model.train()   
    batch_iter = iter(train_dataloader)
    pbar = tqdm(range(batches_per_epoch))
    loss_sum = 0
    for batch_idx in pbar:
        step += 1
        positives, negatives = [tensor.to(device) for tensor in next(batch_iter)]
        model_input = positives[:, :-1]
        last_hidden_state, attentions = model(model_input)
        labels = positives[:, 1:]
        negatives = negatives[:, 1:, :]
        pos_neg_concat = torch.cat([labels.unsqueeze(-1), negatives], dim=-1)


        pos_neg_embeddings = model.get_item_embeddings(pos_neg_concat, output=True)

        mask = (model_input != num_items + 1).float()
        logits = torch.einsum('bse, bsne -> bsn', last_hidden_state, pos_neg_embeddings)
        gt = torch.zeros_like(logits)
        gt[:, :, 0] = 1

        alpha = config.negs_per_pos / (num_items - 1)
        t = config.gbce_t 
        beta = alpha * ((1 - 1/alpha)*t + 1/alpha)
        
        positive_logits = logits[:, :, 0:1].to(torch.float64) #use float64 to increase numerical stability
        negative_logits = logits[:,:,1:].to(torch.float64)
        eps = 1e-10
        positive_probs = torch.clamp(torch.sigmoid(positive_logits), eps, 1-eps)
        positive_probs_adjusted = torch.clamp(positive_probs.pow(-beta), 1+eps, torch.finfo(torch.float64).max)
        to_log = torch.clamp(torch.div(1.0, (positive_probs_adjusted  - 1)), eps, torch.finfo(torch.float64).max)
        positive_logits_transformed = to_log.log()
        logits = torch.cat([positive_logits_transformed, negative_logits], -1)
        loss_per_element = torch.nn.functional.binary_cross_entropy_with_logits(logits, gt, reduction='none').mean(-1)*mask
        loss = loss_per_element.sum() / mask.sum()
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        loss_sum += loss.item()
        pbar.set_description(f"Epoch {epoch} loss: {loss_sum / (batch_idx + 1)}")
    
    writer.add_scalar('Train/BCE', loss_sum / batches_per_epoch, global_step=epoch + 1)

    evaluation_result = evaluate(model, val_dataloader, config.metrics, config.recommendation_limit, 
                                 config.filter_rated, device=device)
    
    writer.add_scalar('Eval/BCE', evaluation_result['BCE'], global_step=epoch + 1)
    writer.add_scalar('Eval/nDCG@10', evaluation_result[str(config.val_metric)], global_step=epoch + 1)

    print(f"Epoch {epoch} evaluation result: {evaluation_result}")
    if evaluation_result[str(config.val_metric)] > best_metric:
        best_metric = evaluation_result[str(config.val_metric)]
        model_name = f"checkpoints/gsasrec-{config.dataset_name}-step:{step}-t:{config.gbce_t}-negs:{config.negs_per_pos}-emb:{config.embedding_dim}-dropout:{config.dropout_rate}-metric:{best_metric}.pt" 
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
