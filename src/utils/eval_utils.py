import torch
import tqdm 
from models.gsasrec import GSASRec
from metrics.metrics import calc_aggregate


def evaluate(model: GSASRec, data_loader, metrics, limit, filter_rated, device):
    model.eval()
    users_processed = 0
    scored_docs = []
    qrels = []
    loss_sum = 0
    max_batches = len(data_loader)

    with torch.no_grad():
        for batch_idx, (data, negatives, rated, target) in tqdm.tqdm(enumerate(data_loader), total=max_batches):
            data, negatives, target = data.to(device), negatives.to(device), target.to(device)

            # cal eval loss
            last_hidden_state, _ = model(data)
            pos_neg_concat = torch.cat([target.unsqueeze(-1), negatives], dim=-1)

            pos_neg_embeddings = model.get_item_embeddings(pos_neg_concat, output=True)

            users = last_hidden_state[:, -1, :]
            logits = torch.einsum('be, bne -> bn', users, pos_neg_embeddings)
            gt = torch.zeros_like(logits)
            gt[:, 0] = 1
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, gt, reduction='none').mean(-1).mean()
            loss_sum += loss.item()

            if filter_rated:
                items, scores = model.get_predictions(data, limit, rated)
            else:
                items, scores = model.get_predictions(data, limit)
            for recommended_items, recommended_scores, target in zip(items, scores, target):
                for item, score in zip(recommended_items, recommended_scores):
                    scored_docs.append((users_processed, item.item(), score.item()))
                qrels.append((users_processed, target.item(), 1))
                users_processed += 1
                pass
            
    result = calc_aggregate(metrics, qrels, scored_docs)
    result['BCE'] = loss_sum / max_batches

    return result
