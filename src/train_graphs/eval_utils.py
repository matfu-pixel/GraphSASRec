import torch
import tqdm 
from models.twhin import TwhinGraphEncoder
from metrics.metrics import calc_aggregate


def evaluate(model: TwhinGraphEncoder, data_loader, metrics, limit, loss_fn, device):
    model.eval()
    users_processed = 0
    scored_docs = []
    qrels = []
    loss_sum = 0
    max_batches = min(len(data_loader), 1)

    with torch.no_grad():
        for batch_idx, batch in tqdm.tqdm(enumerate(data_loader), total=max_batches):
            input_users = batch["node1"]
            input_items = batch["node2"]

            # cal eval loss
            node1, node2 = model(input_users, input_items)
            loss_sum += loss_fn(node1, node2)[0].item()

            items, scores = model.get_predictions(input_users, limit)
            for recommended_items, recommended_scores, target in zip(items, scores, input_items):
                for item, score in zip(recommended_items, recommended_scores):
                    scored_docs.append((users_processed, item.item(), score.item()))
                qrels.append((users_processed, target.item(), 1))
                users_processed += 1
                pass

            break
            
    result = calc_aggregate(metrics, qrels, scored_docs)
    result['twhin-loss'] = loss_sum / max_batches

    return result
