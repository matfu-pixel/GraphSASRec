from argparse import ArgumentParser

import torch
from dataloaders.gsasrec_dataloader import get_test_dataloader
from dataloaders.utils import get_num_items
from utils.eval_utils import evaluate
from utils.utils import build_model, get_device, load_config


parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config_ml1m_sasrec.py')
parser.add_argument('--checkpoint', type=str, required=True)
args = parser.parse_args()
config = load_config(args.config)
num_items = get_num_items(config.dataset_name) 
device = get_device()
model = build_model(config)
model = model.to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device))

test_dataloader = get_test_dataloader(config.dataset_name, batch_size=config.eval_batch_size, max_length=config.sequence_length, 
                                      train_neg_per_positive=config.negs_per_pos)
evaluation_result = evaluate(model, test_dataloader, config.metrics, config.recommendation_limit, 
                                 config.filter_rated, device=device) 
print(evaluation_result)
