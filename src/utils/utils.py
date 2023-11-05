import importlib
from config.config import GSASRecExperimentConfig
from dataloaders.utils import get_num_items, get_num_users
import torch
from models.gsasrec import GSASRec
from models.twhin import TwhinGraphEncoder


def load_config(config_file: str):
    spec = importlib.util.spec_from_file_location("config", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config

def build_model(config: GSASRecExperimentConfig):
    num_items = get_num_items(config.dataset_name)
    num_users = get_num_users(config.dataset_name)
    if config.use_freeze_graph_vectors:
        twhin_model = TwhinGraphEncoder(num_users, num_items)
        twhin_model.load_state_dict(torch.load('checkpoints/twhin-ml1m-step:17280-emb:256-metric:0.05859375.pt'))
        graph_vectors = torch.nn.Embedding.from_pretrained(twhin_model.item_emb.weight)
        model = GSASRec(num_items, sequence_length=config.sequence_length, embedding_dim=config.embedding_dim,
                        num_heads=config.num_heads, num_blocks=config.num_blocks, dropout_rate=config.dropout_rate, freeze_graph_vectors=graph_vectors)
        return model
    else:
        model = GSASRec(num_items, sequence_length=config.sequence_length, embedding_dim=config.embedding_dim,
                        num_heads=config.num_heads, num_blocks=config.num_blocks, dropout_rate=config.dropout_rate)
        return model

def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device="cuda:0"
    return device
