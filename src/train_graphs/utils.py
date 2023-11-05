from config.config import TwhinGraphEncoderConfig
from models.twhin import TwhinGraphEncoder
from dataloaders.utils import get_num_items, get_num_users


def build_model(config: TwhinGraphEncoderConfig):
    num_items = get_num_items(config.dataset_name)
    num_users = get_num_users(config.dataset_name)
    model = TwhinGraphEncoder(num_users, num_items, embedding_dim=config.embedding_dim, use_type=config.use_type, type_num=5)
    return model
