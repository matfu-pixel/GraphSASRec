import json


def get_num_items(dataset):
    with open(f"datasets/{dataset}/dataset_stats.json", 'r') as f:
        stats = json.load(f)
    return stats['num_items']


def get_num_users(dataset):
    with open(f"datasets/{dataset}/dataset_stats.json", 'r') as f:
        stats = json.load(f)
    return stats['num_users']


def get_padding_value(dataset_dir):
    with open(f"{dataset_dir}/dataset_stats.json", 'r') as f:
        stats = json.load(f)
    padding_value = stats['num_items'] + 1
    return padding_value
