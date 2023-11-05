from metrics.metrics import nDCG, R


class GSASRecExperimentConfig(object):
    def __init__(self, dataset_name, sequence_length=200, embedding_dim=256, train_batch_size=128,
                             num_heads=4, num_blocks=3, 
                             dropout_rate=0.0,
                             negs_per_pos=256,
                             max_epochs=10000,
                             max_batches_per_epoch=100,
                             metrics=[nDCG@10, R@1, R@10],
                             val_metric = nDCG@10,
                             early_stopping_patience=200,
                             gbce_t = 0.75,
                             filter_rated=True,
                             eval_batch_size=512,
                             recommendation_limit=10,
                             reuse_item_embeddings=False,
                             use_freeze_graph_vectors=False
                             ):
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.dataset_name = dataset_name
        self.train_batch_size = train_batch_size
        self.negs_per_pos = negs_per_pos
        self.max_epochs = max_epochs
        self.max_batches_per_epoch = max_batches_per_epoch
        self.val_metric = val_metric
        self.metrics = metrics
        self.early_stopping_patience = early_stopping_patience
        self.gbce_t = gbce_t
        self.filter_rated = filter_rated
        self.recommendation_limit = recommendation_limit
        self.eval_batch_size = eval_batch_size
        self.reuse_item_embeddings = reuse_item_embeddings
        self.use_freeze_graph_vectors = use_freeze_graph_vectors


class TwhinGraphEncoderConfig(object):
    def __init__(self, dataset_name, reg_weight=0, embedding_dim=256, use_type=False, train_batch_size=8192,
                 max_epochs=10000,
                 max_batches_per_epoch=10000,
                 metrics=[R@50],
                 val_metric=R@50,
                 early_stopping_patience=200,
                 eval_batch_size=2048,
                 recommendation_limit=50):
        self.dataset_name = dataset_name
        self.reg_weight = reg_weight
        self.embedding_dim = embedding_dim
        self.use_type = use_type
        self.train_batch_size = train_batch_size
        self.max_epochs = max_epochs
        self.max_batches_per_epoch = max_batches_per_epoch
        self.metrics = metrics
        self.val_metric = val_metric
        self.early_stopping_patience = early_stopping_patience
        self.eval_batch_size = eval_batch_size
        self.recommendation_limit = recommendation_limit
