import torch 
from models.transformer_decoder import TransformerBlock

class GSASRec(torch.nn.Module):
    def __init__ (self, num_items, sequence_length=200, embedding_dim=256, num_heads=4, num_blocks=3, 
                  dropout_rate=0.5, reuse_item_embeddings=False, freeze_graph_vectors=None):
        super(GSASRec, self).__init__()

        self.freeze_graph_vectors = freeze_graph_vectors
        if freeze_graph_vectors is not None:
            self.transform_graph_vectors = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim * 2, embedding_dim * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(embedding_dim * 2, embedding_dim)
            )

        self.num_items = num_items
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.embeddings_dropout = torch.nn.Dropout(dropout_rate)

        self.num_heads = num_heads

        self.item_embedding = torch.nn.Embedding(self.num_items + 2, self.embedding_dim)
        self.position_embedding = torch.nn.Embedding(self.sequence_length, self.embedding_dim)
    
        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(self.embedding_dim, self.num_heads, self.embedding_dim, dropout_rate)
            for _ in range(num_blocks)
        ])
        self.seq_norm = torch.nn.LayerNorm(self.embedding_dim)
        self.reuse_item_embeddings = reuse_item_embeddings
        if not self.reuse_item_embeddings:
            self.output_embedding = torch.nn.Embedding(self.num_items + 2, self.embedding_dim)
    
    def get_item_embeddings(self, input, output=False):
        if output:
            if self.reuse_item_embeddings:
                seq = self.output_embedding(input.long().to(self.item_embedding.weight.device))
            else:
                seq = self.item_embedding(input.long().to(self.item_embedding.weight.device))
        else:
            seq = self.item_embedding(input.long().to(self.item_embedding.weight.device))
        
        mask = (input != self.num_items + 1).float().unsqueeze(-1).to(self.item_embedding.weight.device)
        
        if self.freeze_graph_vectors is not None:
            seq_graph = self.transform_graph_vectors(self.freeze_graph_vectors(input.long().to(self.item_embedding.weight.device) * mask.squeeze().long()))
            seq_graph = seq_graph * mask
            seq = seq + seq_graph
        
        return seq

    def forward(self, input):
        seq = self.get_item_embeddings(input.long())
        mask = (input != self.num_items + 1).float().unsqueeze(-1)
        
        bs = seq.size(0)
        positions = torch.arange(seq.shape[1]).unsqueeze(0).repeat(bs, 1).to(input.device)
        pos_embeddings = self.position_embedding(positions)[:input.size(0)]
        seq = seq + pos_embeddings
        seq = self.embeddings_dropout(seq)
        seq *= mask
        
        attentions = []
        for i, block in enumerate(self.transformer_blocks):
            seq, attention = block(seq, mask)
            attentions.append(attention)
        
        seq_emb = self.seq_norm(seq)
        return seq_emb, attentions
    
    def get_predictions(self, input, limit, rated=None):
        with torch.no_grad():
            model_out, _ = self.forward(input)
            seq_emb = model_out[:,-1,:] 
            output_embeddings = self.get_item_embeddings(torch.arange(self.num_items + 2), output=True)
            scores = torch.einsum('bd,nd->bn', seq_emb, output_embeddings)
            scores[:,0] = float("-inf")
            scores[:,self.num_items+1:] = float("-inf")
            if rated is not None:
                for i in range(len(input)):
                    for j in rated[i]:
                        scores[i, j] = float("-inf")
            result = torch.topk(scores, limit, dim=1)
            return result.indices, result.values
