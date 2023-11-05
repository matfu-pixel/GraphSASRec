import torch


class TwhinGraphEncoder(torch.nn.Module):
    """
    twhin-like vectors in user-item graph
    """

    def __init__(self, user_num, item_num, embedding_dim=256, use_type=False, type_num=None):
        super().__init__()

        self._user_num = user_num
        self._item_num = item_num
        self._type_num = type_num
        self._use_type = use_type

        self.item_emb = torch.nn.Embedding(item_num + 1, embedding_dim)
        self.user_emb = torch.nn.Embedding(user_num + 1, embedding_dim)
        if self._use_type:
            self.type_emb = torch.nn.Embedding(type_num + 1, embedding_dim)
    
    def forward(self, users, items, types=None):
        users_embs = self.user_emb(torch.LongTensor(users).to(self.item_emb.weight.device))
        items_embs = self.item_emb(torch.LongTensor(items).to(self.item_emb.weight.device))

        if self._use_type:
            types_embs = self.type_emb(torch.LongTensor(types).to(self.item_emb.weight.device))
            return users_embs + types_embs, items_embs
        else:
            return users_embs, items_embs
    
    def get_output_embeddings(self) -> torch.nn.Embedding:
        return self.item_emb
    
    def get_predictions(self, users, limit):
        with torch.no_grad():
            users_embs = self.user_emb(torch.LongTensor(users).to(self.item_emb.weight.device))
            output_embeddings = self.get_output_embeddings()
            scores = torch.einsum('bd,nd->bn', users_embs, output_embeddings.weight)
            scores[:,0] = float("-inf")
            scores[:,self._item_num:] = float("-inf")
            result = torch.topk(scores, limit, dim=1)
            return result.indices, result.values