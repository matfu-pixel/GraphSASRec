import torch


class TwhinGraphEncoder(torch.nn.Module):
    """
    twhin-like vectors in user-item graph
    """

    def __init__(self, user_num, item_num, type_num, args):
        super().__init__()

        self.dev = args['device']

        self.item_emb = torch.nn.Embedding(item_num + 1, args['hidden_units'])
        self.user_emb = torch.nn.Embedding(user_num + 1, args['hidden_units'])
        self.type_emb = torch.nn.Embedding(type_num + 1, args['hidden_units'])
    
    def forward(self, users, items, types):
        users_embs = self.user_emb(torch.LongTensor(users).to(self.dev))
        items_embs = self.item_emb(torch.LongTensor(items).to(self.dev))
        types_embs = self.type_emb(torch.LongTensor(types).to(self.dev))

        return users_embs + types_embs, types_embs

