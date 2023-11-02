import torch
from math import sqrt
from torch.utils.data import Dataset, DataLoader


class SimpleModel(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

        self.linear1 = torch.nn.Linear(2, 100)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(100, 1)
    
    def forward(self, batch):
        return self.linear2(self.relu(self.linear1(batch)))


model = SimpleModel().to('cuda')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

for i in range(10000):
    optimizer.zero_grad()
    x = torch.rand(128, 2)
    y = (x ** 2).sum(axis=1)
    x = x.to('cuda')
    y = y.to('cuda')

    y_model = model(x)
    loss = ((y - y_model) ** 2).mean()
    print(sqrt(loss.item()))
    loss.backward()

    optimizer.step()