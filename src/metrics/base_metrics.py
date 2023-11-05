import torch


class BaseMetric(torch.nn.Module):
    """
    Base Metric Class
    """

    def __init__(self):
        super().__init__()
        self._eps = 1e-18
    
    def __repr__(self):
        return self.__class__.__name__[1:]
    
    def __str__(self):
        return self.__class__.__name__[1:]


class TopKMetric(BaseMetric):
    """
    TopK Metric Class
    """

    def __init__(self, k=None):
        super().__init__()
        self._k = k

    def __matmul__(self, k):
        return self.__class__(k)
    
    def __repr__(self):
        return self.__class__.__name__[1:] + "@" + str(self._k)
    
    def __str__(self):
        return self.__class__.__name__[1:] + "@" + str(self._k)