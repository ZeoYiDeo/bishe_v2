import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import Dataset


random.seed(0)

class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor
        self.cv= torch.load('../data/Bili_Food/Bili_Food_vit.pt')
        self.ct= torch.load('../data/Bili_Food/Bili_Food_bert.pt')
    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index] , self.cv[int(self.item_tensor[index])], self.ct[int(self.item_tensor[index])]

    def __len__(self):
        return self.user_tensor.size(0)