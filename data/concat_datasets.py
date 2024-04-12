"""
concat_datasets.py.py 2022/6/27 16:17
Written by Wensheng Fan
"""
from torch.utils.data import Dataset


class ConcatDataset(Dataset):

    def __init__(self, dataloader_syn, dataloader_real):

        super().__init__()
        self.datasets = (dataloader_syn, dataloader_real)

    def __getitem__(self, index):
        return tuple(d[index] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)