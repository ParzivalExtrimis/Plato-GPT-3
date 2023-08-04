import torch
from torch.utils.data import Dataset
from typing import List, Tuple

class BookCorpusDataset(Dataset):
    def __init__(self, ids: List[int], block_size: int) -> None:
        self.data = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        ix = index if index < (len(self.data) - self.block_size) else index - self.block_size
        assert ix + self.block_size < len(self.data), 'Indexed out of bounds. The dataloader invoked a sample out of the range of len(data)'

        x = self.data[ix : ix+self.block_size]
        y = self.data[ix+1 : ix+self.block_size+1]
        return x, y