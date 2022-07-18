from typing import Union, List
from pathlib import Path
from torch.utils.data import Dataset

from .base_processor import ABSAFeature

import os
import torch


class SADataset(Dataset):
    def __init__(self, features: List[ABSAFeature], device: str = 'cpu'):
        self.examples = features
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return {key: val.to(self.device) for key, val in self.examples[index].__dict__.items()}


def build_dataset(data_dir: Union[str or os.PathLike],
                  processor,
                  dtype: str = 'train',
                  device: str = 'cpu',
                  overwrite_data: bool = True,) -> SADataset:
    dfile_path = Path(data_dir+f'/{dtype}.jsonl')
    cached_path = dfile_path.with_suffix('.cached')
    if not os.path.exists(cached_path) or overwrite_data:
        examples = processor.read_visd4sa_data(dfile_path)
        features = processor.convert_example_to_features(examples)
        torch.save(features, cached_path)
    else:
        features = torch.load(cached_path)
    return SADataset(features, device=device)


# DEBUG
if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    build_dataset('datasets/samples', tokenizer, dtype='train', max_seq_len=128, device='cuda', overwrite_data=True)