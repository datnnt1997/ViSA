from typing import Union, List
from pathlib import Path
from torch.utils.data import Dataset

from visa.processor import read_absa_data, read_visd4sa_data, convert_example_to_features,\
    convert_example_to_features_with_offset, ABSAFeature
from visa.constants import ASPECT_LABELS, POLARITY_LABELS, ABSA_ASPECT_LABELS, ABSA_POLARITY_LABELS

import os
import torch


class ABSADataset(Dataset):
    def __init__(self, features: List[ABSAFeature], device: str = 'cpu'):
        self.examples = features
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return {key: val.to(self.device) for key, val in self.examples[index].__dict__.items()}


def build_dataset(data_dir: Union[str or os.PathLike],
                  tokenizer,
                  task,
                  dtype: str = 'train',
                  max_seq_len: int = 128,
                  device: str = 'cpu',
                  overwrite_data: bool = True,
                  use_crf: bool = True) -> ABSADataset:
    if task == 'UIT-ViSD4SA':
        dfile_path = Path(data_dir + f'/{dtype}.jsonl')
        cached_path = dfile_path.with_suffix('.cached')
    elif 'ABSA' in task:
        cached_path = Path(data_dir + f'/{dtype}.cached')
    else:
         raise ValueError(f'{task} not found!')
    if not os.path.exists(cached_path) or overwrite_data:
        if task == 'UIT-ViSD4SA':
            examples = read_visd4sa_data(dfile_path, task, dtype)
            features = convert_example_to_features(examples, tokenizer,
                                                   aspect_labels=ASPECT_LABELS,
                                                   polarity_labels=POLARITY_LABELS,
                                                   max_seq_len=max_seq_len,
                                                   use_crf=use_crf,)
        elif 'ABSA' in task:
            examples = read_absa_data(data_dir, task, dtype)
            features = convert_example_to_features_with_offset(examples, tokenizer,
                                                   aspect_labels=ABSA_ASPECT_LABELS,
                                                   polarity_labels=ABSA_POLARITY_LABELS,
                                                   max_seq_len=max_seq_len,
                                                   use_crf=use_crf)

        torch.save(features, cached_path)
    else:
        features = torch.load(cached_path)
    return ABSADataset(features, device=device)


# DEBUG
if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    build_dataset('datasets/samples', tokenizer, dtype='train', max_seq_len=128, device='cuda', overwrite_data=True)