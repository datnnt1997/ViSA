from typing import Union, List
from vncorenlp import VnCoreNLP
from transformers import PhobertTokenizer

from visa.processors.base_processor import BaseProcessor, ABSAExample, ABSAFeature
from visa.constants import ASPECT_LABELS, POLARITY_LABELS

import os
import itertools
import numpy as np


class PhoBERTProcessor(BaseProcessor):
    def __init__(self,
                 vncorenlp: Union[str, os.PathLike] = "vncorenlp/VnCoreNLP-1.1.1.jar",
                 **kwargs):
        super(PhoBERTProcessor, self).__init__(**kwargs)
        self.rdr_segmenter = VnCoreNLP(vncorenlp, annotators="wseg", max_heap_size='-Xmx500m')
        self.tokenizer = PhobertTokenizer.from_pretrained(self.model_name_or_path)

    def word_segment(self, raw: str) -> List[str]:
        sentences = self.rdr_segmenter.tokenize(raw)
        return list(itertools.chain(*sentences))

    def convert_example_to_features(self, examples: List[ABSAExample]):
        features = []
        for example in examples:
            encoding = self.tokenizer(example.tokens,
                                      padding='max_length',
                                      truncation=True,
                                      max_length=self.max_length,
                                      is_split_into_words=True)
            a_tag_ids = [ASPECT_LABELS.index(a_tag) for a_tag in example.aspect_tags]
            p_tag_ids = [POLARITY_LABELS.index(p_tag) for p_tag in example.polarity_tags]

            seq_len = len(example.aspect_tags)

            subwords = self.tokenizer.tokenize(' '.join(example.tokens))
            valid_ids = np.zeros(len(encoding.input_ids), dtype=int)
            label_marks = np.zeros(len(encoding.input_ids), dtype=int)
            valid_a_labels = np.ones(len(encoding.input_ids), dtype=int) * -100
            valid_p_labels = np.ones(len(encoding.input_ids), dtype=int) * -100

            i = 1
            for idx, subword in enumerate(subwords[:self.max_length - 2]):
                if idx != 0 and subwords[idx - 1].endswith("@@"):
                    continue
                if self.use_crf:
                    valid_ids[i - 1] = idx + 1
                else:
                    valid_ids[idx + 1] = 1
                valid_a_labels[idx + 1] = a_tag_ids[i - 1]
                valid_p_labels[idx + 1] = p_tag_ids[i - 1]
                i += 1
            if self.max_length >= seq_len:
                label_padding_size = (self.max_length - seq_len)
                label_marks[:seq_len] = [1] * seq_len
                a_tag_ids.extend([0] * label_padding_size)
                p_tag_ids.extend([0] * label_padding_size)
            else:
                a_tag_ids = a_tag_ids[:self.max_length]
                p_tag_ids = p_tag_ids[:self.max_length]
                label_marks[:-2] = [1] * (self.max_length - 2)
                a_tag_ids[-2:] = [0] * 2
                p_tag_ids[-2:] = [0] * 2
            if self.use_crf and label_marks[0] == 0:
                raise f"{example.tokens} - {a_tag_ids} - {p_tag_ids} have mark == 0 at index 0!"
            items = {key: val for key, val in encoding.items()}
            items['a_labels'] = a_tag_ids if self.use_crf else valid_a_labels
            items['p_labels'] = p_tag_ids if self.use_crf else valid_p_labels
            items['valid_ids'] = valid_ids
            items['label_masks'] = label_marks if self.use_crf else valid_ids
            features.append(ABSAFeature(**items))

            for k, v in items.items():
                assert len(v) == self.max_length, f"Expected length of {k} is {self.max_length} but got {len(v)}"

        return features
