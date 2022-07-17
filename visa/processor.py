from typing import Union, List
from pyvi import ViTokenizer
from transformers import PhobertTokenizer
from collections import defaultdict
from visa.constants import RDRSEGMENTER, ASPECT_LABELS, POLARITY_LABELS

import os
import json
import torch
import numpy as np
import itertools


class ABSAExample(object):
    def __init__(self, tokens, a_tags, p_tags):
        self.tokens: List[str] = tokens
        self.aspect_tags: List[str] = a_tags
        self.polarity_tags: List[str] = p_tags


class ABSAFeature(object):
    def __init__(self, input_ids, token_type_ids, attention_mask, valid_ids, a_labels, p_labels, label_masks):
        self.input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        self.a_labels = torch.as_tensor(a_labels, dtype=torch.long)
        self.p_labels = torch.as_tensor(p_labels, dtype=torch.long)
        self.token_type_ids = torch.as_tensor(token_type_ids, dtype=torch.long)
        self.attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
        self.valid_ids = torch.as_tensor(valid_ids, dtype=torch.long)
        self.label_masks = torch.as_tensor(label_masks, dtype=torch.long)


def word_segment(raw: str) -> List[str]:
    if RDRSEGMENTER is not None:
        sentences = RDRSEGMENTER.tokenize(raw)
        return list(itertools.chain(*sentences))
    else:
        return ViTokenizer.tokenize(raw).split()


def read_data(fpath: Union[str, os.PathLike]) -> List[ABSAExample]:
    examples = []
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data = json.loads(line)
            raw_text = data['text']
            norm_text = []
            aspect_tags = []
            polarity_tags = []
            last_index = 0
            for span in data["labels"]:
                as_tag, senti_tag = span[-1].split("#")
                # Add prefix tokens
                prefix_span_text = word_segment(raw_text[last_index:span[0]])
                norm_text.extend(prefix_span_text)
                aspect_tags.extend(["O"] * len(prefix_span_text))
                polarity_tags.extend(["O"] * len(prefix_span_text))

                aspect_text = word_segment(raw_text[span[0]:span[1]])
                for idx, _ in enumerate(aspect_text):
                    if idx == 0:
                        aspect_tags.append(f"B-{as_tag.strip()}")
                        polarity_tags.append(f"B-{senti_tag.strip()}")
                        continue
                    aspect_tags.append(f"I-{as_tag.strip()}")
                    polarity_tags.append(f"I-{senti_tag.strip()}")
                norm_text.extend(aspect_text)
                last_index = span[1]

            last_span_text = word_segment(raw_text[last_index:])
            norm_text.extend(last_span_text)
            aspect_tags.extend(["O"] * len(last_span_text))
            polarity_tags.extend(["O"] * len(last_span_text))

            assert len(norm_text) == len(aspect_tags), f"Not match: {line}"
            examples.append(ABSAExample(tokens=norm_text, a_tags=aspect_tags, p_tags=polarity_tags))
    return examples


def convert_example_to_features(examples: List[ABSAExample],
                                tokenizer: PhobertTokenizer,
                                max_seq_len: int = 256,
                                use_crf: bool = True):
    features = []
    for example in examples:
        encoding = tokenizer(example.tokens,
                             padding='max_length',
                             truncation=True,
                             max_length=max_seq_len,
                             is_split_into_words=True)
        a_tag_ids = [ASPECT_LABELS.index(a_tag) for a_tag in example.aspect_tags]
        p_tag_ids = [POLARITY_LABELS.index(p_tag) for p_tag in example.polarity_tags]

        seq_len = len(example.aspect_tags)

        subwords = tokenizer.tokenize(' '.join(example.tokens))
        valid_ids = np.zeros(len(encoding.input_ids), dtype=int)
        label_marks = np.zeros(len(encoding.input_ids), dtype=int)
        valid_a_labels = np.ones(len(encoding.input_ids), dtype=int) * -100
        valid_p_labels = np.ones(len(encoding.input_ids), dtype=int) * -100

        i = 1
        for idx, subword in enumerate(subwords[:max_seq_len - 2]):
            if idx != 0 and subwords[idx - 1].endswith("@@"):
                continue
            if use_crf:
                valid_ids[i - 1] = idx + 1
            else:
                valid_ids[idx + 1] = 1
            valid_a_labels[idx + 1] = a_tag_ids[i - 1]
            valid_p_labels[idx + 1] = p_tag_ids[i - 1]
            i += 1
        if max_seq_len >= seq_len:
            label_padding_size = (max_seq_len - seq_len)
            label_marks[:seq_len] = [1] * seq_len
            a_tag_ids.extend([0] * label_padding_size)
            p_tag_ids.extend([0] * label_padding_size)
        else:
            a_tag_ids = a_tag_ids[:max_seq_len]
            p_tag_ids = p_tag_ids[:max_seq_len]
            label_marks[:-2] = [1] * (max_seq_len - 2)
            a_tag_ids[-2:] = [0] * 2
            p_tag_ids[-2:] = [0] * 2
        if use_crf and label_marks[0] == 0:
            raise f"{example.tokens} - {a_tag_ids} - {p_tag_ids} have mark == 0 at index 0!"
        items = {key: val for key, val in encoding.items()}
        items['a_labels'] = a_tag_ids if use_crf else valid_a_labels
        items['p_labels'] = p_tag_ids if use_crf else valid_p_labels
        items['valid_ids'] = valid_ids
        items['label_masks'] = label_marks if use_crf else valid_ids
        features.append(ABSAFeature(**items))

        for k, v in items.items():
            assert len(v) == max_seq_len, f"Expected length of {k} is {max_seq_len} but got {len(v)}"
    return features


# DEBUG
if __name__ == "__main__":
    print(read_data("./datasets/samples/train.jsonl.txt"))