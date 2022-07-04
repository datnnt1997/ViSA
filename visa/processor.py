from typing import Union, List
from pyvi import ViTokenizer
from transformers import PhobertTokenizer

from visa.constants import RDRSEGMENTER

import os
import json
import itertools

class ABSAExample(object):
    def __init__(self, tokens, a_tags, s_tags):
        self.tokens: List[str] = tokens
        self.aspect_tags: List[str] = a_tags
        self.senti_tags: List[str] = s_tags


def word_segment(raw: str) -> List[str]:
    if RDRSEGMENTER is not None:
        sentences = RDRSEGMENTER.tokenize(raw)
        return list(itertools.chain(*sentences))
    else:
        return ViTokenizer.tokenize(raw).split()


def read_data(fpath: Union[str, os.PathLike]):
    examples = []
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data = json.loads(line)
            raw_text = data['text']
            norm_text = []
            aspect_tags = []
            senti_tags = []
            last_index = 0
            for span in data["labels"]:
                as_tag, senti_tag = span[-1].split("#")
                senti_tags.append(senti_tag)

                prefix_span_text = word_segment(raw_text[last_index:span[0]])
                aspect_tags.extend(["O"] * len(prefix_span_text))
                norm_text.extend(prefix_span_text)

                aspect_text = word_segment(raw_text[span[0]:span[1]])
                for idx, _ in enumerate(aspect_text):
                    if idx == 0:
                        aspect_tags.append(f"B-{as_tag.strip()}")
                        continue
                    aspect_tags.append(f"I-{as_tag.strip()}")
                norm_text.extend(aspect_text)

                last_index = span[1]
            assert len(norm_text) == len(aspect_tags), f"Not match: {line}"
            examples.append(ABSAExample(tokens=norm_text, a_tags=aspect_tags, s_tags=senti_tags))
    return examples


def convert_example_to_features(examples: List[ABSAExample],
                                tokenizer: PhobertTokenizer,
                                max_seq_len: int = 256):
    for example in examples:
        encoding = tokenizer(example.tokens,
                             padding='max_length',
                             truncation=True,
                             max_length=max_seq_len,
                             is_split_into_words=True)
        sentence = ' '.join(example.tokens)
        subwords = tokenizer.tokenize(sentence)
        # valid_labels = np.ones(len(encoding.input_ids), dtype=int) * -100

# DEBUG
if __name__ == "__main__":
    read_data("./datasets/samples/train.jsonl.txt")