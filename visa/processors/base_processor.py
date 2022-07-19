from typing import Union, List, Tuple

import os
import json
import torch


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


class BaseProcessor(object):
    def __init__(self,
                 model_name_or_path: str,
                 max_length: int = 128,
                 use_crf: bool = True):
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.use_crf = use_crf

    def word_segment(self, raw: str) -> List[str]:
        return raw.split()

    def read_visd4sa_data(self, fpath: Union[str, os.PathLike]) -> List[ABSAExample]:
        examples = []
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                data = json.loads(line)
                raw_text = data['text']

                norm_words = []  # word sequence
                aspect_tags = []  # tag sequence for target aspect
                polarity_tags = []  # tag sequence for targeted sentiment polarity

                last_index = 0
                for span in data["labels"]:
                    as_tag, senti_tag = span[-1].split("#")
                    # Add prefix tokens
                    prefix_span_text = self.word_segment(raw_text[last_index:span[0]])
                    norm_words.extend(prefix_span_text)
                    aspect_tags.extend(["O"] * len(prefix_span_text))
                    polarity_tags.extend(["O"] * len(prefix_span_text))

                    aspect_text = self.word_segment(raw_text[span[0]:span[1]])
                    for idx, _ in enumerate(aspect_text):
                        if idx == 0:
                            aspect_tags.append(f"B-{as_tag.strip()}")
                            polarity_tags.append(f"B-{senti_tag.strip()}")
                            continue
                        aspect_tags.append(f"I-{as_tag.strip()}")
                        polarity_tags.append(f"I-{senti_tag.strip()}")
                    norm_words.extend(aspect_text)
                    last_index = span[1]

                last_span_text = self.word_segment(raw_text[last_index:])
                norm_words.extend(last_span_text)
                aspect_tags.extend(["O"] * len(last_span_text))
                polarity_tags.extend(["O"] * len(last_span_text))

                assert len(norm_words) == len(aspect_tags), f"Not match: {line}"
                examples.append(ABSAExample(tokens=norm_words, a_tags=aspect_tags, p_tags=polarity_tags))
        return examples

    def read_absa_data(self, fpath: Union[str, os.PathLike]) -> List[ABSAExample]:
        examples = []
        with open(fpath, encoding='UTF-8') as fp:
            def parse_item(word_item: str) -> Tuple[str, Union[str, None], str]:
                eles = word_item.split('=')
                if len(eles) == 2:
                    word, wtag = eles
                else:
                    wtag = eles[-1]
                    word = (len(eles) - 2) * "="
                if wtag == "O":
                    return word, None, wtag
                ptag = wtag.split("-")[-1].strip()
                return word, "T", ptag
            for line in fp:
                _, tag_string = line.strip().split('####')

                norm_words = []  # word sequence
                aspect_tags = []  # tag sequence for target aspect
                polarity_tags = []  # tag sequence for targeted sentiment polarity

                span = []
                prev_tag = None
                items = tag_string.split(' ')
                for idx, item in enumerate(items):
                    # Valid label is: O, T-POS, T-NEG, T-NEU
                    word, prefix, ptag = parse_item(item)
                    if prev_tag is None or prev_tag == ptag:
                        prev_tag = ptag
                        span.append(word)
                        if idx + 1 < len(item):
                            continue
                    spand_text = self.word_segment(" ".join(span))
                    norm_words.extend(spand_text)
                    if prev_tag == 'O':
                        aspect_tags.extend(['O'] * len(spand_text))
                        polarity_tags.extend(['O'] * len(spand_text))
                    else:
                        for idx, _ in enumerate(spand_text):
                            if idx == 0:
                                aspect_tags.append(f"B-SPAN")
                                polarity_tags.append(f"B-{prev_tag.strip()}")
                                continue
                            aspect_tags.append(f"I-SPAN")
                            polarity_tags.append(f"I-{prev_tag.strip()}")
                    span = [word]
                    prev_tag = ptag
                assert len(norm_words) == len(aspect_tags), f"Not match: {line}"
                examples.append(ABSAExample(tokens=norm_words, a_tags=aspect_tags, p_tags=polarity_tags))
        return examples

    def convert_example_to_features(self, examples: List[ABSAExample]) -> List[ABSAFeature]:
        return NotImplemented


if __name__ == "__main__":
    processor = BaseProcessor("vinai/phobert-base")
    processor.read_absa_data("./datasets/absa/laptop14_test.txt")
