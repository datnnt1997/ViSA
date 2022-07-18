from typing import Union, List

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
        return NotImplemented

    def read_visd4sa_data(self, fpath: Union[str, os.PathLike]) -> List[ABSAExample]:
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
                    prefix_span_text = self.word_segment(raw_text[last_index:span[0]])
                    norm_text.extend(prefix_span_text)
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
                    norm_text.extend(aspect_text)
                    last_index = span[1]

                last_span_text = self.word_segment(raw_text[last_index:])
                norm_text.extend(last_span_text)
                aspect_tags.extend(["O"] * len(last_span_text))
                polarity_tags.extend(["O"] * len(last_span_text))

                assert len(norm_text) == len(aspect_tags), f"Not match: {line}"
                examples.append(ABSAExample(tokens=norm_text, a_tags=aspect_tags, p_tags=polarity_tags))
        return examples

    def read_absa_data(self, fpath: Union[str, os.PathLike]) -> List[ABSAExample]:
        examples = []
        with open(fpath, encoding='UTF-8') as fp:
            for line in fp:
                record = {}
                sent, tag_string = line.strip().split('####')
                record['sentence'] = sent
                word_tag_pairs = tag_string.split(' ')
                # tag sequence for targeted sentiment
                ts_tags = []
                # tag sequence for opinion target extraction
                ote_tags = []
                # word sequence
                words = []
                for item in word_tag_pairs:
                    # valid label is: O, T-POS, T-NEG, T-NEU
                    eles = item.split('=')
                    if len(eles) == 2:
                        word, tag = eles
                    elif len(eles) > 2:
                        tag = eles[-1]
                        word = (len(eles) - 2) * "="
                    words.append(word.lower())
                    if tag == 'O':
                        ote_tags.append('O')
                        ts_tags.append('O')
                    elif tag == 'T-POS':
                        ote_tags.append('T')
                        ts_tags.append('T-POS')
                    elif tag == 'T-NEG':
                        ote_tags.append('T')
                        ts_tags.append('T-NEG')
                    elif tag == 'T-NEU':
                        ote_tags.append('T')
                        ts_tags.append('T-NEU')
                    else:
                        raise Exception('Invalid tag %s!!!' % tag)
                record['words'] = words.copy()
                record['ote_raw_tags'] = ote_tags.copy()
                record['ts_raw_tags'] = ts_tags.copy()
                examples.append(record)
        return examples

    def convert_example_to_features(self, examples: List[ABSAExample]) -> List[ABSAFeature]:
        return NotImplemented


