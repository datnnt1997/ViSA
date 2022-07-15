from typing import Optional, Any, List, Tuple
from collections import OrderedDict
from transformers import logging, RobertaConfig, RobertaForTokenClassification
from torchcrf import CRF

from .loss import HierarchicalLossNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.set_verbosity_error()


class ABSAOutput(OrderedDict):
    loss: Optional[torch.FloatTensor] = torch.FloatTensor([0.0])
    a_loss: Optional[torch.FloatTensor] = torch.FloatTensor([0.0])
    s_loss: Optional[torch.FloatTensor] = torch.FloatTensor([0.0])
    a_tags: Optional[List[int]] = []
    s_tags: Optional[List[int]] = []

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        return tuple(self[k] for k in self.keys())


class ABSAConfig(RobertaConfig):
    def __init__(self,
                 num_alabels: int = 21,
                 num_slabels: int = 4,
                 device: str = 'cpu', **kwargs):
        super().__init__(num_labels=num_alabels, **kwargs)
        self.num_alabels = num_alabels
        self.num_slabels = num_slabels
        self.polarity_weight_loss = 0.8
        self.dependency_weight_loss = 0.8
        self.device = device


class ABSAModel(RobertaForTokenClassification):
    def __init__(self, config, **kwargs):
        super(ABSAModel, self).__init__(config=config, **kwargs)
        self.aspect_detection = nn.Linear(config.hidden_size, config.num_alabels)
        self.polarity_weight_loss = config.polarity_weight_loss
        self.dependency_weight_loss = config.dependency_weight_loss
        self.polarity_transformation = nn.Linear(config.hidden_size, config.num_slabels)
        self.activation = nn.Tanh()
        self.layer_norm = nn.LayerNorm(config.num_alabels + config.num_slabels)
        self.polarity_detection = nn.Linear(config.num_alabels + config.num_slabels, config.num_slabels)

        self.a_crf = CRF(config.num_alabels, batch_first=True)
        self.s_crf = CRF(config.num_slabels, batch_first=True)

        self.hier_loss = HierarchicalLossNetwork(device=config.device,
                                                 aspect_func=self.a_crf,
                                                 senti_func=self.s_crf)

        self.post_init()
        self.loss_init()

    def loss_init(self):
        self.hier_loss.s_w = 0.0
        self.hier_loss.beta = 0.0

    def joint_step(self, weight_gain):
        self.hier_loss.s_w = self.hier_loss.s_w + weight_gain if self.hier_loss.s_w < 1 else 1
        self.hier_loss.beta = self.hier_loss.beta + weight_gain if self.hier_loss.beta < 1 else 1

    def forward(self,
                input_ids: torch.LongTensor,
                token_type_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                a_labels: Optional[torch.LongTensor] = None,
                s_labels: Optional[torch.LongTensor] = None,
                valid_ids: Optional[torch.LongTensor] = None,
                label_masks: Optional[torch.LongTensor] = None,
                **kwargs):
        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  head_mask=None)[0]

        batch_size, max_len, feat_dim = seq_output.shape
        range_vector = torch.arange(0, batch_size, dtype=torch.long, device=seq_output.device).unsqueeze(1)
        valid_seq_output = seq_output[range_vector, valid_ids]

        valid_seq_output = self.dropout(valid_seq_output)

        a_logits = self.aspect_detection(valid_seq_output)

        s_feats = torch.cat((self.polarity_transformation(valid_seq_output), a_logits), dim=-1)
        s_feats = self.activation(s_feats)
        s_feats = self.layer_norm(s_feats)
        s_logits = self.polarity_detection(s_feats)

        if a_labels is not None and s_labels is not None:
            loss, a_tags, s_tags = self.hier_loss(aspects=a_logits,
                                                  sentis=s_logits,
                                                  true_a_labels=a_labels,
                                                  true_s_labels=s_labels,
                                                  mask=label_masks)
            return ABSAOutput(loss=loss, a_tags=a_tags, s_tags=s_tags)

        a_tags = self.a_crf.decode(a_logits, mask=label_masks != 0)
        s_tags = self.s_crf.decode(s_logits, mask=label_masks != 0)

        return ABSAOutput(a_tags=a_tags, s_tags=s_tags)


# DEBUG
if __name__ == "__main__":
    config = ABSAConfig.from_pretrained('vinai/phobert-base', num_alabels=21, num_slabels=4)
    model = ABSAModel(config=config)

    input_ids = torch.randint(0, 2999, [2, 20], dtype=torch.long)
    mask = torch.zeros([2, 20], dtype=torch.long)
    alabels = torch.randint(1, 20, [2, 20], dtype=torch.long)
    slabels = torch.randint(1, 3, [2, 20], dtype=torch.long)
    valid_ids = torch.ones([2, 20], dtype=torch.long)
    label_mask = torch.ones([2, 20], dtype=torch.long)
    label_mask[:, -2:] = 0

    print(model(input_ids,
                attention_mask=mask,
                alabels=alabels,
                slabels=slabels,
                valid_ids=valid_ids,
                label_masks=label_mask
                ))
