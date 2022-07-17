from typing import Optional
from transformers import logging, RobertaForTokenClassification
from torchcrf import CRF

from visa.loss import CTDLoss
from .modeling_outputs import ABSAOutput
from .configuration import ABSARoBERTaConfig
import torch
import torch.nn as nn

logging.set_verbosity_error()

class HierRoBERTaML(RobertaForTokenClassification):
    def __init__(self, config: ABSARoBERTaConfig, **kwargs):
        super(HierRoBERTaML, self).__init__(config=config, **kwargs)
        self.aspect_layer_1 = nn.Linear(config.hidden_size, config.num_aspect_labels)
        self.polarity_layer_1 = nn.Linear(config.hidden_size, config.num_polarity_labels)

        self.activation = nn.Tanh()
        self.layer_norm = nn.LayerNorm(config.num_aspect_labels + config.num_polarity_labels)

        self.aspect_layer_2 = nn.Linear(config.num_aspect_labels, config.num_aspect_labels)
        self.polarity_layer_2 = nn.Linear(config.num_aspect_labels + config.num_polarity_labels,
                                          config.num_polarity_labels)

        self.aspect_detection = CRF(config.num_aspect_labels, batch_first=True)
        self.polarity_detection = CRF(config.num_polarity_labels, batch_first=True)

        self.hier_loss = CTDLoss(device=config.device,
                                 aspect_func=self.aspect_detection,
                                 senti_func=self.polarity_detection)
        self.post_init()

    def forward(self,
                input_ids: torch.LongTensor,
                token_type_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                a_labels: Optional[torch.LongTensor] = None,
                p_labels: Optional[torch.LongTensor] = None,
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

        a_feats = self.aspect_layer_1(valid_seq_output)
        a_logits = self.aspect_layer_2(a_feats)

        p_feats = torch.cat((self.polarity_layer_1(valid_seq_output), a_logits), dim=-1)
        p_feats = self.activation(p_feats)
        p_feats = self.layer_norm(p_feats)
        p_logits = self.polarity_layer_2(p_feats)

        if a_labels is not None and p_labels is not None:
            loss, a_tags, p_tags = self.hier_loss(aspects=a_logits,
                                                  polarities=p_logits,
                                                  true_aspects=a_labels,
                                                  true_polarities=p_labels,
                                                  mask=label_masks)
            return ABSAOutput(loss=loss, aspects=a_tags, polarities=p_tags)

        a_tags = self.aspect_detection.decode(a_logits, mask=label_masks != 0)
        p_tags = self.polarity_detection.decode(p_logits, mask=label_masks != 0)
        return ABSAOutput(a_tags=a_tags, s_tags=p_tags)


# DEBUG
if __name__ == "__main__":
    config = ABSARoBERTaConfig.from_pretrained('vinai/phobert-base', num_alabels=21, num_slabels=4)
    model = HierRoBERTaML.from_pretrained('vinai/phobert-base', config=config)

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
