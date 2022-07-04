from transformers import RobertaConfig, RobertaForTokenClassification
from torchcrf import CRF

import torch
import torch.nn as nn
import random


class ABSAConfig(RobertaConfig):
    def __init__(self,
                 num_alabels: int = 21,
                 num_slabels: int = 4,
                 teacher_forcing_ratio: float =0.5, **kwargs):
        super().__init__(num_labels=num_alabels, **kwargs)
        self.num_alabels = num_alabels
        self.num_slabels = num_slabels
        self.teacher_forcing_ratio = teacher_forcing_ratio


class ABSAModel(RobertaForTokenClassification):
    def __init__(self, config, **kwargs):
        super(ABSAModel, self).__init__(config=config, **kwargs)
        self.teacher_forcing_ratio = config.teacher_forcing_ratio
        self.aspect_detection = nn.Linear(config.hidden_size, config.num_alabels)
        self.sentiment_detection = nn.Linear(config.hidden_size, config.num_slabels)

        self.a_crf = CRF(config.num_alabels, batch_first=True)
        self.s_crf = CRF(config.num_slabels, batch_first=True)

        self.post_init()

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                alabels=None,
                slabels=None,
                valid_ids=None,
                label_masks=None):
        seq_output = self.roberta(input_ids, token_type_ids, attention_mask, head_mask=None)[0]

        batch_size, max_len, feat_dim = seq_output.shape
        range_vector = torch.arange(0, batch_size, dtype=torch.long, device=seq_output.device).unsqueeze(1)
        valid_seq_output = seq_output[range_vector, valid_ids]

        valid_seq_output = self.dropout(valid_seq_output)

        a_logits = self.aspect_detection(valid_seq_output)
        seq_tags = self.a_crf.decode(a_logits, mask=label_masks != 0)
        if alabels is not None:
            a_loss = 1 - self.crf(a_logits, alabels, mask=label_masks.type(torch.uint8))
            teacher_force = random.random() < self.teacher_forcing_ratio
            if teacher_force:

        return None


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
