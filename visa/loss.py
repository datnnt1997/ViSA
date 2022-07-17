from typing import List
from visa.constants import ASPECT_LABELS, POLARITY_LABELS
from visa.helper import split_tag

import torch
import itertools


class CTDLoss:
    """
    Logics to calculate the controllable task dependency loss of the model.
    """
    def __init__(self, aspect_func, senti_func, device='cpu', alpha=1, beta=0.8, gamma=1, p_loss=3):
        self.a_func = aspect_func
        self.s_func = senti_func
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.p_loss = p_loss
        self.device = device

    def check_hierarchy(self, aspects, polarities):
        """
        Check if the predicted tag of aspect and polarity is not same IOB prefix for the entire batch
        """
        # check using the dictionary whether the current level's prediction
        # belongs to the superclass (prediction from the prev layer)
        seq_len = aspects.size()[0]
        bool_tensor = []
        for i in range(seq_len):
            a_prefix, _ = split_tag(ASPECT_LABELS[aspects[i].item()])
            p_prefix, _ = split_tag(POLARITY_LABELS[polarities[i].item()])
            if a_prefix == p_prefix:
                bool_tensor.append(False)
            else:
                bool_tensor.append(True)
        return torch.FloatTensor(bool_tensor).to(self.device)

    def calculate_tloss(self, aspects, polarities, true_aspects, true_polarities, mask=None):
        """
        Calculate the task loss.
        """
        a_loss = 1 - self.a_func(aspects, true_aspects, mask=mask.type(torch.uint8), reduction='mean')
        s_loss = 1 - self.s_func(polarities, true_polarities, mask=mask.type(torch.uint8), reduction='mean')
        tloss = self.gamma * a_loss + self.beta * s_loss
        return self.alpha * tloss

    def calculate_dloss(self,
                        aspects: List[int],
                        polarities: List[int],
                        true_aspects: torch.LongTensor,
                        true_polarities: torch.LongTensor):
        """
        Calculate the dependence loss.
        """

        dloss = 0
        aspect_tags = torch.LongTensor(aspects).to(self.device)
        polarity_tags = torch.LongTensor(polarities).to(self.device)
        D_l = self.check_hierarchy(aspect_tags, polarity_tags)

        l_prev = torch.where(aspect_tags == true_aspects,
                             torch.FloatTensor([0]).to(self.device),
                             torch.FloatTensor([1]).to(self.device))
        l_curr = torch.where(polarity_tags == true_polarities,
                             torch.FloatTensor([0]).to(self.device),
                             torch.FloatTensor([1]).to(self.device))
        dloss += torch.sum(torch.pow(self.p_loss, D_l * l_prev) * torch.pow(self.p_loss, D_l * l_curr) - 1)
        return self.beta * dloss

    def __call__(self, aspects, polarities, true_aspects, true_polarities, mask=None):
        lloss = self.calculate_tloss(aspects, polarities, true_aspects, true_polarities, mask)
        aspect_tags = list(itertools.chain(*self.a_func.decode(aspects, mask=mask != 0)))
        polarity_tags = list(itertools.chain(*self.s_func.decode(polarities, mask=mask != 0)))
        active_labels = mask.view(-1) != 0
        true_aspects = torch.masked_select(true_aspects.view(-1), active_labels).to(self.device)
        true_polarities = torch.masked_select(true_polarities.view(- 1), active_labels).to(self.device)
        dloss = self.calculate_dloss(aspect_tags, polarity_tags, true_aspects, true_polarities)
        return lloss + dloss, aspect_tags, polarity_tags


if __name__ == "__main__":
    loss_network = CTDLoss(None, None)
    ex_a = torch.LongTensor([ASPECT_LABELS.index(id) for id in [ 'O', 'O', 'B-PERFORMANCE', 'B-STORAGE',  'I-STORAGE',  'B-STORAGE',  'I-STORAGE']])
    ex_p = torch.LongTensor([POLARITY_LABELS.index(id) for id in ['O', 'B-NEGATIVE', 'O',             'I-NEGATIVE', 'B-POSITIVE', 'B-NEGATIVE', 'I-NEGATIVE']])
    print(loss_network.check_hierarchy(ex_p, ex_a))
