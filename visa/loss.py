from typing import List
from .constants import ASPECT_LABELS, SENTIMENT_LABELS

import torch
import itertools


class HierarchicalLossNetwork:
    '''
        Logics to calculate the loss of the model.
    '''

    def __init__(self, aspect_func, senti_func, device='cpu', alpha=1, beta=0.8, p_loss=3):
        self.a_func = aspect_func
        self.s_func = senti_func
        self.alpha = alpha
        self.beta = beta
        self.p_loss = p_loss
        self.device = device

    def check_hierarchy(self, senti_task, aspect_task):
        '''
            Check if the predicted class at level l is a children of the class predicted at
            level l-1 for the entire batch.
        '''
        # check using the dictionary whether the current level's prediction
        # belongs to the superclass (prediction from the prev layer)
        seq_len = aspect_task.size()[0]
        bool_tensor = []
        for i in range(seq_len):
            if (ASPECT_LABELS[aspect_task[i].item()] != 'O' and SENTIMENT_LABELS[senti_task[i].item()] == 'O') or \
                    (ASPECT_LABELS[aspect_task[i].item()] == 'O' and SENTIMENT_LABELS[senti_task[i].item()] != 'O'):
                bool_tensor.append(True)
            else:
                bool_tensor.append(False)
        return torch.FloatTensor(bool_tensor).to(self.device)

    def calculate_tloss(self, aspects, sentis, true_a_labels, true_s_labels, mask=None):
        '''
            Calculates the task loss.
        '''
        a_loss = 1 - self.a_func(aspects, true_a_labels, mask=mask.type(torch.uint8))
        s_loss = 1 - self.s_func(sentis, true_s_labels, mask=mask.type(torch.uint8))
        tloss = a_loss + s_loss
        return self.alpha * tloss

    def calculate_dloss(self,
                        aspect_tags: List[int],
                        senti_tags: List[int],
                        true_a_labels: torch.LongTensor,
                        true_s_labels: torch.LongTensor):
        '''
            Calculate the dependence loss.
        '''

        dloss = 0
        aspect_tags = torch.LongTensor(aspect_tags).to(self.device)
        senti_tags = torch.LongTensor(senti_tags).to(self.device)
        D_l = self.check_hierarchy(senti_tags, aspect_tags)

        l_prev = torch.where(aspect_tags == true_a_labels, torch.FloatTensor([0]).to(self.device), torch.FloatTensor([1]).to(self.device))
        l_curr = torch.where(senti_tags == true_s_labels, torch.FloatTensor([0]).to(self.device), torch.FloatTensor([1]).to(self.device))
        dloss += torch.sum(torch.pow(self.p_loss, D_l * l_prev) * torch.pow(self.p_loss, D_l * l_curr) - 1)
        return self.beta * dloss

    def __call__(self, aspects, sentis, true_a_labels, true_s_labels, mask=None):
        lloss = self.calculate_tloss(aspects, sentis, true_a_labels, true_s_labels, mask)
        aspect_tags = list(itertools.chain(*self.a_func.decode(aspects, mask=mask != 0)))
        senti_tags =list(itertools.chain(*self.s_func.decode(sentis, mask=mask != 0)))
        active_labels = mask.view(-1) != 0
        true_a_labels = torch.masked_select(true_a_labels.view(-1), active_labels).to(self.device)
        true_s_labels = torch.masked_select(true_s_labels.view(-1), active_labels).to(self.device)
        dloss = self.calculate_dloss(aspect_tags, senti_tags, true_a_labels, true_s_labels)
        return lloss + dloss, aspect_tags, senti_tags