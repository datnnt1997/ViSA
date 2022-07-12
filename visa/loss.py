import torch
import torch.nn as nn


class HierarchicalLossNetwork:
    '''
        Logics to calculate the loss of the model.
    '''

    def __init__(self, id_to_labels, id_to_coarse_labels, numeric_hierarchy, device='cpu', alpha=1, beta=0.8, p_loss=3):
        self.alpha = alpha
        self.beta = beta
        self.p_loss = p_loss
        self.device = device
        self.id_to_labels = id_to_labels
        self.id_to_coarse_labels = id_to_coarse_labels  # ["partial", "raw", "safe"]
        self.numeric_hierarchy = numeric_hierarchy  # {"partial": ["bikini", "miniskirt", "chest"], "raw": ["raw"], "safe": ["safe"]}

    def __getattr__(self, item):
        return None

    def check_hierarchy(self, current_level, previous_level):
        '''
            Check if the predicted class at level l is a children of the class predicted at
            level l-1 for the entire batch.
        '''
        # check using the dictionary whether the current level's prediction
        # belongs to the superclass (prediction from the prev layer)
        bool_tensor = [not self.id_to_labels[current_level[i].item()] in self.numeric_hierarchy[self.id_to_coarse_labels[previous_level[i].item()]]
                       for i in range(previous_level.size()[0])]
        return torch.FloatTensor(bool_tensor).to(current_level.device)

    def calculate_lloss(self, coarse, gran, true_coarse_labels, true_gran_labels):
        '''
            Calculates the layer loss.
        '''

        loss_func = nn.CrossEntropyLoss()
        lloss = loss_func(coarse, true_coarse_labels) + loss_func(gran, true_gran_labels)
        return self.alpha * lloss

    def calculate_dloss(self, coarse, gran, true_coarse_labels, true_gran_labels):
        '''
            Calculate the dependence loss.
        '''

        dloss = 0
        current_lvl_pred = torch.argmax(nn.Softmax(dim=1)(gran), dim=1)
        prev_lvl_pred = torch.argmax(nn.Softmax(dim=1)(coarse), dim=1)
        D_l = self.check_hierarchy(current_lvl_pred, prev_lvl_pred)

        l_prev = torch.where(prev_lvl_pred == true_coarse_labels, torch.FloatTensor([0]).to(true_coarse_labels.device),
                             torch.FloatTensor([1]).to(true_coarse_labels.device))
        l_curr = torch.where(current_lvl_pred == true_gran_labels, torch.FloatTensor([0]).to(true_gran_labels.device),
                             torch.FloatTensor([1]).to(true_gran_labels.device))
        dloss += torch.sum(torch.pow(self.p_loss, D_l * l_prev) * torch.pow(self.p_loss, D_l * l_curr) - 1)
        return self.beta * dloss

    def __call__(self, predictions, true_gran_labels, true_coarse_labels):
        coarse, gran = predictions
        lloss = self.calculate_lloss(coarse, gran, true_coarse_labels, true_gran_labels)
        dloss = self.calculate_dloss(coarse, gran, true_coarse_labels, true_gran_labels)
        return lloss + dloss