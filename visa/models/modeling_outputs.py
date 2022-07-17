from typing import Optional, Any, Tuple, List
from collections import OrderedDict

import torch

class ModelOutput(OrderedDict):
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
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())


class ABSAOutput(ModelOutput):
    """
    Base class for outputs of Span detection for aspect-based sentiment analysis.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Controllable task-dependency loss is the weighted sum of a task loss and a dependency loss.
        t_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Task loss.
        d_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            dependency loss.
        aspects (`List(int)`):
            List of predicted aspect ids.
        polarities (`List(int)`):
            List of predicted polarities ids.
    """
    loss: Optional[torch.FloatTensor] = None
    t_loss: Optional[torch.FloatTensor] = None
    d_loss: Optional[torch.FloatTensor] = None
    aspects: Optional[List[int]] = None
    polarities: Optional[List[int]] = None
