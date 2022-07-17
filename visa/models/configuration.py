from transformers import RobertaConfig

""" ABSA configuration"""

class ABSARoBERTaConfig(RobertaConfig):
    def __init__(self, num_aspect_labels: int = 21, num_polarity_labels: int = 4, device: str = 'cpu', **kwargs):
        """Constructs RobertaConfig."""
        super().__init__(num_aspect_labels=num_aspect_labels,
                         num_polarity_labels=num_polarity_labels,
                         device=device,
                         **kwargs)
