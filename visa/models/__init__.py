from .configuration import ABSARoBERTaConfig, ABSABERTConfig
from .hier_roberta_sl import HierRoBERTaSL
from .hier_roberta_ml import HierRoBERTaML
from .hier_bert_ml import HierBertML


ViSA_CONFIG_ARCHIVE_MAP = {
    "hier_roberta_sl": ABSARoBERTaConfig,
    "hier_roberta_ml": ABSARoBERTaConfig,
    'hier_bert_ml': ABSABERTConfig
}

ViSA_MODEL_ARCHIVE_MAP = {
    "hier_roberta_sl": HierRoBERTaSL,
    "hier_roberta_ml": HierRoBERTaML,
    'hier_bert_ml': HierBertML
}


__all__ = ["ABSARoBERTaConfig", "ViSA_MODEL_ARCHIVE_MAP", "ViSA_CONFIG_ARCHIVE_MAP"]
