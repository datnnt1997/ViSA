from typing import Text, Union, Tuple, List
import os
import logging
import torch
import random
import numpy as np


def set_ramdom_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_total_model_parameters(model):
    total_params, trainable_params = 0, 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        if parameter.requires_grad:
            trainable_params += params
        total_params += params
    return total_params,


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("%(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        if not os.path.isdir('./logs'):
            os.makedirs('./logs')
        log_file = os.path.join('./logs/', log_file)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger

def split_tag(chunk_tag: Text) -> Union[Tuple[str, None], List[str]]:
    """
    Split chunk tag into IOBES prefix and chunk_type
    e.g. B-PER -> (B, PER)
        O -> (O, None)
    """
    if chunk_tag == 'O':
        return 'O', None
    return chunk_tag.split('-', maxsplit=1)


def is_chunk_end(prev_tag, tag):
    """
    check if the previous chunk ended between the previous and current word
    e.g.
    (B-PER, I-PER) -> False
    (B-LOC, O)  -> True

    Note: in case of contradicting tags, e.g. (B-PER, I-LOC)
    this is considered as (B-PER, B-LOC)
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix1 == 'O':
        return False
    if prefix2 == 'O':
        return prefix1 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']


def is_chunk_start(prev_tag, tag):
    """
    check if a new chunk started between the previous and current word
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix2 == 'O':
        return False
    if prefix1 == 'O':
        return prefix2 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']