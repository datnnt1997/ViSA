from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from visa.constants import LOGGER
from visa.helper import set_ramdom_seed
from visa.arguments import get_train_argument
from visa.dataset import build_dataset
from visa.model import ABSAConfig, ABSAModel

import os
import torch
import sys


def save_model(args, saved_file, model):
    saved_data = {
        'model': model.state_dict(),
        'classes': args.label2id,
        'args': args
    }
    torch.save(saved_data, saved_file)

def train_one_epoch():
    NotImplemented

def validate():
    NotImplemented

def test():
    NotImplemented

def train():
    args = get_train_argument()
    LOGGER.info(f"Arguments: {args}")
    set_ramdom_seed(args.seed)
    device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'
    use_crf = True if 'crf' in args.model_arch else False
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    assert os.path.isdir(args.data_dir), f'{args.data_dir} not found!'

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_set = build_dataset(args.data_dir,
                              tokenizer,
                              dtype='train',
                              max_seq_len=args.max_seq_length,
                              device=device,
                              overwrite_data=args.overwrite_data,
                              use_crf=use_crf)
    eval_set = build_dataset(args.data_dir,
                             tokenizer,
                             dtype='test',
                             max_seq_len=args.max_seq_length,
                             device=device,
                             overwrite_data=args.overwrite_data,
                             use_crf=use_crf)

    config = ABSAConfig.from_pretrained(args.model_name_or_pathj, )
    model = ABSAModel.from_pretrained(args.model_name_or_path, config=config)
    model.resize_position_embeddings(len(tokenizer))
    model.to(device)

    if args.load_weights is not None:
        LOGGER.info(f'Load pretrained model weights from "{args.load_weights}"')
        if device == 'cpu':
            checkpoint_data = torch.load(args.load_weights, map_location='cpu')
        else:
            checkpoint_data = torch.load(args.load_weights)
        model.load_state_dict(checkpoint_data['model'])
        checkpoint_data = None

    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    model_param_optimizer = list(model.named_parameters())
    bert_param_optimizer = list(model.roberta.named_parameters())

    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model_param_optimizer if any(nd in n for nd in bert_param_optimizer)],
         'lr': args.classifier_learning_rate, 'weight_decay': args.weight_decay}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    train_steps_per_epoch = len(train_set) // args.train_batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=train_steps_per_epoch,
                                                num_training_steps=args.epochs * train_steps_per_epoch)

if __name__ == "__main__":
    if sys.argv[1] == 'train':
        LOGGER.info("[TRAINER] Start TRAIN process...")
        train()
    elif sys.argv[1] == 'test':
        LOGGER.info("[TRAINER] Start TEST process...")
        test()
    else:
        LOGGER.error(f'[ERROR] - `{sys.argv[1]}` not found!!!')