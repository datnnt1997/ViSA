from typing import Union
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import RandomSampler, DataLoader

from visa.constants import LOGGER, ASPECT_LABELS, SENTIMENT_LABELS
from visa.helper import set_ramdom_seed
from visa.arguments import get_train_argument
from visa.dataset import build_dataset
from visa.model import ABSAConfig, ABSAModel

import os
import torch
import sys
import datetime
import time
import itertools


def save_model(args, saved_file, model):
    saved_data = {
        'model': model.state_dict(),
        'classes': args.label2id,
        'args': args
    }
    torch.save(saved_data, saved_file)


def train_one_epoch(model, iterator, optim, cur_epoch: int, max_grad_norm: float = 1.0, scheduler=None):
    start_time = time.time()
    tr_loss = 0.0
    model.train()
    tqdm_bar = tqdm(enumerate(iterator), total=len(iterator), desc=f'[TRAIN-EPOCH {cur_epoch}]')
    for idx, batch in tqdm_bar:
        outputs = model(**batch)
        # backward pass
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        optim.zero_grad()
        outputs.loss.backward()
        optim.step()
        if scheduler:
            scheduler.step()
        tr_loss += outputs.loss.detach().item()
    epoch_loss = tr_loss / len(iterator)
    LOGGER.info(f"\t{'*' * 20}Train Summary{'*' * 20}")
    LOGGER.info(f"\tTraining Lr: {optim.param_groups[0]['lr']}; "
                f"Loss: {epoch_loss:.4f}; "
                f"Spend time: {datetime.timedelta(seconds=(time.time() - start_time))}")
    return epoch_loss


def validate(model, task, iterator, cur_epoch: int, output_dir: Union[str, os.PathLike] = './', is_test=False):
    start_time = time.time()
    model.eval()
    eval_loss = 0.0
    eval_aspect_golds, eval_senti_golds, eval_aspect_preds, eval_senti_preds = [], [], [], []
    # Run one step on sub-dataset
    with torch.no_grad():
        tqdm_desc = f'[EVAL- Epoch {cur_epoch}]'
        eval_bar = tqdm(enumerate(iterator), total=len(iterator), desc=tqdm_desc)
        for idx, batch in eval_bar:
            outputs = model(**batch)
            eval_loss += outputs.loss.detach().item()
            active_accuracy = batch['label_masks'].view(-1) != 0
            a_labels = torch.masked_select(batch['a_labels'].view(-1), active_accuracy)
            s_labels = torch.masked_select(batch['s_labels'].view(-1), active_accuracy)
            eval_aspect_golds.extend(a_labels.detach().cpu().tolist())
            eval_senti_golds.extend(s_labels.detach().cpu().tolist())
            if isinstance(outputs.a_tags[-1], list):
                eval_aspect_preds.extend(list(itertools.chain(*outputs.a_tags)))
            else:
                eval_aspect_preds.extend(outputs.a_tags)
            if isinstance(outputs.s_tags[-1], list):
                eval_senti_preds.extend(list(itertools.chain(*outputs.s_tags)))
            else:
                eval_senti_preds.extend(outputs.s_tags)
    epoch_loss = eval_loss / len(iterator)
    aspect_reports: dict = classification_report(eval_aspect_golds, eval_aspect_preds,
                                                 output_dict=True,
                                                 zero_division=0)
    senti_reports: dict = classification_report(eval_senti_golds, eval_senti_preds,
                                                output_dict=True,
                                                zero_division=0)
    epoch_aspect_avg_f1 = aspect_reports['macro avg']['f1-score']
    epoch_senti_avg_f1 = senti_reports['macro avg']['f1-score']
    epoch_aspect_avg_acc = aspect_reports['accuracy']
    epoch_senti_avg_acc = senti_reports['accuracy']
    LOGGER.info(f"\t{'*' * 20}Validate Summary{'*' * 20}")
    LOGGER.info(f"\tValidation Loss: {epoch_loss:.4f};\n"
                f"\t[Aspect] BIO-Accuracy: {epoch_aspect_avg_acc:.4f}; BIO-Macro-F1 score: {epoch_aspect_avg_f1:.4f};\n"
                f"\t[Sentiment] BIO-Accuracy: {epoch_senti_avg_acc:.4f}; BIO-Macro-F1 score: {epoch_senti_avg_f1:.4f};\n"
                f"\tSpend time: {datetime.timedelta(seconds=(time.time() - start_time))}")
    return epoch_loss, (epoch_aspect_avg_f1, epoch_aspect_avg_acc), (epoch_senti_avg_f1, epoch_senti_avg_acc)


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

    config = ABSAConfig.from_pretrained(args.model_name_or_path,
                                        num_slabels=len(SENTIMENT_LABELS),
                                        num_alabels=len(ASPECT_LABELS))
    model = ABSAModel.from_pretrained(args.model_name_or_path, config=config)
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
    encoder_param_optimizer = list(model.roberta.named_parameters())
    task_param_optimizer = [(n, p) for n, p in model.named_parameters() if "roberta" not in n]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in encoder_param_optimizer if not any(nd in n for nd in no_decay)],
         'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        {'params': [p for n, p in encoder_param_optimizer if any(nd in n for nd in no_decay)],
         'lr': args.learning_rate, 'weight_decay': 0.0},
        {'params': [p for n, p in task_param_optimizer if not any(nd in n for nd in no_decay)],
         'lr': args.classifier_learning_rate, 'weight_decay': args.weight_decay},
        {'params': [p for n, p in task_param_optimizer if any(nd in n for nd in no_decay)],
         'lr': args.classifier_learning_rate, 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    train_steps_per_epoch = len(train_set) // args.train_batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=train_steps_per_epoch,
                                                num_training_steps=args.epochs * train_steps_per_epoch)
    train_sampler = RandomSampler(train_set)
    train_iterator = DataLoader(train_set,
                                sampler=train_sampler,
                                batch_size=args.train_batch_size,
                                num_workers=args.num_workers)
    eval_iterator = DataLoader(eval_set, batch_size=args.eval_batch_size, num_workers=args.num_workers)
    best_score = 0.0
    best_loss = float('inf')
    cumulative_early_steps = 0
    for epoch in range(int(args.epochs)):
        if cumulative_early_steps > args.early_stop:
            LOGGER.info(f"Early stopping. Check your saved model.")
            break
        LOGGER.info(f"\n{'=' * 30}Training epoch {epoch}{'=' * 30}")
        # Fit model with dataset
        tr_loss = train_one_epoch(model=model,
                                  optim=optimizer,
                                  iterator=train_iterator,
                                  cur_epoch=epoch,
                                  max_grad_norm=args.max_grad_norm,
                                  scheduler=scheduler)

        # Validate trained model on dataset
        eval_loss, aspect_scores, senti_scores = validate(model=model,
                                                task=args.task,
                                                iterator=eval_iterator,
                                                cur_epoch=epoch,
                                                is_test=False)
        #
        # LOGGER.info(f"\t{'*' * 20}Epoch Summary{'*' * 20}")
        # LOGGER.info(f"\tEpoch Loss = {eval_loss:.6f} ; Best loss = {best_loss:.6f}")
        # LOGGER.info(f"\tEpoch BIO-F1 score = {eval_f1:.6f} ; Best score = {best_score:.6f}")
        #
        # if eval_loss < best_loss:
        #     best_loss = eval_loss
        # if eval_f1 > best_score:
        #     cumulative_early_steps = 0
        #     best_score = eval_f1
        #     saved_file = Path(args.output_dir + f"/best_model.pt")
        #     LOGGER.info(f"\t***New best model, saving to {saved_file}...***")
        #     save_model(args, saved_file, model)
        # else:
        #     cumulative_early_steps += 1


if __name__ == "__main__":
    if sys.argv[1] == 'train':
        LOGGER.info("[TRAINER] Start TRAIN process...")
        train()
    elif sys.argv[1] == 'test':
        LOGGER.info("[TRAINER] Start TEST process...")
        test()
    else:
        LOGGER.error(f'[ERROR] - `{sys.argv[1]}` not found!!!')