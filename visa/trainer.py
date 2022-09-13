from tqdm import tqdm
from pathlib import Path
from prettytable import PrettyTable
from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import RandomSampler, DataLoader

from visa.constants import LOGGER, ASPECT_LABELS, POLARITY_LABELS
from visa.helper import set_ramdom_seed, get_total_model_parameters
from visa.dataset import build_dataset
from visa.models import ViSA_MODEL_ARCHIVE_MAP, ABSARoBERTaConfig
from visa.metrics import calc_score, calc_overall_score

import visa.arguments as arguments

import os
import torch
import sys
import datetime
import time
import itertools


def save_model(args, saved_file, model):
    saved_data = {
        'model': model.state_dict(),
        'a_classes': ASPECT_LABELS,
        'p_classes': POLARITY_LABELS,
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


def validate(model, task, iterator, cur_epoch: int, is_test: bool = False):
    start_time = time.time()
    model.eval()
    eval_loss = 0.0
    eval_aspect_golds, eval_polarity_golds, eval_aspect_preds, eval_polarity_preds = [], [], [], []
    # Run one step on sub-dataset
    with torch.no_grad():
        tqdm_desc = f"[{'TEST' if is_test else 'EVAL'}- Epoch {cur_epoch}]"
        eval_bar = tqdm(enumerate(iterator), total=len(iterator), desc=tqdm_desc)
        for idx, batch in eval_bar:
            outputs = model(**batch)
            eval_loss += outputs.loss.detach().item()
            active_accuracy = batch['label_masks'].view(-1) != 0
            a_labels = torch.masked_select(batch['a_labels'].view(-1), active_accuracy)
            p_labels = torch.masked_select(batch['p_labels'].view(-1), active_accuracy)
            eval_aspect_golds.extend(a_labels.detach().cpu().tolist())
            eval_polarity_golds.extend(p_labels.detach().cpu().tolist())
            if isinstance(outputs.aspects[-1], list):
                eval_aspect_preds.extend(list(itertools.chain(*outputs.aspects)))
            else:
                eval_aspect_preds.extend(outputs.aspects)
            if isinstance(outputs.polarities[-1], list):
                eval_polarity_preds.extend(list(itertools.chain(*outputs.polarities)))
            else:
                eval_polarity_preds.extend(outputs.polarities)

    epoch_loss = eval_loss / len(iterator)
    aspect_reports: dict = classification_report(eval_aspect_golds, eval_aspect_preds,
                                                 output_dict=True,
                                                 zero_division=0)
    senti_reports: dict = classification_report(eval_polarity_golds, eval_polarity_preds,
                                                output_dict=True,
                                                zero_division=0)
    epoch_aspect_avg_f1 = aspect_reports['macro avg']['f1-score']
    epoch_polarity_avg_f1 = senti_reports['macro avg']['f1-score']
    epoch_aspect_avg_acc = aspect_reports['accuracy']
    epoch_polarity_avg_acc = senti_reports['accuracy']
    LOGGER.info(f"\t{'*' * 20}{'Test' if is_test else 'Validate'} Summary{'*' * 20}")
    LOGGER.info(f"\tValidation Loss: {epoch_loss:.4f};")
    LOGGER.info(f"\tChunk-Report:")
    LOGGER.info(f"\t[Aspect]:")
    calc_score([ASPECT_LABELS[g_aid] for g_aid in eval_aspect_golds],
               [ASPECT_LABELS[p_aid] for p_aid in eval_aspect_preds], is_test=is_test)
    LOGGER.info(f"\t[Sentiment]:")
    calc_score([POLARITY_LABELS[g_sid] for g_sid in eval_polarity_golds],
               [POLARITY_LABELS[p_sid] for p_sid in eval_polarity_preds], is_test=is_test)
    LOGGER.info(f"\t[Aspect-Sentiment]:")
    overall_scores = calc_overall_score(true_apsects=[ASPECT_LABELS[g_aid] for g_aid in eval_aspect_golds],
                                        pred_apsects=[ASPECT_LABELS[p_aid] for p_aid in eval_aspect_preds],
                                        true_polarities=[POLARITY_LABELS[g_sid] for g_sid in eval_polarity_golds],
                                        pred_polarities=[POLARITY_LABELS[p_sid] for p_sid in eval_polarity_preds],
                                        is_test=is_test)
    LOGGER.info(f"\tBIO-Report:")
    LOGGER.info(f"\t[Aspect] Acc: {epoch_aspect_avg_acc:.4f}; macro-F1: {epoch_aspect_avg_f1:.4f};\n"
                f"\t[Sentiment] Acc: {epoch_polarity_avg_acc:.4f}; macro-F1: {epoch_polarity_avg_f1:.4f};\n"
                f"\tSpend time: {datetime.timedelta(seconds=(time.time() - start_time))}")
    return epoch_loss, overall_scores


def test():
    args = arguments.get_test_argument()
    LOGGER.info(f"Arguments: {args}")
    device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'
    assert os.path.exists(args.model_path), f'Checkpoint file `{args.model_path}` not exists!'
    if device == 'cpu':
        checkpoint_data = torch.load(args.model_path, map_location='cpu')
    else:
        checkpoint_data = torch.load(args.model_path)
    configs = checkpoint_data['args']
    use_crf = True if 'hier' in args.model_arch else False
    tokenizer = AutoTokenizer.from_pretrained(configs.model_name_or_path)

    config = ABSARoBERTaConfig.from_pretrained(args.model_name_or_path,
                                               num_aspect_labels=len(checkpoint_data['a_classes']),
                                               num_polarity_labels=len(checkpoint_data['p_classes']))
    model_clss = ViSA_MODEL_ARCHIVE_MAP[configs.model_arch]
    model = model_clss(config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    LOGGER.info("Load model trained weights")
    model.load_state_dict(checkpoint_data['model'])

    test_set = build_dataset(args.data_dir,
                             tokenizer,
                             dtype='test',
                             max_seq_len=args.max_seq_length,
                             device=device,
                             overwrite_data=args.overwrite_data,
                             use_crf=use_crf)
    test_iterator = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)
    # Summary
    total_params, _ = get_total_model_parameters(model)
    LOGGER.info(f"{'=' * 20}TESTER SUMMARY{'=' * 20}")
    summary_table = PrettyTable(["Parameters", "Values"])
    summary_table.add_rows([['Task', configs.task],
                            ['Model architecture', configs.model_arch],
                            ['Encoder name', configs.model_name_or_path],
                            ['Total params', total_params],
                            ['Model path', args.model_path],
                            ['Data dir', args.data_dir],
                            ['Number of examples', len(test_set)],
                            ['Max sequence length', configs.max_seq_length],
                            ['Test batch size', args.batch_size],
                            ['Number of workers', args.num_worker],
                            ['Use Cuda', not args.no_cuda],
                            ['Ovewrite dataset', args.overwrite_data]])
    LOGGER.info(summary_table)

    validate(model=model,
             task=args.task,
             iterator=test_iterator,
             is_test=True,
             cur_epoch=0)


def train():
    args = arguments.get_train_argument()
    LOGGER.info(f"Arguments: {args}")
    set_ramdom_seed(args.seed)
    device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'
    use_crf = True if 'hier' in args.model_arch else False
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    tensorboard_writer = SummaryWriter()
    assert os.path.isdir(args.data_dir), f'{args.data_dir} not found!'

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    train_set = build_dataset(args.data_dir,
                              tokenizer,
                              task=args.task,
                              dtype='train',
                              max_seq_len=args.max_seq_length,
                              device=device,
                              overwrite_data=args.overwrite_data,
                              use_crf=use_crf)

    eval_set = build_dataset(args.data_dir,
                             tokenizer,
                             task=args.task,
                             dtype='test',
                             max_seq_len=args.max_seq_length,
                             device=device,
                             overwrite_data=args.overwrite_data,
                             use_crf=use_crf)

    config = ABSARoBERTaConfig.from_pretrained(args.model_name_or_path,
                                               num_aspect_labels=len(ASPECT_LABELS),
                                               num_polarity_labels=len(POLARITY_LABELS))
    model = ViSA_MODEL_ARCHIVE_MAP[args.model_arch].from_pretrained(args.model_name_or_path, config=config)
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
    best_epoch = 0
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
        tensorboard_writer.add_scalar('TRAIN/Loss', tr_loss, epoch)
        # Validate trained model on dataset
        eval_loss, overall_scores = validate(model=model,
                                             task=args.task,
                                             iterator=eval_iterator,
                                             cur_epoch=epoch)

        tensorboard_writer.add_scalar('EVAL/Loss', eval_loss, epoch)
        tensorboard_writer.add_scalar('EVAL/micro-F1', overall_scores["micro"][-1], epoch)
        tensorboard_writer.add_scalar('EVAL/macro-F1', overall_scores["macro"][-1], epoch)
        LOGGER.info(f"\t{'*' * 20}Epoch Summary{'*' * 20}")
        LOGGER.info(f"\tEpoch Loss = {eval_loss:.6f} ; Best loss = {best_loss:.6f};")
        LOGGER.info(
            f"\tEpoch Overall-F1 score = {overall_scores['macro'][-1]:.6f} ; Best score = {best_score:.6f} "
            f"at Epoch-{best_epoch};")

        if eval_loss < best_loss:
            best_loss = eval_loss
        if overall_scores['macro'][-1] > best_score:
            best_epoch = epoch
            cumulative_early_steps = 0
            best_score = overall_scores['macro'][-1]
            saved_file = Path(args.output_dir + f"/best_model.pt")
            LOGGER.info(f"\t***New best model, saving to {saved_file}...***")
            save_model(args, saved_file, model)
        else:
            cumulative_early_steps += 1
    if args.run_test:
        test_set = build_dataset(args.data_dir,
                                 tokenizer,
                                 dtype='test',
                                 max_seq_len=args.max_seq_length,
                                 device=device,
                                 overwrite_data=args.overwrite_data,
                                 use_crf=use_crf)
        test_iterator = DataLoader(test_set, batch_size=args.eval_batch_size, num_workers=args.num_workers)
        if device == 'cpu':
            checkpoint_data = torch.load(args.output_dir + f"/best_model.pt", map_location='cpu')
        else:
            checkpoint_data = torch.load(args.output_dir + f"/best_model.pt")
        model.load_state_dict(checkpoint_data['model'])
        _, _ = validate(model=model,
                        task=args.task,
                        iterator=test_iterator,
                        is_test=True,
                        cur_epoch=0)


if __name__ == "__main__":
    if sys.argv[1] == 'train':
        LOGGER.info("[TRAINER] Start TRAIN process...")
        train()
    elif sys.argv[1] == 'test':
        LOGGER.info("[TRAINER] Start TEST process...")
        test()
    else:
        LOGGER.error(f'[ERROR] - `{sys.argv[1]}` not found!!!')
