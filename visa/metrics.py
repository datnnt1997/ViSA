from typing import Optional, Text, Tuple, Union, List
from collections import defaultdict

from .constants import LOGGER
from .helper import split_tag, is_chunk_start, is_chunk_end


def count_chunks(true_seqs, pred_seqs):
    """
        true_seqs: a list of true tags
        pred_seqs: a list of predicted tags

        return:
        correct_chunks: a dict (counter),
                        key = chunk types,
                        value = number of correctly identified chunks per type
        true_chunks:    a dict, number of true chunks per type
        pred_chunks:    a dict, number of identified chunks per type

        correct_counts, true_counts, pred_counts: similar to above, but for tags
    """
    correct_chunks = defaultdict(int)
    true_chunks = defaultdict(int)
    pred_chunks = defaultdict(int)

    correct_counts = defaultdict(int)
    true_counts = defaultdict(int)
    pred_counts = defaultdict(int)

    prev_true_tag, prev_pred_tag = 'O', 'O'
    correct_chunk = None

    for true_tag, pred_tag in zip(true_seqs, pred_seqs):
        if true_tag == pred_tag:
            correct_counts[true_tag] += 1
        true_counts[true_tag] += 1
        pred_counts[pred_tag] += 1

        _, true_type = split_tag(true_tag)
        _, pred_type = split_tag(pred_tag)

        if correct_chunk is not None:
            true_end = is_chunk_end(prev_true_tag, true_tag)
            pred_end = is_chunk_end(prev_pred_tag, pred_tag)

            if pred_end and true_end:
                correct_chunks[correct_chunk] += 1
                correct_chunk = None
            elif pred_end != true_end or true_type != pred_type:
                correct_chunk = None

        true_start = is_chunk_start(prev_true_tag, true_tag)
        pred_start = is_chunk_start(prev_pred_tag, pred_tag)

        if true_start and pred_start and true_type == pred_type:
            correct_chunk = true_type
        if true_start:
            true_chunks[true_type] += 1
        if pred_start:
            pred_chunks[pred_type] += 1

        prev_true_tag, prev_pred_tag = true_tag, pred_tag
    if correct_chunk is not None:
        correct_chunks[correct_chunk] += 1

    return (correct_chunks, true_chunks, pred_chunks,
            correct_counts, true_counts, pred_counts)


def merge_tags(aspect_tags, polarity_tags):
    merged_tags = []
    prev_a_tag, prev_ptag = 'O', 'O'
    for a_tag, ptag in zip(aspect_tags, polarity_tags):
        if a_tag == 'O' or ptag == 'O':
            merged_tags.append('O')
        else:
            _, a_type = split_tag(a_tag)
            _, s_type = split_tag(ptag)

            a_start = is_chunk_start(prev_a_tag, a_tag)
            s_start = is_chunk_start(prev_ptag, ptag)
            if a_start or s_start:
                merged_tag = f"B-{a_type}#{s_type}"
            else:
                merged_tag = f"I-{a_type}#{s_type}"

            merged_tags.append(merged_tag)

        prev_a_tag, prev_ptag = a_tag, ptag
    return merged_tags


def calc_macro_metrics(tp, p, t, percent=True):
    """
        Compute overall precision, recall and FB1 (default values are 0.0)
        if percent is True, return 100 * original decimal value
    """
    macro_precision = tp / p if p else 0
    macro_recall = tp / t if t else 0
    macro_fb1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall) if macro_precision + macro_recall else 0

    if percent:
        return 100 * macro_precision, 100 * macro_recall, 100 * macro_fb1
    else:
        return macro_precision, macro_recall, macro_fb1


def get_result(correct_chunks, true_chunks, pred_chunks,
               correct_counts, true_counts, pred_counts, verbose=True, is_test=False):
    sum_correct_chunks = sum(correct_chunks.values())
    sum_true_chunks = sum(true_chunks.values())
    sum_pred_chunks = sum(pred_chunks.values())

    sum_correct_counts = sum(correct_counts.values())
    sum_true_counts = sum(true_counts.values())

    chunk_types = sorted(list(set(list(true_chunks) + list(pred_chunks))))
    micro_prec, micro_rec, micro_f1 = calc_macro_metrics(sum_correct_chunks, sum_pred_chunks, sum_true_chunks, percent=False)
    res = {"micro": (micro_prec, micro_rec, micro_f1), "marco": ()}

    sum_prec, sum_rec, sum_f1 = 0.0, 0.0, 0.0
    for t in chunk_types:
        prec, rec, f1 = calc_macro_metrics(correct_chunks[t], pred_chunks[t], true_chunks[t], percent=False)
        if is_test:
            LOGGER.info(f"\t  {t:17s}: P: {prec:0.4f}; R: {rec:0.4f}; F1: {f1:0.4f}  {pred_chunks[t]}")
        sum_prec += prec
        sum_rec += rec
        sum_f1 += f1
    macro_prec, macro_rec, macro_f1 = (sum_prec/len(chunk_types), sum_rec/len(chunk_types), sum_f1/len(chunk_types))
    res["macro"] = (macro_prec, macro_rec, macro_f1)
    if is_test:
        LOGGER.info(f"\t Acc: {sum_correct_counts / sum_true_counts:0.4f}; "
                    f"micro-P: {micro_prec:0.4f}; micro-R: {micro_rec:0.4f}; micro-F1: {micro_f1:0.4f}; "
                    f"macro-P: {macro_prec:0.4f}; macro-R: {macro_rec:0.4f}; macro-F1: {macro_f1:0.4f}")
    else:
        LOGGER.info(f"\t\tAcc: {sum_correct_counts / sum_true_counts:0.4f}; micro-F1: {micro_f1:0.4f}; "
                    f"macro-F1: {macro_f1:0.4f}")
    return res


def calc_score(true_seqs, pred_seqs, verbose=True, is_test=False):
    correct_chunks, true_chunks, pred_chunks, correct_counts, true_counts, pred_counts = count_chunks(true_seqs,
                                                                                                      pred_seqs)
    result = get_result(correct_chunks, true_chunks, pred_chunks, correct_counts, true_counts, pred_counts,
                        verbose=verbose, is_test=is_test)
    return result


def calc_overall_score(true_apsects, pred_apsects, true_polarities, pred_polarities, verbose=True, is_test=False):
    true_seqs = merge_tags(true_apsects, true_polarities)
    pred_seqs = merge_tags(pred_apsects, pred_polarities)
    correct_chunks, true_chunks, pred_chunks, correct_counts, true_counts, pred_counts = count_chunks(true_seqs,
                                                                                                      pred_seqs)
    result = get_result(correct_chunks, true_chunks, pred_chunks, correct_counts, true_counts, pred_counts,
                        verbose=verbose, is_test=is_test)
    return result
