# @Time    : 1/19/2020 11:10 AM
# @Author  : mikelkl
from __future__ import absolute_import, division, print_function
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

import logging
import collections
import json
import random
import re
import argparse
import os
import transformers.tokenization_roberta as tokenization
import torch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

AnswerType = {
    "UNKNOWN": 0,
    "YES": 1,
    "NO": 2,
    "SHORT": 3,
    "LONG": 4
}


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 answer_text="",
                 answer_type=AnswerType['SHORT']):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.answer_text = answer_text
        self.answer_type = answer_type


class InputLSFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 long_start_position=None,
                 long_end_position=None,
                 short_start_position=None,
                 short_end_position=None,
                 answer_text="",
                 answer_type=AnswerType['SHORT']):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.long_start_position = long_start_position
        self.long_end_position = long_end_position
        self.short_start_position = short_start_position
        self.short_end_position = short_end_position
        self.answer_text = answer_text
        self.answer_type = answer_type


def get_candidate_type(candidate_tokens):
    """Returns the candidate's type: Table, Paragraph, List or Other."""
    first_token = candidate_tokens[0]
    if first_token == "<Table>":
        return "Table"
    elif first_token == "<P>":
        return "Paragraph"
    elif first_token in ("<Ul>", "<Dl>", "<Ol>"):
        return "List"
    elif first_token in ("<Tr>", "<Li>", "<Dd>", "<Dt>"):
        return "Other"
    else:
        logger.warning("Unknoww candidate type found: %s", first_token)
        return "Other"


EX_TOKEN_MAP = json.load(open('roberta_large/ex_token_map.json'))


def roberta_tokenize(tokenizer, text):
    tokens = []
    for token in text.split(" "):
        if token in EX_TOKEN_MAP:
            tokens.append(token)
        else:
            sub_tokens = tokenizer._tokenize(token, add_prefix_space=True)
            tokens.extend(sub_tokens)
    return tokens


def roberta_convert_tokens_to_ids(tokenizer, tokens):
    output_ids = []
    for token in tokens:
        if token in EX_TOKEN_MAP:
            output_ids.append(EX_TOKEN_MAP[token])
        else:
            output_ids.append(tokenizer._convert_token_to_id(token))
    return output_ids


def check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def create_example(line, tfidf_dict, is_training, args):
    """
    Creates an NQ example from a given line of JSON.
    :param line: str
    :return: dict
    """
    sample = json.loads(line, object_pairs_hook=collections.OrderedDict)
    example_id = sample['example_id']
    question_text = sample['question_text']
    ori_doc_tokens = sample['document_text'].split()

    # 抽取特定段落list[list]
    tfidf_cands_ids = tfidf_dict[str(example_id)]
    # tfidf并不保证所有段落必定出现在所选段落内
    if is_training:
        long_answer_cand = sample['annotations'][0]['long_answer']['candidate_index']

        if long_answer_cand != -1:
            # answer_cand保证top_level是true
            if sample['long_answer_candidates'][long_answer_cand]['top_level'] is False:
                gt_start_token = sample['long_answer_candidates'][long_answer_cand]['start_token']
                gt_end_token = sample['long_answer_candidates'][long_answer_cand]['end_token']
                for il, cand in enumerate(sample['long_answer_candidates']):
                    if cand['start_token'] <= gt_start_token and cand['end_token'] >= gt_end_token \
                            and cand['top_level'] is True:
                        long_answer_cand = il
                        break
            # training的时候当tfidf中没有包含正确答案，且long_answer是存在的时候，tfidf的结果则只选目标段落
            hit_answer = False
            for pids in tfidf_cands_ids:
                if long_answer_cand in pids:
                    hit_answer = True
                    break
            if hit_answer is False:
                tfidf_cands_ids = [[]]
                token_count = 0
                for ic, cand in enumerate(sample['long_answer_candidates']):
                    if cand['top_level'] is True:
                        tfidf_cands_ids[-1].append(ic)
                        token_count += (cand['end_token'] - cand['start_token'])
                        if token_count > 600:
                            tfidf_cands_ids.append([])
                            token_count = 0
                while len(tfidf_cands_ids[-1]) == 0:
                    tfidf_cands_ids.pop(-1)
                # 防止负样本爆炸，只选目标段落
                tfidf_cands_ids = [cands for cands in tfidf_cands_ids if long_answer_cand in cands]

    # 由于接下来要对special_tokens排序，所以这里tfidf选择的段落要按照首段排序
    tfidf_cands_ids = sorted(tfidf_cands_ids, key=lambda x: x[0])

    if args.do_combine:  # 如果do_combine，我们把所有抽取的candidates合并到一起
        tfidf_cands_ids_ = []
        for c in tfidf_cands_ids:
            tfidf_cands_ids_.extend(c)
        tfidf_cands_ids = [tfidf_cands_ids_]

    # 获取candidate的type信息，去除HTML符号
    # 保留特殊token到段首
    # 注意table paragraph list最小起步是1
    special_tokens_count = {'ContextId': -1, 'Table': 0, 'Paragraph': 0, 'List': 0}

    # 为了保证一致性，TABLE, Paragraph等结构信息还是尽可能保留...
    selected_ps = []
    for i, cand_ids in enumerate(tfidf_cands_ids):
        position_map = []  # 新paragraph到老paragraph的token位置映射
        map_to_origin = {}  # 为了保证能够对答案位置进行正确的偏移，这里需要重新搞一波map映射
        p_tokens = []
        for cand_id in cand_ids:
            st = sample['long_answer_candidates'][cand_id]['start_token']
            ed = sample['long_answer_candidates'][cand_id]['end_token']
            ind = st  # 追踪pos_map
            ori_cand_tokens = ori_doc_tokens[st:ed]
            # 先加ContextId特殊token
            special_tokens_count['ContextId'] += 1
            special_tokens_count['ContextId'] = min(special_tokens_count['ContextId'], args.max_position)
            p_tokens.append('[ContextId={}]'.format(special_tokens_count['ContextId']))
            position_map.append(ind)
            cand_type = get_candidate_type(ori_cand_tokens)
            if cand_type in special_tokens_count:
                special_tokens_count[cand_type] += 1
                special_tokens_count[cand_type] = min(special_tokens_count[cand_type], args.max_position)
                p_tokens.append('[' + cand_type + '=' + str(special_tokens_count[cand_type]) + ']')
                position_map.append(ind)
            for token in ori_cand_tokens:
                if '<' not in token:  # 去除HTML符号
                    p_tokens.append(token)
                    position_map.append(ind)
                map_to_origin[ind] = len(position_map) - 1
                ind += 1
            assert len(position_map) == len(p_tokens)

        selected_ps.append({'paragraph_tokens': p_tokens,
                            'question_text': question_text,
                            'position_map': position_map,
                            'map_to_origin': map_to_origin,
                            'example_id': example_id,
                            'paragraph_id': str(example_id) + '_' + str(i),
                            'answer_type': AnswerType['UNKNOWN'],
                            'long_start': -1,
                            'long_end': -1,
                            'short_start': -1,
                            'short_end': -1,
                            'short_answer_text': None})

    answer = None
    answer_text = None
    if is_training and 'annotations' in sample:
        # 答案只取第一个标注
        annotation = sample['annotations'][0]
        if annotation is not None:
            long_answer = annotation['long_answer']
            if long_answer['candidate_index'] != -1:
                answer_type = AnswerType['LONG']
                ori_long_start = long_answer['start_token']
                ori_long_end = long_answer['end_token']
            else:
                answer_type = AnswerType['UNKNOWN']
                ori_long_start = -1
                ori_long_end = -1

            assert annotation["yes_no_answer"] in ("YES", "NO", "NONE")
            if annotation["yes_no_answer"] == 'YES':
                answer_text = 'YES'
                answer_type = AnswerType['YES']
            elif annotation["yes_no_answer"] == 'NO':
                answer_text = 'NO'
                answer_type = AnswerType['NO']

            short_answers = annotation['short_answers']
            # 这里short answer必须排序
            short_answers = sorted(short_answers, key=lambda x: x['start_token'])
            if len(short_answers) > 0:
                # TODO:可能存在多个short，multi-tag
                answer_type = AnswerType['SHORT']
                short_ans = random.choice(short_answers)
                ori_short_start = short_ans['start_token']
                ori_short_end = short_ans['end_token']
                answer_text = ori_doc_tokens[ori_short_start:ori_short_end]
                answer_text = " ".join([at for at in answer_text if '<' not in at])
            else:
                ori_short_start = -1
                ori_short_end = -1
        else:
            answer_type = AnswerType['UNKNOWN']
            ori_long_start = -1
            ori_long_end = -1
            ori_short_start = -1
            ori_short_end = -1

        answer = {'answer_type': answer_type,
                  'ori_long_start': ori_long_start,
                  'ori_long_end': ori_long_end,
                  'ori_short_start': ori_short_start,
                  'ori_short_end': ori_short_end}

        if answer['answer_type'] == AnswerType['SHORT'] and answer_text == "":
            print('WRONG SHORT', answer, answer_text)
            answer['answer_type'] = AnswerType['LONG']
            answer['ori_short_start'] = -1
            answer['ori_short_end'] = -1

    examples = []
    for p_sample in selected_ps:
        if answer and answer['answer_type'] != AnswerType['UNKNOWN']:
            # 如果长答案在候选里，那么首位必然都在这个候选里，!!!注意这里的ori_long_end必须-1，否则可能会漏!!!
            if answer['ori_long_start'] in p_sample['map_to_origin'] \
                    and answer['ori_long_end'] - 1 in p_sample['map_to_origin']:
                final_long_start = p_sample['map_to_origin'][answer['ori_long_start']]
                final_long_end = p_sample['map_to_origin'][answer['ori_long_end'] - 1] + 1
                long_answer_text = " ".join(p_sample['paragraph_tokens'][final_long_start:final_long_end])

                p_sample['answer_type'] = answer['answer_type']
                p_sample['long_start'] = final_long_start
                p_sample['long_end'] = final_long_end

                # 短答案必然在长答案所在段落里面
                if answer['answer_type'] == AnswerType['SHORT']:
                    final_short_start = p_sample['map_to_origin'][answer['ori_short_start']]
                    final_short_end = p_sample['map_to_origin'][answer['ori_short_end'] - 1] + 1
                    p_sample['short_start'] = final_short_start
                    p_sample['short_end'] = final_short_end

                    new_answer_text = " ".join(p_sample['paragraph_tokens'][final_short_start:final_short_end])
                    assert new_answer_text == answer_text, (new_answer_text, answer_text, long_answer_text)
                    p_sample['short_answer_text'] = new_answer_text

            # 由于negative的段落太多了，所以这里先过滤掉一部分
            elif is_training and random.random() > args.example_neg_filter:
                continue

        # 由于negative的段落太多了，所以这里先过滤掉一部分
        elif is_training and random.random() > args.example_neg_filter:
            continue

        p_sample.pop('map_to_origin')
        examples.append(p_sample)

    return examples


def read_nq_examples(input_file, tfidf_dict, is_training, args):
    """
    Read a NQ json file into a list of NqExample.
    """
    all_examples = []
    positive_paragraphs = 0
    negative_paragraphs = 0
    logger.info("Reading: %s", input_file)
    with open(input_file, "r") as f:
        for index, line in tqdm(enumerate(f)):
            new_examples = create_example(line, tfidf_dict, is_training, args)
            if is_training:
                for example in new_examples:
                    if example['answer_type'] == AnswerType['UNKNOWN']:
                        negative_paragraphs += 1
                    else:
                        positive_paragraphs += 1
                if index % 5000 == 0:
                    print('Positive paragraphs:', positive_paragraphs, 'Negative paragraphs:', negative_paragraphs)
            all_examples.extend(new_examples)
    return all_examples


def convert_examples_to_features(examples, tokenizer, is_training, args):
    """Converts a list of NqExamples into InputFeatures."""
    all_features = []
    positive_features = 0
    negative_features = 0
    logger.info("Converting a list of NqExamples into InputFeatures ...")
    for index, example in enumerate(tqdm(examples)):
        example_index = example['example_id']
        paragraph_id = example['paragraph_id']
        assert args.do_ls is True
        features = convert_single_ls_example(example, tokenizer, is_training, args)

        for feature in features:
            feature.example_index = example_index
            feature.unique_id = paragraph_id + '_' + str(feature.doc_span_index)
            all_features.append(feature)
            if is_training:
                if feature.answer_type == AnswerType['UNKNOWN']:
                    negative_features += 1
                else:
                    positive_features += 1

        if is_training and index % 5000 == 0:
            print('Positive features:', positive_features, 'Negative features:', negative_features)

    print('Positive features:', positive_features, 'Negative features:', negative_features)

    return all_features


# def convert_examples_to_features_(example, tokenizer, is_training, args):
#     """
#     Converts a list of NqExamples into InputFeatures.
#     Args:
#         examples: list
#         tokenizer:
#         is_training: bool
#         args:
#
#     Returns:
#         features_: list
#         postive_incremental: int
#         negative_incremental: int
#     """
#     features_ = []
#     example_index = example['example_id']
#     paragraph_id = example['paragraph_id']
#     if args.do_ls:
#         features = convert_single_ls_example(example, tokenizer, is_training, args)
#     else:
#         raise ValueError("Must do_ls")
#     negative_incremental, postive_incremental = 0, 0
#     for feature in features:
#         feature.example_index = example_index
#         feature.unique_id = paragraph_id + '_' + str(feature.doc_span_index)
#         features_.append(feature)
#         if is_training:
#             if feature.answer_type == AnswerType['UNKNOWN']:
#                 negative_incremental += 1
#             else:
#                 postive_incremental += 1
#     return features_, postive_incremental, negative_incremental

# def convert_examples_to_features(examples, tokenizer, is_training, args):
#     """
#     Parallelly Converts a list of NqExamples into InputFeatures.
#     Args:
#         examples: list
#         tokenizer:
#         is_training: bool
#         args:
#
#     Returns:
#         all_features: list
#     """
#     """Converts a list of NqExamples into InputFeatures."""
#     all_features = []
#     positive_features = 0
#     negative_features = 0
#     logger.info("Converting a list of NqExamples into InputFeatures ...")
#     index = 0
#     with Pool(args.num_workers) as pool:
#         with tqdm(total=len(examples), initial=0) as pbar:
#             # For very long iterables using a large value for chunksize can make the job complete much faster than using the default value of 1.
#             for features_, postive_incremental, negative_incremental in pool.imap(partial(convert_examples_to_features_, tokenizer=tokenizer, is_training=is_training, args=args), examples, chunksize=1000):
#                 all_features.extend(features_)
#                 positive_features += postive_incremental
#                 negative_features += negative_incremental
#                 pbar.update()
#
#                 if is_training and (index) % 5000 == 0:
#                     print('Positive features:', positive_features, 'Negative features:', negative_features)
#                 index += 1
#
#     print('Positive features:', positive_features, 'Negative features:', negative_features)
#
#     return all_features


def convert_single_ls_example(example, tokenizer, is_training, args):
    """Converts a single NqExample into a list of InputFeatures."""
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []  # all subtokens of original doc after tokenizing
    features = []
    for (i, token) in enumerate(example['paragraph_tokens']):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = roberta_tokenize(tokenizer, token)
        tok_to_orig_index.extend([i] * len(sub_tokens))
        all_doc_tokens.extend(sub_tokens)
    orig_to_tok_index.append(len(all_doc_tokens))

    # 特别注意！由于在paragraph_tokens中我们的token已经映射过一次了
    # 这里wordpiece等于又映射了一遍，所以这里的操作是二次映射
    if example['position_map']:
        tok_to_orig_index = [example['position_map'][index] for index in tok_to_orig_index]

    # QUERY
    query_tokens = []
    query_tokens.append("[Q]")
    query_tokens.extend(roberta_tokenize(tokenizer, example['question_text']))
    if len(query_tokens) > args.max_query_length:
        query_tokens = query_tokens[-args.max_query_length:]

    # ANSWER 预处理的时候先长短分开
    tok_long_start_position = -1
    tok_long_end_position = -1
    tok_short_start_position = -1
    tok_short_end_position = -1
    # 这里终点是必然在para_tokens内的
    if is_training:
        if example['answer_type'] != AnswerType['UNKNOWN']:
            tok_long_start_position = orig_to_tok_index[example['long_start']]
            tok_long_end_position = orig_to_tok_index[example['long_end']] - 1
        if example['answer_type'] == AnswerType['SHORT']:
            tok_short_start_position = orig_to_tok_index[example['short_start']]
            tok_short_end_position = orig_to_tok_index[example['short_end']] - 1

    # Get max tokens number for original doc,
    # should minus query tokens number and 3 special tokens
    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = args.max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset  # compute number of tokens remaining unsliding
        length = min(length, max_tokens_for_doc)  # determine current sliding window size
        doc_spans.append(_DocSpan(start=start_offset, length=length))

        # Consider case for reaching end of original doc
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, args.doc_stride)

    # Convert window + query + special tokens to feature
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        tokens.append("<cls>")
        segment_ids.append(0)  # roberta的segment_id全0
        tokens.extend(query_tokens)
        segment_ids.extend([0] * len(query_tokens))
        tokens.append("<sep>")
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = check_is_max_context(doc_spans, doc_span_index, split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(0)
        tokens.append("<sep>")
        segment_ids.append(0)
        assert len(tokens) == len(segment_ids)

        input_ids = roberta_convert_tokens_to_ids(tokenizer, tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        # roberta的mask为1
        one_padding = [1] * (args.max_seq_length - len(input_ids))
        zero_padding = [0] * (args.max_seq_length - len(input_ids))
        input_ids.extend(one_padding)
        input_mask.extend(zero_padding)
        segment_ids.extend(zero_padding)

        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length

        long_start_position = None
        long_end_position = None
        short_start_position = None
        short_end_position = None
        answer_type = None
        answer_text = ""
        if is_training:
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            contains_an_annotation = (tok_long_start_position >= doc_start and tok_long_end_position <= doc_end)
            # 负样本需要经过采样，且目标为[CLS]
            if (not contains_an_annotation) or example['answer_type'] == AnswerType['UNKNOWN']:
                if args.include_unknowns < 0 or random.random() > args.include_unknowns:
                    continue
                long_start_position = 0
                long_end_position = 0
                short_start_position = 0
                short_end_position = 0
                answer_type = AnswerType['UNKNOWN']
            else:
                doc_offset = len(query_tokens) + 2
                long_start_position = tok_long_start_position - doc_start + doc_offset
                long_end_position = tok_long_end_position - doc_start + doc_offset
                if example['answer_type'] == AnswerType['SHORT']:
                    short_start_position = tok_short_start_position - doc_start + doc_offset
                    short_end_position = tok_short_end_position - doc_start + doc_offset
                else:
                    short_start_position = 0
                    short_end_position = 0
                answer_type = example['answer_type']

                # 如果是短答案，对一下答案是否正确
                if example['answer_type'] == AnswerType['SHORT']:
                    answer_text = " ".join(tokens[short_start_position:(short_end_position + 1)])
                    answer_text = answer_text.replace(' ', '').replace(u"Ġ", ' ').strip()
                    gt_answer = example['short_answer_text']
                    answer_text_chars = [c for c in answer_text if c not in " \t\r\n" and ord(c) != 0x202F]
                    gt_answer_chars = [c for c in gt_answer if c not in " \t\r\n" and ord(c) != 0x202F]
                    if "".join(answer_text_chars) != "".join(gt_answer_chars) \
                            and len("".join(answer_text_chars)) != len("".join(gt_answer_chars)):
                        print(answer_text, 'V.S.', gt_answer)

        feature = InputLSFeatures(
            unique_id=None,
            example_index=None,
            doc_span_index=doc_span_index,
            tokens=tokens,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            long_start_position=long_start_position,
            long_end_position=long_end_position,
            short_start_position=short_start_position,
            short_end_position=short_end_position,
            answer_text=answer_text,
            answer_type=answer_type)

        features.append(feature)

    return features


if __name__ == '__main__':
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default='data/simplified-nq-train.jsonl', type=str,
                        help="NQ json for training. E.g., simplified-nq-train.jsonl")
    # parser.add_argument("--train_file", default='data/tiny-train.jsonl', type=str,
    #                     help="NQ json for training. E.g., simplified-nq-train.jsonl")
    parser.add_argument("--dev_file", default='data/simplified-nq-dev.jsonl', type=str,
                        help="NQ json for predictions. E.g., simplified-nq-test.jsonl")
    parser.add_argument("--test_file", default='data/simplified-nq-test.jsonl', type=str,
                        help="NQ json for predictions. E.g., simplified-nq-test.jsonl")
    parser.add_argument("--output_dir", default='dataset', type=str)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--seed", default=556, type=int)
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_position", type=int, default=50,
                        help="Maximum context position for which to generate special tokens.")
    parser.add_argument("--example_neg_filter", type=float, default=0.2,
                        help="If positive, probability of including answers of type `UNKNOWN`.")
    parser.add_argument("--include_unknowns", type=float, default=0.138,
                        help="If positive, probability of including answers of type `UNKNOWN`.")
    parser.add_argument("--skip_nested_contexts", type=bool, default=True,
                        help="Completely ignore context that are not top level nodes in the page.")
    parser.add_argument("--do_ls", type=bool, default=True,
                        help="Whether to use long short index labels?")
    parser.add_argument("--do_combine", type=bool, default=False,
                        help="Whether to combine all remained examples from each line?")
    parser.add_argument("--tfidf_train_file", type=str, default='dataset/train_cand_selected_600.json')
    parser.add_argument("--tfidf_dev_file", type=str, default='dataset/dev_cand_selected_600.json')
    parser.add_argument("--tfidf_test_file", type=str, default='dataset/test_cand_selected_600.json')
    parser.add_argument("--num_workers", default=16, type=int,
                        help="Number of processes for parallel preprocessing")

    args = parser.parse_args()

    # 由于combine了以后一个example会非常长，所以会额外多出很多负样本，这里我们吧neg example的过滤复合到feat里
    # if args.do_combine:
    #     args.include_unknowns = args.include_unknowns * args.example_neg_filter

    random.seed(args.seed)
    tokenizer = tokenization.RobertaTokenizer(vocab_file='roberta_large/vocab.json',
                                              merges_file='roberta_large/merges.txt')
    # train preprocess
    example_output_file = os.path.join(args.output_dir,
                                       'train_data_maxlen{}_tfidf_examples.json'.format(args.max_seq_length))
    # example_output_file = os.path.join(args.output_dir,
    #                                    'tiny_train_data_maxlen{}_tfidf_examples.json'.format(args.max_seq_length))
    feature_output_file = os.path.join(args.output_dir,
                                       'train_data_maxlen{}_includeunknowns{}_roberta_tfidf_features.bin'.format(
                                           args.max_seq_length, args.include_unknowns))
    if args.do_ls:
        feature_output_file = feature_output_file.replace('_features', '_ls_features')
    if args.do_combine:
        example_output_file = example_output_file.replace('_examples', '_combine_examples')
        feature_output_file = feature_output_file.replace('_features', '_combine_features')
    if not os.path.exists(feature_output_file):
        tfidf_dict = json.load(open(args.tfidf_train_file))
        if os.path.exists(example_output_file):
            print("Reading: %s" % (example_output_file))
            examples = json.load(open(example_output_file))
        else:
            examples = read_nq_examples(input_file=args.train_file, tfidf_dict=tfidf_dict, is_training=True, args=args)
            with open(example_output_file, 'w') as w:
                json.dump(examples, w)
        features = convert_examples_to_features(examples=examples, tokenizer=tokenizer, is_training=True, args=args)
        torch.save(features, feature_output_file)

    # dev preprocess
    example_output_file = os.path.join(args.output_dir,
                                       'dev_data_maxlen{}_tfidf_examples.json'.format(args.max_seq_length))
    feature_output_file = os.path.join(args.output_dir,
                                       'dev_data_maxlen{}_roberta_tfidf_features.bin'.format(args.max_seq_length))
    if args.do_ls:
        feature_output_file = feature_output_file.replace('_features', '_ls_features')
    if args.do_combine:
        example_output_file = example_output_file.replace('_examples', '_combine_examples')
        feature_output_file = feature_output_file.replace('_features', '_combine_features')
    if not os.path.exists(feature_output_file):
        tfidf_dict = json.load(open(args.tfidf_dev_file))
        if os.path.exists(example_output_file):
            examples = json.load(open(example_output_file))
        else:
            examples = read_nq_examples(input_file=args.dev_file, tfidf_dict=tfidf_dict, is_training=False, args=args)
            with open(example_output_file, 'w') as w:
                json.dump(examples, w)
        features = convert_examples_to_features(examples=examples, tokenizer=tokenizer, is_training=False, args=args)
        torch.save(features, feature_output_file)

    # test preprocess
    example_output_file = os.path.join(args.output_dir,
                                       'test_data_maxlen{}_tfidf_examples.json'.format(args.max_seq_length))
    feature_output_file = os.path.join(args.output_dir,
                                       'test_data_maxlen{}_roberta_tfidf_features.bin'.format(args.max_seq_length))
    if args.do_ls:
        feature_output_file = feature_output_file.replace('_features', '_ls_features')
    if args.do_combine:
        example_output_file = example_output_file.replace('_examples', '_combine_examples')
        feature_output_file = feature_output_file.replace('_features', '_combine_features')
    if not os.path.exists(feature_output_file):
        tfidf_dict = json.load(open(args.tfidf_test_file))
        if os.path.exists(example_output_file):
            examples = json.load(open(example_output_file))
        else:
            examples = read_nq_examples(input_file=args.test_file, tfidf_dict=tfidf_dict, is_training=False, args=args)
            with open(example_output_file, 'w') as w:
                json.dump(examples, w)
        features = convert_examples_to_features(examples=examples, tokenizer=tokenizer, is_training=False, args=args)
        torch.save(features, feature_output_file)

    # # Reset segment_id as 0 for roberta
    # import torch
    # paths = ["/users/liukanglong/projects/TF2-QA/ewrfcas/dataset/train_data_maxlen512_roberta_tfidf_ls_features.bin",
    #          "/users/liukanglong/projects/TF2-QA/ewrfcas/dataset/dev_data_maxlen512_roberta_tfidf_ls_features.bin",
    #          "/users/liukanglong/projects/TF2-QA/ewrfcas/dataset/test_data_maxlen512_roberta_tfidf_ls_features.bin"]
    # for path in paths:
    #     print("Processing:", path)
    #     feat = torch.load(path)
    #     for i, _ in tqdm(enumerate(feat)):
    #         feat[i].segment_ids=[0]*512
    #     torch.save(feat, path)

    # import torch
    # path = "/users/liukanglong/projects/TF2-QA/ewrfcas/dataset/dev_data_maxlen512_roberta_tfidf_ls_features.bin"
    # feat = torch.load(path)
