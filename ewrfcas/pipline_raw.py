# STEP1 将数据集去HTML化，然后按照字数阈值（600）划分存储

from tqdm import tqdm
import json


def split_data(input_dir, output_dir, token_limit=600, is_training=False):
    para_splited_data = []
    with open(input_dir, 'r') as f:
        for line in tqdm(f):
            temp_data = json.loads(line)
            context = temp_data['document_text']
            doc_tokens = context.split()
            cands = temp_data['long_answer_candidates']
            split_tokens = []
            for i, cand in enumerate(cands):
                if cand['top_level'] is True:
                    split_tokens.append({'cand_id': i, 'start': cand['start_token'], 'end': cand['end_token'],
                                         'doc_tokens': doc_tokens[cand['start_token']:cand['end_token']]})

            # 去除html元素
            # 构成段落，这里以800词分割，即累积超过800词，重开一段。但是要注意吧cand_id信息也对应存入
            paras = []
            for i in range(len(split_tokens)):
                split_tokens[i]['doc_tokens'] = [t for t in split_tokens[i]['doc_tokens'] if '<' not in t]
                paras.append({'cand_id': split_tokens[i]['cand_id'], 'para': " ".join(split_tokens[i]['doc_tokens'])})

            split_paras = []
            new_para = {'para': "", 'cand_ids': []}
            for para in paras:
                new_para['cand_ids'].append(para['cand_id'])
                if new_para['para'] != "":
                    new_para['para'] += " "
                new_para['para'] += para['para']
                if len(new_para['para'].split()) > token_limit:
                    split_paras.append(new_para)
                    new_para = {'para': "", 'cand_ids': []}
            if len(new_para['cand_ids']) > 0:
                split_paras.append(new_para)

            # 由于答案所在的cand不一定是top_level，我们要把答案所在cand映射到top_level所在的cand里
            if is_training:
                annotations = temp_data['annotations'][0]  # TODO:先取第一个？
                gt_cand_id = annotations['long_answer']['candidate_index']
                if gt_cand_id == -1:
                    true_cand_id = -1
                else:
                    true_cand_id = None
                    if cands[gt_cand_id]['top_level'] is False:
                        start = annotations['long_answer']['start_token']
                        end = annotations['long_answer']['end_token']
                        for spt in split_tokens:
                            if spt['start'] <= start and spt['end'] >= end:
                                true_cand_id = spt['cand_id']
                                break
                    else:
                        true_cand_id = gt_cand_id

                assert true_cand_id is not None
            else:
                true_cand_id = -1

            para_splited_data.append({
                'example_id': temp_data['example_id'],
                'question_text': temp_data['question_text'],
                'true_cand_id': true_cand_id,
                'split_paras': split_paras
            })

    with open(output_dir, 'w') as w:
        json.dump(para_splited_data, w, indent=2)


split_data('data/simplified-nq-test.jsonl', 'dataset/test_splited_600.json', token_limit=600, is_training=False)

# STEP2 将分段后的语料选出top8
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# top1: 51%
# top3: 82.4%
# top5: 91.6%
# top8: 96.7%
# top10: 98%
topk = 8

with open('data/simplified-nq-test.jsonl', 'r') as f:
    para_splited_data = json.load(f)

# 停用词
stopwords = set()
with open("dataset/stopwords.txt", 'r') as f:
    for line in f:
        stopwords.add(line.strip())

total_valid_num = 0
hit_num = 0

tfidf_cand_select = {}

with tqdm(total=len(para_splited_data), desc='Building, Top{}'.format(topk)) as pbar:
    for para_sample in para_splited_data:
        question_text = para_sample['question_text'].lower()
        question_text = [qt for qt in question_text.split() if qt not in stopwords]
        question_text = " ".join(question_text)

        cv = CountVectorizer()
        tfidf = TfidfTransformer()

        paras = para_sample['split_paras']
        gt = para_sample['true_cand_id']
        paras_text = [p['para'].lower() for p in paras]
        corpus = [question_text]
        corpus.extend(paras_text)

        words = cv.fit_transform(corpus)
        question_indices = words[0].indices
        tfidf_scores = tfidf.fit_transform(words)
        tfidf_scores = tfidf_scores.toarray()[1:]
        tfidf_scores = np.sum(tfidf_scores[:, question_indices], axis=1)

        best_para_ids = np.argsort(tfidf_scores)[::-1]
        best_para_ids = best_para_ids[:topk]

        best_paras = []
        pred_cand_ids = []
        cand_set = set()
        for best_para_id in best_para_ids:
            best_paras.append(paras[best_para_id])
            pred_cand_ids.append(paras[best_para_id]['cand_ids'])
            for ci in pred_cand_ids[-1]:
                cand_set.add(ci)

        if gt != -1:  # 统计tfidf截取段落后准确率
            total_valid_num += 1
            if gt in cand_set:
                hit_num += 1

        pbar.set_postfix({'Acc': '{0:1.5f}'.format(hit_num / (total_valid_num + 1e-5))})
        pbar.update(1)

        tfidf_cand_select[para_sample['example_id']] = pred_cand_ids

with open("dataset/test_cand_selected_600.json", 'w') as w:
    json.dump(tfidf_cand_select, w, indent=2)

# STEP3 预处理
import logging
import collections
import json
import random
import re
import argparse
import os
import albert_tokenization as tokenization
import torch

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


# A special token in NQ is made of non-space chars enclosed in square brackets.
_SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)


def albert_tokenize(tokenizer, text):
    tokens = []
    for token in text.split(" "):
        if _SPECIAL_TOKENS_RE.match(token) or token in tokenizer.EX_TOKEN_MAP:
            if token in tokenizer.vocab or token in tokenizer.EX_TOKEN_MAP:
                tokens.append(token)
            else:
                tokens.append(tokenizer.wordpiece_tokenizer.unk_token)
        else:
            sub_tokens = tokenization.encode_pieces(
                tokenizer.sp_model,
                tokenization.preprocess_text(token, lower=True),
                return_unicode=False)
            tokens.extend(sub_tokens)
    return tokens


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
        if args.do_ls:
            features = convert_single_ls_example(example, tokenizer, is_training, args)
        else:
            features = convert_single_example(example, tokenizer, is_training, args)

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


def convert_single_example(example, tokenizer, is_training, args):
    """Converts a single NqExample into a list of InputFeatures."""
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []  # all subtokens of original doc after tokenizing
    features = []
    for (i, token) in enumerate(example['paragraph_tokens']):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = albert_tokenize(tokenizer, token)
        tok_to_orig_index.extend([i] * len(sub_tokens))
        all_doc_tokens.extend(sub_tokens)

    # 特别注意！由于在paragraph_tokens中我们的token已经映射过一次了
    # 这里wordpiece等于又映射了一遍，所以这里的操作是二次映射
    if example['position_map']:
        tok_to_orig_index = [example['position_map'][index] for index in tok_to_orig_index]

    # QUERY
    query_tokens = []
    query_tokens.append("[Q]")
    query_tokens.extend(albert_tokenize(tokenizer, example['question_text']))
    if len(query_tokens) > args.max_query_length:
        query_tokens = query_tokens[-args.max_query_length:]

    # ANSWER 预处理的时候先长短分开
    tok_start_position = -1
    tok_end_position = -1
    # 这里终点是必然在para_tokens内的
    if is_training:
        # 现阶段，有短答案预测短答案，否则预测长答案
        if example['answer_type'] != AnswerType['UNKNOWN']:
            tok_long_start_position = orig_to_tok_index[example['long_start']]
            if example['long_end'] == len(orig_to_tok_index):
                tok_long_end_position = orig_to_tok_index[-1]
            else:
                tok_long_end_position = orig_to_tok_index[example['long_end']] - 1
            tok_start_position = tok_long_start_position
            tok_end_position = tok_long_end_position
        if example['answer_type'] == AnswerType['SHORT']:
            tok_short_start_position = orig_to_tok_index[example['short_start']]
            if example['short_end'] == len(orig_to_tok_index):
                tok_short_end_position = orig_to_tok_index[-1]
            else:
                tok_short_end_position = orig_to_tok_index[example['short_end']] - 1
            tok_start_position = tok_short_start_position
            tok_end_position = tok_short_end_position

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
        tokens.append("[CLS]")
        segment_ids.append(0)
        tokens.extend(query_tokens)
        segment_ids.extend([0] * len(query_tokens))
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = check_is_max_context(doc_spans, doc_span_index, split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        assert len(tokens) == len(segment_ids)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (args.max_seq_length - len(input_ids))
        input_ids.extend(padding)
        input_mask.extend(padding)
        segment_ids.extend(padding)

        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length

        start_position = None
        end_position = None
        answer_type = None
        answer_text = ""
        if is_training:
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            contains_an_annotation = (tok_start_position >= doc_start and tok_end_position <= doc_end)
            # 负样本需要经过采样，且目标为[CLS]
            if (not contains_an_annotation) or example['answer_type'] == AnswerType['UNKNOWN']:
                if args.include_unknowns < 0 or random.random() > args.include_unknowns:
                    continue
                start_position = 0
                end_position = 0
                answer_type = AnswerType['UNKNOWN']
            else:
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
                answer_type = example['answer_type']

                # 如果是短答案，对一下答案是否正确
                if example['answer_type'] == AnswerType['SHORT']:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    answer_text = answer_text.replace(' ', '').replace(u"▁", ' ').strip()
                    gt_answer = example['short_answer_text'].lower()
                    answer_text_chars = [c for c in answer_text if c not in " \t\r\n" and ord(c) != 0x202F]
                    gt_answer_chars = [c for c in gt_answer if c not in " \t\r\n" and ord(c) != 0x202F]
                    if "".join(answer_text_chars) != "".join(gt_answer_chars):
                        print(answer_text, 'V.S.', gt_answer)

        feature = InputFeatures(
            unique_id=None,
            example_index=None,
            doc_span_index=doc_span_index,
            tokens=tokens,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            start_position=start_position,
            end_position=end_position,
            answer_text=answer_text,
            answer_type=answer_type)

        features.append(feature)

    return features


def convert_single_ls_example(example, tokenizer, is_training, args):
    """Converts a single NqExample into a list of InputFeatures."""
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []  # all subtokens of original doc after tokenizing
    features = []
    for (i, token) in enumerate(example['paragraph_tokens']):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = albert_tokenize(tokenizer, token)
        tok_to_orig_index.extend([i] * len(sub_tokens))
        all_doc_tokens.extend(sub_tokens)

    # 特别注意！由于在paragraph_tokens中我们的token已经映射过一次了
    # 这里wordpiece等于又映射了一遍，所以这里的操作是二次映射
    if example['position_map']:
        tok_to_orig_index = [example['position_map'][index] for index in tok_to_orig_index]

    # QUERY
    query_tokens = []
    query_tokens.append("[Q]")
    query_tokens.extend(albert_tokenize(tokenizer, example['question_text']))
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
            if example['long_end'] == len(orig_to_tok_index):
                tok_long_end_position = orig_to_tok_index[-1]
            else:
                tok_long_end_position = orig_to_tok_index[example['long_end']] - 1
        if example['answer_type'] == AnswerType['SHORT']:
            tok_short_start_position = orig_to_tok_index[example['short_start']]
            if example['short_end'] == len(orig_to_tok_index):
                tok_short_end_position = orig_to_tok_index[-1]
            else:
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
        tokens.append("[CLS]")
        segment_ids.append(0)
        tokens.extend(query_tokens)
        segment_ids.extend([0] * len(query_tokens))
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = check_is_max_context(doc_spans, doc_span_index, split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        assert len(tokens) == len(segment_ids)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (args.max_seq_length - len(input_ids))
        input_ids.extend(padding)
        input_mask.extend(padding)
        segment_ids.extend(padding)

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
                    answer_text = answer_text.replace(' ', '').replace(u"▁", ' ').strip()
                    gt_answer = example['short_answer_text'].lower()
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


# parameters
parser = argparse.ArgumentParser()
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
parser.add_argument("--include_unknowns", type=float, default=0.025,
                    help="If positive, probability of including answers of type `UNKNOWN`.")
parser.add_argument("--skip_nested_contexts", type=bool, default=True,
                    help="Completely ignore context that are not top level nodes in the page.")
parser.add_argument("--do_ls", type=bool, default=True,
                    help="Whether to use long short index labels?")
parser.add_argument("--do_combine", type=bool, default=False,
                    help="Whether to combine all remained examples from each line?")
parser.add_argument("--tfidf_test_file", type=str, default='dataset/test_cand_selected_600.json')

args = parser.parse_args(args=[])

random.seed(args.seed)
tokenizer = tokenization.FullTokenizer(
    vocab_file='albert_xxlarge/30k-clean.vocab', do_lower_case=True,
    spm_model_file='albert_xxlarge/30k-clean.model')

# test preprocess
example_output_file = os.path.join(args.output_dir,
                                   'test_data_maxlen{}_tfidf_examples.json'.format(args.max_seq_length))
feature_output_file = os.path.join(args.output_dir,
                                   'test_data_maxlen{}_albert_tfidf_features.bin'.format(args.max_seq_length))
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

# STEP4 构建模型推断
import torch
import argparse
from albert_modeling import AlbertConfig, AlBertJointForNQ2
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os
import json
import collections
import pickle
from utils_nq import read_candidates_from_one_split, compute_pred_dict, InputLSFeatures
import pandas as pd

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id",
                                    "long_start_topk_logits", "long_start_topk_index",
                                    "long_end_topk_logits", "long_end_topk_index",
                                    "short_start_topk_logits", "short_start_topk_index",
                                    "short_end_topk_logits", "short_end_topk_index",
                                    "long_cls_logits", "short_cls_logits",
                                    "answer_type_logits"])


def evaluate(model, args, dev_features, device, ei):
    # Eval!
    if os.path.exists(os.path.join(args.output_dir, 'test_RawResults_ensemble{}.pkl'.format(ei))):
        all_results = pickle.load(
            open(os.path.join(args.output_dir, 'test_RawResults_ensemble{}.pkl'.format(ei)), 'rb'))
    else:
        all_results = []
        for batch in tqdm(eval_dataloader, desc="Evaluating Ensemble-{}".format(ei)):
            model.eval()
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                input_ids, input_mask, segment_ids, example_indices = batch
                inputs = {'input_ids': input_ids,
                          'attention_mask': input_mask,
                          'token_type_ids': segment_ids}
                outputs = model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = dev_features[example_index.item()]
                unique_id = str(eval_feature.unique_id)

                result = RawResult(unique_id=unique_id,
                                   # [topk]
                                   long_start_topk_logits=outputs['long_start_topk_logits'][i].cpu().numpy(),
                                   long_start_topk_index=outputs['long_start_topk_index'][i].cpu().numpy(),
                                   long_end_topk_logits=outputs['long_end_topk_logits'][i].cpu().numpy(),
                                   long_end_topk_index=outputs['long_end_topk_index'][i].cpu().numpy(),
                                   # [topk, topk]
                                   short_start_topk_logits=outputs['short_start_topk_logits'][i].cpu().numpy(),
                                   short_start_topk_index=outputs['short_start_topk_index'][i].cpu().numpy(),
                                   short_end_topk_logits=outputs['short_end_topk_logits'][i].cpu().numpy(),
                                   short_end_topk_index=outputs['short_end_topk_index'][i].cpu().numpy(),
                                   answer_type_logits=to_list(outputs['answer_type_logits'][i]),
                                   long_cls_logits=outputs['long_cls_logits'][i].cpu().numpy(),
                                   short_cls_logits=outputs['short_cls_logits'][i].cpu().numpy())
                all_results.append(result)

        pickle.dump(all_results, open(os.path.join(args.output_dir, 'test_RawResults_ensemble{}.pkl'.format(ei)), 'wb'))

    candidates_dict = read_candidates_from_one_split(args.predict_file)
    nq_pred_dict = compute_pred_dict(candidates_dict, dev_features,
                                     [r._asdict() for r in all_results],
                                     args.n_best_size, args.max_answer_length, topk_pred=True,
                                     long_n_top=5, short_n_top=5, ensemble=True)

    output_prediction_file = os.path.join(args.output_dir, 'test_predictions{}.json'.format(ei))
    with open(output_prediction_file, 'w') as f:
        json.dump({'predictions': list(nq_pred_dict.values())}, f)


def load_cached_data(feature_dir, output_features=False, evaluate=False):
    features = torch.load(feature_dir)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    else:
        all_long_start_positions = torch.tensor([f.long_start_position for f in features], dtype=torch.long)
        all_long_end_positions = torch.tensor([f.long_end_position for f in features], dtype=torch.long)
        all_short_start_positions = torch.tensor([f.short_start_position for f in features], dtype=torch.long)
        all_short_end_positions = torch.tensor([f.short_end_position for f in features], dtype=torch.long)
        all_answer_types = torch.tensor([f.answer_type for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_long_start_positions, all_long_end_positions,
                                all_short_start_positions, all_short_end_positions,
                                all_answer_types)

    if output_features:
        return dataset, features
    return dataset


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def make_submission(output_prediction_file, output_dir):
    print("***** Making submmision *****")
    test_answers_df = pd.read_json(output_prediction_file)

    def create_short_answer(entry):
        """
        :param entry: dict
        :return: str
        """
        if entry['answer_type'] == 0:
            return ""

        if entry["yes_no_answer"] != "NONE":
            return entry["yes_no_answer"]

        answer = []
        for short_answer in entry["short_answers"]:
            if short_answer["start_token"] > -1:
                answer.append(str(short_answer["start_token"]) + ":" + str(short_answer["end_token"]))
        return " ".join(answer)

    def create_long_answer(entry):
        if entry['answer_type'] == 0:
            return ''

        answer = []
        if entry["long_answer"]["start_token"] > -1:
            answer.append(str(entry["long_answer"]["start_token"]) + ":" + str(entry["long_answer"]["end_token"]))
        return " ".join(answer)

    for var_name in ['long_answer_score', 'short_answers_score', 'answer_type']:
        test_answers_df[var_name] = test_answers_df['predictions'].apply(lambda q: q[var_name])

    test_answers_df["long_answer"] = test_answers_df["predictions"].apply(create_long_answer)
    test_answers_df["short_answer"] = test_answers_df["predictions"].apply(create_short_answer)
    test_answers_df["example_id"] = test_answers_df["predictions"].apply(lambda q: str(q["example_id"]))

    long_answers = dict(zip(test_answers_df["example_id"], test_answers_df["long_answer"]))
    short_answers = dict(zip(test_answers_df["example_id"], test_answers_df["short_answer"]))

    sample_submission = pd.read_csv("data/sample_submission.csv")

    long_prediction_strings = sample_submission[sample_submission["example_id"].str.contains("_long")].apply(
        lambda q: long_answers[q["example_id"].replace("_long", "")], axis=1)
    short_prediction_strings = sample_submission[sample_submission["example_id"].str.contains("_short")].apply(
        lambda q: short_answers[q["example_id"].replace("_short", "")], axis=1)

    sample_submission.loc[
        sample_submission["example_id"].str.contains("_long"), "PredictionString"] = long_prediction_strings
    sample_submission.loc[
        sample_submission["example_id"].str.contains("_short"), "PredictionString"] = short_prediction_strings

    sample_submission.to_csv(os.path.join(output_dir, "submission.csv"), index=False)

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_ids", default="0", type=str)
parser.add_argument("--eval_batch_size", default=16, type=int)
parser.add_argument("--n_best_size", default=20, type=int)
parser.add_argument("--max_answer_length", default=30, type=int)
parser.add_argument("--unk_th", default=1.0, type=float, help='unk_thresholds')
parser.add_argument("--float16", default=False, type=bool)

parser.add_argument("--predict_file", default='data/simplified-nq-test.jsonl', type=str)
parser.add_argument("--bert_config_file", default='check_points/albert-xxlarge-tfidf-600-top8-V0', type=str)
parser.add_argument("--init_restore_dirs", default=['check_points/albert-xxlarge-tfidf-600-top8-V0',
                                                    'check_points/albert-xxlarge-tfidf-600-top8-V01',
                                                    'check_points/albert-xxlarge-tfidf-600-top8-V02',
                                                    'check_points/albert-xxlarge-tfidf-600-top8-V03'], type=list)
parser.add_argument("--output_dir", default='check_points/albert-xxlarge-V0-ensemble', type=str)
parser.add_argument("--dev_feat_dir", default='dataset/test_data_maxlen512_albert_tfidf_ls_features.bin', type=str)

args = parser.parse_args(args=[])
args.bert_config_file = os.path.join('albert_xxlarge', 'albert_config.json')
init_dirs = [os.path.join(dir, 'best_checkpoint.pth') for dir in args.init_restore_dirs]
args.log_file = os.path.join(args.output_dir, args.log_file)
os.makedirs(args.output_dir, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
device = torch.device("cuda")
n_gpu = torch.cuda.device_count()
print("device %s n_gpu %d" % (device, n_gpu))
print("device: {} n_gpu: {} 16-bits training: {}".format(device, n_gpu, args.float16))

# Loading data
print('Loading data...')
dev_dataset, dev_features = load_cached_data(feature_dir=args.dev_feat_dir, output_features=True, evaluate=True)
eval_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.eval_batch_size)

bert_config = AlbertConfig.from_json_file(args.bert_config_file)

for i, init_dir in enumerate(init_dirs):
    print('Load weights from', init_dir)
    model = AlBertJointForNQ2(bert_config)
    model.load_state_dict(torch.load(init_dir, map_location='cpu'), strict=True)
    if args.float16:
        model.half()
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    output_prediction_file = os.path.join(args.output_dir, 'test_predictions{}.json'.format(i))
    if not os.path.exists(output_prediction_file):
        evaluate(model, args, dev_features, device, i)
    else:
        print(output_prediction_file, 'exists, skip...')

# get the final score
from glob import glob
import numpy as np
import enum


class AnswerType(enum.IntEnum):
    """Type of NQ answer."""
    UNKNOWN = 0
    YES = 1
    NO = 2
    SHORT = 3
    LONG = 4


all_preds = glob(args.output_dir + '/test_predictions*.json')
ensemble_pred_dict = {}
for pred_file in all_preds:
    with open(pred_file) as f:
        preds = json.load(f)['predictions']
        for pred in tqdm(preds):
            example_id = pred['example_id']
            long_answer = pred['long_answer']
            short_answer = pred['short_answers'][0]
            answer_type_logits = pred['answer_type_logits']
            if example_id not in ensemble_pred_dict:
                ensemble_pred_dict[example_id] = {
                    "long_answer_dict": {
                        (long_answer['start_token'], long_answer['end_token']): pred['long_answer_score']},
                    "short_answer_dict": {
                        (short_answer['start_token'], short_answer['end_token']): pred['short_answers_score']},
                    "answer_type_logits": np.array(answer_type_logits).astype(np.float64)}
            else:
                if (long_answer['start_token'], long_answer['end_token']) in ensemble_pred_dict[example_id][
                    'long_answer_dict']:
                    ensemble_pred_dict[example_id]['long_answer_dict'][
                        (long_answer['start_token'], long_answer['end_token'])] += pred['long_answer_score']
                else:
                    ensemble_pred_dict[example_id]['long_answer_dict'][
                        (long_answer['start_token'], long_answer['end_token'])] = pred['long_answer_score']

                if (short_answer['start_token'], short_answer['end_token']) in ensemble_pred_dict[example_id][
                    'short_answer_dict']:
                    ensemble_pred_dict[example_id]['short_answer_dict'][
                        (short_answer['start_token'], short_answer['end_token'])] += pred['short_answers_score']
                else:
                    ensemble_pred_dict[example_id]['short_answer_dict'][
                        (short_answer['start_token'], short_answer['end_token'])] = pred['short_answers_score']
                ensemble_pred_dict[example_id]['answer_type_logits'] += np.array(answer_type_logits).astype(
                    np.float64)

final_preds = []
for exp_id in ensemble_pred_dict:
    long_answer_result = sorted(
        [(sted, score) for sted, score in ensemble_pred_dict[exp_id]['long_answer_dict'].items()],
        key=lambda x: x[1], reverse=True)[0]
    short_answer_result = sorted(
        [(sted, score) for sted, score in ensemble_pred_dict[exp_id]['short_answer_dict'].items()],
        key=lambda x: x[1], reverse=True)[0]
    answer_type_logits = ensemble_pred_dict[exp_id]['answer_type_logits']
    answer_type_logits[0] *= args.unk_th
    answer_type = int(np.argmax(answer_type_logits))
    if answer_type == AnswerType.YES:
        yes_no_answer = "YES"
    elif answer_type == AnswerType.NO:
        yes_no_answer = "NO"
    else:
        yes_no_answer = "NONE"

    long_start = int(long_answer_result[0][0]) if answer_type != AnswerType.UNKNOWN else -1
    long_end = int(long_answer_result[0][1]) if answer_type != AnswerType.UNKNOWN else -1

    short_start = int(short_answer_result[0][0]) if answer_type == AnswerType.SHORT else -1
    short_end = int(short_answer_result[0][1]) if answer_type == AnswerType.SHORT else -1

    final_preds.append({"example_id": exp_id,
                        "long_answer": {'start_token': long_start,
                                        'end_token': long_end,
                                        'start_byte': -1,
                                        'end_byte': -1},
                        "short_answers": [{'start_token': short_start,
                                           'end_token': short_end,
                                           'start_byte': -1,
                                           'end_byte': -1}],
                        'long_answer_score': long_answer_result[1],
                        'short_answers_score': short_answer_result[1],
                        'answer_type': answer_type,
                        'yes_no_answer': yes_no_answer,
                        'answer_type_logits': list(answer_type_logits)})

output_prediction_file = os.path.join(args.output_dir, 'ensemble_test_predictions.json')
with open(output_prediction_file, 'w') as f:
    json.dump({'predictions': final_preds}, f)

make_submission(output_prediction_file, args.output_dir)
