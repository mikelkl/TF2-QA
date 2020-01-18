# @Time    : 1/18/2020 4:54 PM
# @Author  : mikelkl
# STEP1 将数据集去HTML化，然后按照字数阈值（600）划分存储

from tqdm import tqdm
import json
import os

# STEP2 将分段后的语料选出top8
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


# 停用词
stopwords = set()
with open("stopwords", 'r') as f:
    for line in f:
        stopwords.add(line.strip())

# top1: 51%
# top3: 82.4%
# top5: 91.6%
# top8: 96.7%
# top10: 98%
def select_cands_by_tfidf(question_text, doc_tokens, cands, topk = 8):
    question_text = [qt for qt in question_text.split() if qt not in stopwords]
    question_text = " ".join(question_text)

    cv = CountVectorizer()
    tfidf = TfidfTransformer()
    cands_text = []
    cands_id_top_level_true = []
    for i, cand in enumerate(cands):
        if cand['top_level'] is True:
            cand_tokens = doc_tokens[cand['start_token']:cand['end_token']]
            cand_text = []
            for t in cand_tokens:
                if '<' not in t:
                    cand_text.append(t)
            cand_text = " ".join(cand_text).lower()
            cands_text.append(cand_text)
            cands_id_top_level_true.append(i)
    corpus = [question_text]
    corpus.extend(cands_text)

    words = cv.fit_transform(corpus)
    question_indices = words[0].indices
    tfidf_scores = tfidf.fit_transform(words)
    tfidf_scores = tfidf_scores.toarray()[1:]
    tfidf_scores = np.sum(tfidf_scores[:, question_indices], axis=1)
    tfidf_scores_with_id = list(zip(cands_id_top_level_true, tfidf_scores))

    best_para_ids = sorted(tfidf_scores_with_id, key=lambda x: x[1], reverse=True)
    best_para_ids = best_para_ids[:topk]

    selected_cands_with_id = []
    for best_para_id, _ in best_para_ids:
        selected_cands_with_id.append((best_para_id, cands[best_para_id]))

    return selected_cands_with_id

def split_data(input_dir, output_dir, token_limit=600, is_training=False, topk=16):
    tfidf_cand_select = {}
    total_valid_num = 0
    hit_num = 0
    with open(input_dir, 'r') as f:
        with tqdm(total=1600, desc='Building, Top{}'.format(topk)) as pbar:
            for line in f:
                temp_data = json.loads(line)
                context = temp_data['document_text']
                doc_tokens = context.split()
                cands = temp_data['long_answer_candidates']
                selected_cands_with_id = select_cands_by_tfidf(temp_data['question_text'], doc_tokens, cands, topk=topk)
                split_tokens = []
                for i, cand in selected_cands_with_id:
                    if cand['top_level'] is True:
                        split_tokens.append({'cand_id': i, 'start': cand['start_token'], 'end': cand['end_token'],
                                             'doc_tokens': doc_tokens[cand['start_token']:cand['end_token']]})

                # 去除html元素
                # 构成段落，这里以800词分割，即累积超过800词，重开一段。但是要注意吧cand_id信息也对应存入
                paras = []
                for i in range(len(split_tokens)):
                    split_tokens[i]['doc_tokens'] = [t for t in split_tokens[i]['doc_tokens'] if '<' not in t]
                    paras.append({'cand_id': split_tokens[i]['cand_id'], 'para': " ".join(split_tokens[i]['doc_tokens'])})

                pred_cand_ids = []
                para_ = ""
                cand_ids = []
                for para in paras:
                    cand_ids.append(para['cand_id'])
                    if para_ != "":
                        para_ += " "
                    para_ += para['para']
                    if len(para_.split()) > token_limit:
                        pred_cand_ids.append(cand_ids)
                        para_ = ""
                        cand_ids = []
                if len(cand_ids) > 0:
                    pred_cand_ids.append(cand_ids)

                # 由于答案所在的cand不一定是top_level，我们要把答案所在cand映射到top_level所在的cand里
                if is_training:
                    annotations = temp_data['annotations'][0]  # TODO:先取第一个？
                    gt_cand_id = annotations['long_answer']['candidate_index']
                    if gt_cand_id == -1:
                        true_cand_id = -1
                    else:
                        true_cand_id = -1
                        if cands[gt_cand_id]['top_level'] is False:
                            start = annotations['long_answer']['start_token']
                            end = annotations['long_answer']['end_token']
                            for spt in split_tokens:
                                if spt['start'] <= start and spt['end'] >= end:
                                    true_cand_id = spt['cand_id']
                                    break
                        else:
                            true_cand_id = gt_cand_id

                    assert true_cand_id is not None, true_cand_id
                else:
                    true_cand_id = -1

                cand_set = [i for i, _ in selected_cands_with_id]
                if true_cand_id != -1:  # 统计tfidf截取段落后准确率
                    total_valid_num += 1
                    if true_cand_id in cand_set:
                        hit_num += 1

                pbar.set_postfix({'Acc': '{0:1.5f}'.format(hit_num / (total_valid_num + 1e-5))})
                pbar.update(1)

                tfidf_cand_select[temp_data['example_id']] = pred_cand_ids

    with open(output_dir, 'w') as w:
        json.dump(tfidf_cand_select, w, indent=2)


split_data('../input/tensorflow2-question-answering/simplified-nq-dev.jsonl', 'dev_cand_selected_600.json', token_limit=600, is_training=True)

