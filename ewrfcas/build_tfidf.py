import json
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

data_dir = "dataset/train_splited_600.json"
output_dir = "dataset/train_cand_selected_600.json"
# top1: 51%
# top3: 82.4%
# top5: 91.6%
# top8: 96.7%
# top10: 98%
topk = 8

with open(data_dir, 'r') as f:
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

with open(output_dir, 'w') as w:
    json.dump(tfidf_cand_select, w, indent=2)
