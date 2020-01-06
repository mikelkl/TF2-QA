import json
import argparse
import requests
from tqdm import tqdm


# ranking acc (hit rate):
# # top1: 64.5%
# # top3: 88.6%
# # top5: 94.5%
# # top8: 97.7%
# # top10: 98.6%
def select_cand_by_ranking(data_dir, output_dir, k_cands=[1, 3, 5, 8, 10],
                           url="http://20.7.0.9:8095/gammalab9crawlingrankingenglishjhd"):
    """
    Select paragraph candidates by ranking.
    :param data_dir: str
    :param output_dir: str
    :param k_cands: int or list
            Save top k best paragraph candidates
    :param url: str
            Ranking interface url
    """
    assert type(k_cands) in [list, int], "k_cands must be list or int"
    if type(k_cands) != list:
        k_cands = [k_cands]

    with open(data_dir, 'r') as f:
        para_splited_data = json.load(f)
    total_valid_num = 0

    hit_num = {}
    # format like dict(k:int=dict(example_id:str=cand_ids:list[int]))
    tfidf_cand_select = {}
    # init global variables
    for k in k_cands:
        hit_num[k] = 0
        tfidf_cand_select[k] = {}

    with tqdm(total=len(para_splited_data), desc='Building with ranking', position=0, leave=True) as pbar:
        for para_sample in para_splited_data:
            # # rank paras by question
            question_text = para_sample['question_text'].lower()
            paras = para_sample['split_paras']
            gt = para_sample['true_cand_id']
            paras_text = [p['para'].lower() for p in paras]
            data = dict(contexts=paras_text, question=question_text)
            res = requests.post(url, json=data).json()
            res = [float(i) for i in res["result"]]
            paras_with_score = zip(paras, res)
            topk = max(k_cands)
            paras_with_score_topk = [x for x, _ in sorted(paras_with_score, key=lambda t: t[1], reverse=True)[:topk]]

            pred_cand_ids = {}
            cand_set = {}
            for k in k_cands:
                # init temp variables
                pred_cand_ids[k] = []
                cand_set[k] = set()

                # archive top k paras
                for p in paras_with_score_topk[:k]:
                    pred_cand_ids[k].append(p['cand_ids'])
                    for ci in pred_cand_ids[k][-1]:
                        cand_set[k].add(ci)

            # 统计截取段落后准确率
            if gt != -1:
                total_valid_num += 1
                for k in k_cands:
                    if gt in cand_set[k]:
                        hit_num[k] += 1

            acc_dict = {}
            for k in k_cands:
                acc_dict["Acc top{}".format(k)] = '{0:1.5f}'.format(hit_num[k] / (total_valid_num + 1e-5))
                # archive selected candidates
                tfidf_cand_select[k][para_sample['example_id']] = pred_cand_ids[k]
            pbar.set_postfix(acc_dict)
            pbar.update(1)

    with open(output_dir, 'w') as w:
        # format like dict(k:int=dict(example_id:str=cand_ids:list[int]))
        json.dump(tfidf_cand_select, w, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../input/tensorflow2-question-answering/train_splited_600.json",
                        type=str)
    parser.add_argument("--output_dir",
                        default="../input/tensorflow2-question-answering/train_cand_selected_600_ranking.json",
                        type=str)
    args = parser.parse_args()
    select_cand_by_ranking(args.data_dir, args.output_dir)
