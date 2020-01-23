# 将数据集去HTML化，然后按照字数阈值（800）划分存储
from tqdm import tqdm
import json

def split_data(input_dir, output_dir, token_limit=600, is_training=False):
    para_splited_data = []
    with open(input_dir,'r') as f:
        for line in tqdm(f):
            temp_data = json.loads(line)
            context = temp_data['document_text']
            doc_tokens = context.split()
            cands = temp_data['long_answer_candidates']
            split_tokens = []
            for i, cand in enumerate(cands):
                if cand['top_level'] is True:
                    split_tokens.append({'cand_id':i, 'start':cand['start_token'], 'end':cand['end_token'],
                                         'doc_tokens':doc_tokens[cand['start_token']:cand['end_token']]})

            # 去除html元素
            # 构成段落，这里以800词分割，即累积超过800词，重开一段。但是要注意吧cand_id信息也对应存入
            paras = []
            for i in range(len(split_tokens)):
                split_tokens[i]['doc_tokens'] = [t for t in split_tokens[i]['doc_tokens'] if '<' not in t]
                paras.append({'cand_id':split_tokens[i]['cand_id'], 'para':" ".join(split_tokens[i]['doc_tokens'])})

            split_paras = []
            new_para = {'para':"", 'cand_ids':[]}
            for para in paras:
                new_para['cand_ids'].append(para['cand_id'])
                if new_para['para'] != "":
                    new_para['para'] += " "
                new_para['para'] += para['para']
                if len(new_para['para'].split())>token_limit:
                    split_paras.append(new_para)
                    new_para = {'para':"", 'cand_ids':[]}
            if len(new_para['cand_ids']) > 0:
                split_paras.append(new_para)

            # 由于答案所在的cand不一定是top_level，我们要把答案所在cand映射到top_level所在的cand里
            if is_training:
                annotations = temp_data['annotations'][0] # TODO:先取第一个？
                gt_cand_id = annotations['long_answer']['candidate_index']
                if gt_cand_id == -1:
                    true_cand_id = -1
                else:
                    true_cand_id = None
                    if cands[gt_cand_id]['top_level'] is False:
                        start = annotations['long_answer']['start_token']
                        end = annotations['long_answer']['end_token']
                        for spt in split_tokens:
                            if spt['start']<=start and spt['end']>=end:
                                true_cand_id = spt['cand_id']
                                break
                    else:
                        true_cand_id = gt_cand_id

                assert true_cand_id is not None
            else:
                true_cand_id = -1

            para_splited_data.append({
                'example_id':temp_data['example_id'],
                'question_text':temp_data['question_text'],
                'true_cand_id':true_cand_id,
                'split_paras':split_paras
            })

    with open(output_dir, 'w') as w:
        json.dump(para_splited_data, w, indent=2)
        
split_data('data/simplified-nq-train.jsonl', 'dataset/train_splited_600.json', token_limit=600, is_training=True)
# split_data('data/simplified-nq-dev.jsonl', 'dataset/dev_splited_600.json', token_limit=600, is_training=False)
# split_data('data/simplified-nq-test.jsonl', 'dataset/test_splited_600.json', token_limit=600, is_training=False)