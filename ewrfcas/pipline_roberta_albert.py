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


para_splited_data = []

split_data('data/simplified-nq-test.jsonl', 'dataset/test_splited_600.json', token_limit=600, is_training=False)

# STEP2 将分段后的语料选出top8
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

# top1: 51%
# top3: 82.4%
# top5: 91.6%
# top8: 96.7%
# top10: 98%
topk = 8

with open('dataset/test_splited_600.json', 'r') as f:
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

tfidf_cand_select = {}
para_splited_data = []

# STEP3 roberta_large LS preprocess
import argparse
import os
import torch
from roberta_preprocess import read_nq_examples as real_roberta_examples
from roberta_preprocess import convert_examples_to_features as convert_roberta_features
from transformers import tokenization_roberta

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
parser.add_argument("--include_unknowns", type=float, default=0.138,
                    help="If positive, probability of including answers of type `UNKNOWN`.")
parser.add_argument("--skip_nested_contexts", type=bool, default=True,
                    help="Completely ignore context that are not top level nodes in the page.")
parser.add_argument("--do_ls", type=bool, default=True,
                    help="Long short answers.")
parser.add_argument("--tfidf_test_file", type=str, default='dataset/test_cand_selected_600.json')

args = parser.parse_args()
tokenizer = tokenization_roberta.RobertaTokenizer(vocab_file='roberta_large/vocab.json',
                                                  merges_file='roberta_large/merges.txt')

example_output_file = os.path.join(args.output_dir,
                                   'test_data_maxlen{}_tfidf_examples.json'.format(args.max_seq_length))
feature_output_file = os.path.join(args.output_dir,
                                   'test_data_maxlen{}_roberta_tfidf_ls_features.bin'.format(args.max_seq_length))
if not os.path.exists(feature_output_file):
    tfidf_dict = json.load(open(args.tfidf_test_file))
    if os.path.exists(example_output_file):
        examples = json.load(open(example_output_file))
    else:
        examples = real_roberta_examples(input_file=args.test_file, tfidf_dict=tfidf_dict, is_training=False, args=args)
        with open(example_output_file, 'w') as w:
            json.dump(examples, w)
    features = convert_roberta_features(examples=examples, tokenizer=tokenizer, is_training=False, args=args)
    torch.save(features, feature_output_file)

examples = []
features = []

# STEP 4 预测长答案
from roberta_modeling import RobertaJointForNQ2
from transformers.modeling_roberta import RobertaConfig, RobertaModel
from torch.utils.data import TensorDataset, DataLoader
import utils
from roberta_ls_inference import load_cached_data as load_roberta_data
from roberta_ls_inference import evaluate as roberta_evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_ids", default="0", type=str)
parser.add_argument("--eval_batch_size", default=16, type=int)
parser.add_argument("--n_best_size", default=20, type=int)
parser.add_argument("--max_answer_length", default=30, type=int)
parser.add_argument("--float16", default=False, type=bool)
parser.add_argument("--thresholds1", default=[1.5], type=list)
parser.add_argument("--thresholds2", default=[1.0], type=list)

parser.add_argument("--bert_config_file", default='roberta_large/config.json', type=str)
parser.add_argument("--init_restore_dir", default='check_points/roberta-large-tfidf-600-top8-V1/best_checkpoint.pth',
                    type=str)
parser.add_argument("--output_dir", default='check_points/roberta-large-tfidf-600-top8-V1', type=str)
parser.add_argument("--predict_file", default='data/simplified-nq-test.jsonl', type=str)
parser.add_argument("--dev_feat_dir", default='dataset/test_data_maxlen512_roberta_tfidf_ls_features.bin', type=str)
parser.add_argument("--is_test", default=True, type=bool)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
device = torch.device("cuda")
n_gpu = torch.cuda.device_count()
print("device %s n_gpu %d" % (device, n_gpu))
print("device: {} n_gpu: {} 16-bits training: {}".format(device, n_gpu, args.float16))

# Loading data
print('Loading data...')
dev_dataset, dev_features = load_roberta_data(feature_dir=args.dev_feat_dir, output_features=True, evaluate=True)
eval_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.eval_batch_size)

dev_steps_per_epoch = len(dev_features) // args.eval_batch_size
if len(dev_dataset) % args.eval_batch_size != 0:
    dev_steps_per_epoch += 1

bert_config = RobertaConfig.from_json_file(args.bert_config_file)
model = RobertaJointForNQ2(RobertaModel(bert_config), bert_config)
utils.torch_show_all_params(model)
utils.torch_init_model(model, args.init_restore_dir)
if args.float16:
    model.half()
model.to(device)
if n_gpu > 1:
    model = torch.nn.DataParallel(model)

roberta_evaluate(model, args, dev_features, device)

# STEP 5 预测短答案，并合并
from albert_modeling import AlBertJointForShort, AlbertConfig
from albert_short_preprocess import read_nq_examples as read_albert_examples
from albert_short_preprocess import convert_examples_to_features as convert_albert_features
from utils_nq import read_candidates_from_one_split
from albert_short_inference import evaluate as albert_evaluate
from albert_short_inference import get_ensemble_result
import albert_tokenization

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_ids", default="0", type=str)
parser.add_argument("--eval_batch_size", default=16, type=int)
parser.add_argument("--n_best_size", default=20, type=int)
parser.add_argument("--max_position", default=50, type=int)
parser.add_argument("--max_seq_length", default=512, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                         "longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--max_query_length", default=64, type=int,
                    help="The maximum number of tokens for the question. Questions longer than this will "
                         "be truncated to this length.")
parser.add_argument("--skip_nested_contexts", type=bool, default=True,
                    help="Completely ignore context that are not top level nodes in the page.")
parser.add_argument("--doc_stride", default=128, type=int,
                    help="When splitting up a long document into chunks, how much stride to take between chunks.")
parser.add_argument("--max_answer_length", default=30, type=int)
parser.add_argument("--float16", default=True, type=bool)
parser.add_argument("--thresholds", default=[2.0], type=list)
parser.add_argument("--yesno_thresholds", default=[0], type=list, help='This th is added to the logits')

parser.add_argument("--bert_config_file", default='albert_xxlarge/albert_config.json', type=str)
parser.add_argument("--init_restore_dir", default=['check_points/albert-xxlarge-short-V00/best_checkpoint.pth',
                                                   'check_points/albert-xxlarge-short-V03/best_checkpoint.pth'],
                    type=list)
parser.add_argument("--output_dir", default='check_points/albert-xxlarge-short-ensemble', type=str)
parser.add_argument("--predict_file", default='data/simplified-nq-test.jsonl', type=str)
parser.add_argument("--long_pred_file",
                    default='check_points/roberta-large-tfidf-600-top8-V1/test_long_predictions.json',
                    type=str)
parser.add_argument("--is_test", default=True, type=bool)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
device = torch.device("cuda")
n_gpu = torch.cuda.device_count()
print("device %s n_gpu %d" % (device, n_gpu))
print("device: {} n_gpu: {} 16-bits training: {}".format(device, n_gpu, args.float16))

# Loading data
print('Loading data...')
tokenizer = albert_tokenization.FullTokenizer(
    vocab_file='albert_xxlarge/30k-clean.vocab', do_lower_case=True,
    spm_model_file='albert_xxlarge/30k-clean.model')

# map long answer prediction span to its long candidate index
with open(args.long_pred_file, "r") as f:
    long_preds = json.load(f)['predictions']
cand_dict = {}
candidates_dict = read_candidates_from_one_split(args.predict_file)
for long_pred in long_preds:
    example_id = long_pred["example_id"]
    start = long_pred["long_answer"]["start_token"]
    end = long_pred["long_answer"]["end_token"]
    cand_dict[example_id] = -1

    dtype = type(list(candidates_dict.keys())[0])
    if dtype == str:
        example_id = str(example_id)
    else:
        example_id = int(example_id)
    for idx, c in enumerate(candidates_dict[example_id]):
        if start == c["start_token"] and end == c["end_token"]:
            cand_dict[example_id] = idx
            break

dev_examples = read_albert_examples(args.predict_file, mode='test', args=args, test_cand_dict=cand_dict)
dev_features = convert_albert_features(dev_examples, tokenizer, is_training=False, args=args)

all_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
dev_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
eval_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.eval_batch_size)

for i, init_restore_dir in enumerate(args.init_restore_dir):
    bert_config = AlbertConfig.from_json_file(args.bert_config_file)
    model = AlBertJointForShort(bert_config)
    utils.torch_show_all_params(model)
    utils.torch_init_model(model, init_restore_dir)
    if args.float16:
        model.half()
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    albert_evaluate(model, args, dev_features, device, i)

get_ensemble_result(args)
