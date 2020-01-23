import torch
import argparse
# from modeling import BertJointForNQ, BertConfig
from albert_modeling import AlbertConfig, AlBertJointForNQ2
from torch.utils.data import TensorDataset, DataLoader
import utils
from tqdm import tqdm
import os
import json
import collections
import pickle
from nq_eval import get_metrics_as_dict
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", default="0,1,2,3,4,5,6,7", type=str)
    parser.add_argument("--train_epochs", default=3, type=int)
    parser.add_argument("--train_batch_size", default=48, type=int)
    parser.add_argument("--eval_batch_size", default=128, type=int)
    parser.add_argument("--n_best_size", default=20, type=int)
    parser.add_argument("--max_answer_length", default=30, type=int)
    parser.add_argument("--eval_steps", default=1300, type=int)
    parser.add_argument('--seed', type=int, default=556)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--warmup_rate', type=float, default=0.06)
    parser.add_argument("--unk_th", default=1.7, type=float, help='unk_thresholds')
    parser.add_argument("--schedule", default='warmup_linear', type=str, help='schedule')
    parser.add_argument("--weight_decay_rate", default=0.01, type=float, help='weight_decay_rate')
    parser.add_argument("--float16", default=True, type=bool)

    parser.add_argument("--predict_file", default='data/simplified-nq-test.jsonl', type=str)
    parser.add_argument("--bert_config_file", default='check_points/albert-xxlarge-tfidf-600-top8-V0', type=str)
    parser.add_argument("--init_restore_dirs", default=['check_points/albert-xxlarge-tfidf-600-top8-V0',
                                                        'check_points/albert-xxlarge-tfidf-600-top8-V01',
                                                        'check_points/albert-xxlarge-tfidf-600-top8-V02',
                                                        'check_points/albert-xxlarge-tfidf-600-top8-V03'], type=list)
    parser.add_argument("--output_dir", default='check_points/albert-xxlarge-V0-ensemble', type=str)
    parser.add_argument("--log_file", default='log.txt', type=str)

    # dev_data_maxlen512_tfidf_features.bin
    parser.add_argument("--dev_feat_dir", default='dataset/test_data_maxlen512_albert_tfidf_ls_features.bin', type=str)

    args = parser.parse_args()
    args.bert_config_file = os.path.join('albert_xxlarge', 'albert_config.json')
    init_dirs = [os.path.join(dir, 'best_checkpoint.pth') for dir in args.init_restore_dirs]
    args.log_file = os.path.join(args.output_dir, args.log_file)
    os.makedirs(args.output_dir, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print("device %s n_gpu %d" % (device, n_gpu))
    print("device: {} n_gpu: {} 16-bits training: {}".format(device, n_gpu, args.float16))

    # # Loading data
    # print('Loading data...')
    # dev_dataset, dev_features = load_cached_data(feature_dir=args.dev_feat_dir, output_features=True, evaluate=True)
    # eval_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.eval_batch_size)
    #
    # bert_config = AlbertConfig.from_json_file(args.bert_config_file)
    #
    # for i, init_dir in enumerate(init_dirs):
    #     print('Load weights from', init_dir)
    #     model = AlBertJointForNQ2(bert_config)
    #     model.load_state_dict(torch.load(init_dir, map_location='cpu'), strict=True)
    #     if args.float16:
    #         model.half()
    #     model.to(device)
    #     if n_gpu > 1:
    #         model = torch.nn.DataParallel(model)
    #
    #     output_prediction_file = os.path.join(args.output_dir, 'test_predictions{}.json'.format(i))
    #     if not os.path.exists(output_prediction_file):
    #         evaluate(model, args, dev_features, device, i)
    #     else:
    #         print(output_prediction_file, 'exists, skip...')

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
