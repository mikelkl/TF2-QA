import torch
import argparse
from roberta_modeling import RobertaJointForNQ2
from transformers.modeling_roberta import RobertaConfig, RobertaModel
from torch.utils.data import TensorDataset, DataLoader
import utils
from glob import glob
from tqdm import tqdm
import os
import json
import collections
import pickle
import pandas as pd
from nq_eval import get_metrics_as_dict
from utils_nq import load_all_annotations_from_dev, compute_long_pred, ls_ensemble_combine
from albert_preprocess import InputLSFeatures

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id",
                                    "long_start_topk_logits", "long_start_topk_index",
                                    "long_end_topk_logits", "long_end_topk_index",
                                    "short_start_topk_logits", "short_start_topk_index",
                                    "short_end_topk_logits", "short_end_topk_index",
                                    "long_cls_logits", "short_cls_logits",
                                    "answer_type_logits"])


def check_args(args):
    args.setting_file = os.path.join(args.output_dir, args.setting_file)
    args.log_file = os.path.join(args.output_dir, args.log_file)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.setting_file, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        print('------------ Options -------------')
        for k in args.__dict__:
            v = args.__dict__[k]
            opt_file.write('%s: %s\n' % (str(k), str(v)))
            print('%s: %s' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')
        print('------------ End -------------')

    return args


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


def evaluate(model, args, dev_features, device, ei):
    # Eval!
    print("***** Running evaluation Ensemble{} *****".format(ei))
    all_results = []
    ensemble_name = args.init_restore_dir[ei].split('/')[-2].split('-')[-1]
    if args.is_test:
        pkl_path = os.path.join(args.output_dir, 'test_long_RawResults_{}.pkl'.format(ensemble_name))
    else:
        pkl_path = os.path.join(args.output_dir, 'dev_long_RawResults_{}.pkl'.format(ensemble_name))
    if not os.path.exists(pkl_path):
        for batch in tqdm(eval_dataloader, desc="Evaluating{}".format(ei)):
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

        pickle.dump(all_results, open(pkl_path, 'wb'))
    else:
        all_results = pickle.load(open(pkl_path, 'rb'))

    ground_truth_dict = load_all_annotations_from_dev(args.predict_file, is_test=args.is_test)
    nq_pred_dict = compute_long_pred(ground_truth_dict, dev_features, all_results, args.n_best_size,
                                     1.0, 1.0, do_ls=True, remain_topk=args.remain_topk)

    if args.is_test:
        output_prediction_file = os.path.join(args.output_dir,
                                              'test_long_predictions_{}.json'.format(ensemble_name))
    else:
        output_prediction_file = os.path.join(args.output_dir,
                                              'dev_long_predictions_{}.json'.format(ensemble_name))
    with open(output_prediction_file, 'w') as f:
        json.dump(nq_pred_dict, f)


def get_ensemble_result(args):
    ensemble_names = [init_restore_dir.split('/')[-2].split('-')[-1] for init_restore_dir in args.init_restore_dir]
    if args.is_test:
        all_preds = [os.path.join(args.output_dir, 'test_long_predictions_' + e + '.json') for e in ensemble_names]
    else:
        all_preds = [os.path.join(args.output_dir, 'dev_long_predictions_' + e + '.json') for e in ensemble_names]

    output_prediction_file = None
    for th1 in args.thresholds1:
        for th2 in args.thresholds2:
            print('UNK threshold:', th1, 'SHORT threshold', th2)
            ensemble_ls_pred_dict = ls_ensemble_combine(all_preds, th1, th2)

            if args.is_test:
                output_prediction_file = os.path.join(args.output_dir, 'all_test_ls_predictions.json')
            else:
                output_prediction_file = os.path.join(args.output_dir, 'all_dev_ls_predictions.json')
            with open(output_prediction_file, 'w') as f:
                json.dump({'predictions': list(ensemble_ls_pred_dict.values())}, f)

            if not args.is_test:
                results = get_metrics_as_dict(args.predict_file, output_prediction_file)
                print(json.dumps(results, indent=2))

        # if args.is_test:
        #     make_submission(output_prediction_file, args.output_dir)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", default="0,1,2,3,4,5,6,7", type=str)
    parser.add_argument("--remain_topk", default=1, type=int)
    parser.add_argument("--eval_batch_size", default=256, type=int)
    parser.add_argument("--n_best_size", default=20, type=int)
    parser.add_argument("--max_answer_length", default=30, type=int)
    parser.add_argument("--float16", default=True, type=bool)
    parser.add_argument("--thresholds1", default=[1.35], type=list)
    parser.add_argument("--thresholds2", default=[1.0], type=list)

    parser.add_argument("--bert_config_file", default='roberta_large/config.json', type=str)
    parser.add_argument("--init_restore_dir",
                        default=['check_points/roberta-large-tfidf-600-top8-V1_retrain/best_checkpoint.pth',
                                 # 'check_points/roberta-large-tfidf-600-top8-V11/best_checkpoint.pth'],
                                 'check_points/roberta-large-tfidf-600-top8-V12/best_checkpoint.pth'],
                        # 'check_points/roberta-large-tfidf-600-top8-V1_retrain/best_checkpoint.pth'
                        type=list)
    parser.add_argument("--output_dir", default='check_points/roberta-large-ls-ensemble', type=str)
    parser.add_argument("--log_file", default='log.txt', type=str)
    parser.add_argument("--setting_file", default='setting.txt', type=str)

    parser.add_argument("--predict_file", default='data/simplified-nq-test.jsonl', type=str)
    parser.add_argument("--dev_feat_dir", default='dataset/test_data_maxlen512_roberta_tfidf_ls_features.bin', type=str)
    # parser.add_argument("--dev_feat_dir", default='dataset/test_data_maxlen512_roberta_tfidf_features.bin', type=str)
    parser.add_argument("--is_test", default=True, type=bool)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print("device %s n_gpu %d" % (device, n_gpu))
    print("device: {} n_gpu: {} 16-bits training: {}".format(device, n_gpu, args.float16))

    # Loading data
    print('Loading data...')
    dev_dataset, dev_features = load_cached_data(feature_dir=args.dev_feat_dir, output_features=True, evaluate=True)
    eval_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.eval_batch_size)

    dev_steps_per_epoch = len(dev_features) // args.eval_batch_size
    if len(dev_dataset) % args.eval_batch_size != 0:
        dev_steps_per_epoch += 1

    for i, init_restore_dir in enumerate(args.init_restore_dir):
        bert_config = RobertaConfig.from_json_file(args.bert_config_file)
        model = RobertaJointForNQ2(RobertaModel(bert_config), bert_config)
        utils.torch_show_all_params(model)
        utils.torch_init_model(model, init_restore_dir)
        if args.float16:
            model.half()
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        evaluate(model, args, dev_features, device, i)

    get_ensemble_result(args)
