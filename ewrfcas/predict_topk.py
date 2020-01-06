import torch
import argparse
from albert_modeling import AlBertJointForNQ2, AlbertConfig
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
import utils
from tqdm import tqdm
import os
import json
import collections
import pickle
import pandas as pd
from albert_preprocess import InputLSFeatures
from utils_nq import read_candidates_from_one_split, compute_pred_dict
from nq_eval import get_metrics_as_dict

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id",
                                    "long_start_topk_logits", "long_start_topk_index",
                                    "long_end_topk_logits", "long_end_topk_index",
                                    "short_start_topk_logits", "short_start_topk_index",
                                    "short_end_topk_logits", "short_end_topk_index",
                                    "answer_type_logits"])


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

        # if entry["short_answers_score"] < 1.5:
        #     return ""

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

        # if entry["long_answer_score"] < 1.5:
        #     return ""

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
    parser.add_argument("--gpu_ids", default="4,5,6,7", type=str)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--n_best_size", default=20, type=int)
    parser.add_argument("--max_answer_length", default=30, type=int)
    parser.add_argument("--float16", default=True, type=bool)
    parser.add_argument("--bert_config_file", default=None, type=str)
    parser.add_argument("--init_restore_dir", default=None, type=str)
    parser.add_argument("--predict_file", default='data/simplified-nq-test.jsonl', type=str)
    parser.add_argument("--output_dir", default='check_points/albert-xxlarge-tfidf-600-top8-V0',
                        type=str)
    parser.add_argument("--predict_feat", default='dataset/test_data_maxlen512_albert_tfidf_ls_features.bin',
                        type=str)
    args = parser.parse_args()
    args.bert_config_file = os.path.join('albert_xxlarge', 'albert_config.json')
    args.init_restore_dir = os.path.join(args.output_dir, 'best_checkpoint.pth')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print("device %s n_gpu %d" % (device, n_gpu))
    print("device: {} n_gpu: {} 16-bits training: {}".format(device, n_gpu, args.float16))

    bert_config = AlbertConfig.from_json_file(args.bert_config_file)
    model = AlBertJointForNQ2(bert_config, long_n_top=5, short_n_top=5)
    utils.torch_show_all_params(model)
    utils.torch_init_model(model, args.init_restore_dir)
    if args.float16:
        model.half()
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    dataset, features = load_cached_data(feature_dir=args.predict_feat, output_features=True, evaluate=True)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    print("***** Running evaluation *****")
    print("  Num examples =", len(dataset))
    print("  Batch size =", args.eval_batch_size)

    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, example_indices = batch
            inputs = {'input_ids': input_ids,
                      'attention_mask': input_mask,
                      'token_type_ids': segment_ids}
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
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
                               answer_type_logits=to_list(outputs['answer_type_logits'][i]))
            all_results.append(result)

    pickle.dump(all_results, open(os.path.join(args.output_dir, 'RawResults_test.pkl'), 'wb'))

    print("Going to candidates file")
    candidates_dict = read_candidates_from_one_split(args.predict_file)

    print("Compute_pred_dict")
    nq_pred_dict = compute_pred_dict(candidates_dict, features,
                                     [r._asdict() for r in all_results],
                                     args.n_best_size, args.max_answer_length, topk_pred=True,
                                     long_n_top=5, short_n_top=5)

    output_prediction_file = os.path.join(args.output_dir, 'predictions_test.json')
    print("Saving predictions to", output_prediction_file)
    with open(output_prediction_file, 'w') as f:
        json.dump({'predictions': list(nq_pred_dict.values())}, f)

    make_submission(output_prediction_file, args.output_dir)
