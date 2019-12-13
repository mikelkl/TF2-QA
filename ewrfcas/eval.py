import torch
import argparse
from modeling import BertJointForNQ, BertConfig
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
import utils
from tqdm import tqdm
import os
import json
import collections
import pickle
import utils_nq as utils_nq
from nq_eval import get_metrics_as_dict
from utils_nq import read_candidates_from_one_split, compute_pred_dict


def load_and_cache_examples(cached_features_example_file, output_examples=False, evaluate=False):
    features, examples = torch.load(cached_features_example_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions)

    if output_examples:
        return dataset, examples, features
    return dataset


def to_list(tensor):
    return tensor.detach().cpu().tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", default="4,5,6,7", type=str)
    parser.add_argument("--eval_batch_size", default=128, type=int)
    parser.add_argument("--n_best_size", default=20, type=int)
    parser.add_argument("--max_answer_length", default=30, type=int)
    parser.add_argument("--float16", default=True, type=bool)
    parser.add_argument("--bert_config_file", default=None, type=str)
    parser.add_argument("--init_restore_dir", default=None, type=str)
    parser.add_argument("--predict_file", default='data/simplified-nq-dev.jsonl', type=str)
    parser.add_argument("--output_dir", default='check_points/bert-large-wwm-finetuned-squad/checkpoint-41224',
                        type=str)
    parser.add_argument("--cached_file", default='dataset/cached_dev_pytorch_model.bin_512',
                        type=str)
    args = parser.parse_args()
    args.bert_config_file = os.path.join(args.output_dir, 'config.json')
    args.init_restore_dir = os.path.join(args.output_dir, 'pytorch_model.bin')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print("device %s n_gpu %d" % (device, n_gpu))
    print("device: {} n_gpu: {} 16-bits training: {}".format(device, n_gpu, args.float16))

    bert_config = BertConfig.from_json_file(args.bert_config_file)
    model = BertJointForNQ(bert_config)
    utils.torch_show_all_params(model)
    utils.torch_init_model(model, args.init_restore_dir)
    if args.float16:
        model.half()
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    dataset, examples, features = load_and_cache_examples(cached_features_example_file=args.cached_file,
                                                          output_examples=True, evaluate=True)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    print("***** Running evaluation *****")
    print("  Num examples =", len(dataset))
    print("  Batch size =", args.eval_batch_size)
    RawResult = collections.namedtuple(
        "RawResult",
        ["unique_id", "start_logits", "end_logits", "answer_type_logits"])

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
            unique_id = int(eval_feature.unique_id)
            result = RawResult(unique_id=unique_id,
                               start_logits=to_list(outputs[0][i]),
                               end_logits=to_list(outputs[1][i]),
                               answer_type_logits=to_list(outputs[2][i]))
            all_results.append(result)

    pickle.dump(all_results, open(os.path.join(args.output_dir, 'RawResults.pkl'), 'wb'))

    # print("Going to candidates file")
    # candidates_dict = read_candidates_from_one_split(args.predict_file)
    #
    # print("Compute_pred_dict")
    # nq_pred_dict = compute_pred_dict(candidates_dict, features,
    #                                  [r._asdict() for r in all_results],
    #                                  args.n_best_size, args.max_answer_length)
    #
    # output_prediction_file = os.path.join(args.output_dir, 'predictions.json')
    # print("Saving predictions to", output_prediction_file)
    # with open(output_prediction_file, 'w') as f:
    #     json.dump({'predictions': list(nq_pred_dict.values())}, f)
    #
    # print("Computing f1 score")
    # results = get_metrics_as_dict(args.predict_file, output_prediction_file)
