import torch
import argparse
from roberta_modeling import RobertaJointForLong
from transformers.modeling_roberta import RobertaConfig, RobertaModel
from torch.utils.data import TensorDataset, DataLoader
import utils
from tqdm import tqdm
import os
import random
import numpy as np
import json
import collections
import pickle
from nq_eval import get_metrics_as_dict
from utils_nq import load_all_annotations_from_dev, compute_long_pred
from roberta_long_preprocess import InputLongFeatures
from pytorch_optimization import get_optimization, warmup_linear

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id",
                                    "long_start_logits",
                                    "long_end_logits"])


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


def evaluate(model, args, dev_features, device):
    # Eval!
    print("***** Running evaluation *****")
    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, example_indices = batch
            inputs = {'input_ids': input_ids,
                      'attention_mask': input_mask,
                      'token_type_ids': segment_ids}
            start_logits, end_logits = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = dev_features[example_index.item()]
            unique_id = str(eval_feature.unique_id)

            result = RawResult(unique_id=unique_id,
                               long_start_logits=start_logits[i].cpu().numpy(),
                               long_end_logits=end_logits[i].cpu().numpy())
            all_results.append(result)

    if args.is_test:
        pickle.dump(all_results, open(os.path.join(args.output_dir, 'test_long_RawResults.pkl'), 'wb'))
    else:
        pickle.dump(all_results, open(os.path.join(args.output_dir, 'dev_long_RawResults.pkl'), 'wb'))
    # all_results = pickle.load(open(os.path.join(args.output_dir, 'dev_long_RawResults.pkl'), 'rb'))

    for th in args.thresholds:
        print('UNK type threshold:', th)
        ground_truth_dict = load_all_annotations_from_dev(args.predict_file, is_test=args.is_test)
        nq_pred_dict = compute_long_pred(ground_truth_dict, dev_features, all_results, args.n_best_size, th)

        if args.is_test:
            output_prediction_file = os.path.join(args.output_dir,
                                                  'test_long_predictions.json')
        else:
            output_prediction_file = os.path.join(args.output_dir,
                                                  'dev_long_predictions.json')
        with open(output_prediction_file, 'w') as f:
            json.dump({'predictions': list(nq_pred_dict.values())}, f)

        if not args.is_test:
            results = get_metrics_as_dict(args.predict_file, output_prediction_file)
            print(json.dumps(results, indent=2))


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
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions)

    if output_features:
        return dataset, features
    return dataset


def to_list(tensor):
    return tensor.detach().cpu().tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", default="6,7", type=str)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--n_best_size", default=20, type=int)
    parser.add_argument("--max_answer_length", default=30, type=int)
    parser.add_argument("--float16", default=True, type=bool)
    parser.add_argument("--thresholds", default=[1.3], type=list)

    parser.add_argument("--bert_config_file", default='roberta_large/config.json', type=str)
    parser.add_argument("--init_restore_dir", default='check_points/roberta-large-long-V00/best_checkpoint.pth',
                        type=str)
    parser.add_argument("--output_dir", default='check_points/roberta-large-long-V00', type=str)
    parser.add_argument("--log_file", default='log.txt', type=str)
    parser.add_argument("--setting_file", default='setting.txt', type=str)

    parser.add_argument("--predict_file", default='data/simplified-nq-dev.jsonl', type=str)
    parser.add_argument("--dev_feat_dir", default='dataset/dev_data_maxlen512_roberta_tfidf_features.bin', type=str)
    # parser.add_argument("--dev_feat_dir", default='dataset/test_data_maxlen512_roberta_tfidf_features.bin', type=str)
    parser.add_argument("--is_test", default=False, type=bool)

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

    bert_config = RobertaConfig.from_json_file(args.bert_config_file)
    model = RobertaJointForLong(RobertaModel(bert_config), bert_config)
    utils.torch_show_all_params(model)
    utils.torch_init_model(model, args.init_restore_dir)
    if args.float16:
        model.half()
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    evaluate(model, args, dev_features, device)
