import torch
import argparse
from modeling import BertJointForNQ, BertConfig
from torch.utils.data import TensorDataset, DataLoader
import utils
from tqdm import tqdm
import os
import json
import collections
import pickle
from nq_eval import get_metrics_as_dict
from utils_nq import read_candidates_from_one_split, compute_pred_dict, InputFeatures

RawResult = collections.namedtuple(
    "RawResult",
    ["unique_id", "start_logits", "end_logits", "answer_type_logits"])


def evaluate(model, args, dev_features, device, global_steps):
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
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = dev_features[example_index.item()]
            unique_id = str(eval_feature.unique_id)
            result = RawResult(unique_id=unique_id,
                               start_logits=to_list(outputs[0][i]),
                               end_logits=to_list(outputs[1][i]),
                               answer_type_logits=to_list(outputs[2][i]))
            all_results.append(result)

    pickle.dump(all_results, open(os.path.join(args.output_dir, 'RawResults.pkl'), 'wb'))
    # all_results = pickle.load(open(os.path.join(args.output_dir, 'RawResults.pkl'), 'rb'))

    # print("Going to candidates file")
    candidates_dict = read_candidates_from_one_split(args.predict_file)

    # print("Compute_pred_dict")
    nq_pred_dict = compute_pred_dict(candidates_dict, dev_features,
                                     [r._asdict() for r in all_results],
                                     args.n_best_size, args.max_answer_length)

    output_prediction_file = os.path.join(args.output_dir, 'predictions' + str(global_steps) + '.json')
    # print("Saving predictions to", output_prediction_file)
    with open(output_prediction_file, 'w') as f:
        json.dump({'predictions': list(nq_pred_dict.values())}, f)

    # print("Computing f1 score")
    results = get_metrics_as_dict(args.predict_file, output_prediction_file)
    print('Steps:{}'.format(global_steps))
    print(json.dumps(results, indent=2))

    model.train()

    return results


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
        all_answer_types = torch.tensor([f.answer_type for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions, all_answer_types)

    if output_features:
        return dataset, features
    return dataset


def to_list(tensor):
    return tensor.detach().cpu().tolist()


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
    parser.add_argument("--schedule", default='warmup_linear', type=str, help='schedule')
    parser.add_argument("--weight_decay_rate", default=0.01, type=float, help='weight_decay_rate')
    parser.add_argument("--float16", default=True, type=bool)

    parser.add_argument("--bert_config_file", default='../output/models/bert-large-uncased-whole-word-masking-finetuned-squad/config.json', type=str)
    parser.add_argument("--init_restore_dir", default='../output/models/bert-large-uncased-whole-word-masking-finetuned-squad/pytorch_model.bin', type=str)
    parser.add_argument("--output_dir", default='../output/models/bert-large-uncased-whole-word-masking-finetuned-squad/', type=str)
    parser.add_argument("--log_file", default='log.txt', type=str)

    parser.add_argument("--predict_file", default='../input/tensorflow2-question-answering/simplified-nq-dev.jsonl', type=str)
    # parser.add_argument("--train_feat_dir", default='dataset/train_data_maxlen512_tfidf_features.bin', type=str)
    parser.add_argument("--dev_feat_dir", default='../input/tensorflow2-question-answering/dev_data_maxlen512_ranking_top10_features.bin', type=str)

    args = parser.parse_args()
    args.log_file = os.path.join(args.output_dir, args.log_file)
    os.makedirs(args.output_dir, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print("device %s n_gpu %d" % (device, n_gpu))
    print("device: {} n_gpu: {} 16-bits training: {}".format(device, n_gpu, args.float16))

    # Loading data
    print('Loading data...')
    dev_dataset, dev_features = load_cached_data(feature_dir=args.dev_feat_dir, output_features=True, evaluate=True)
    eval_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.eval_batch_size)

    bert_config = BertConfig.from_json_file(args.bert_config_file)
    model = BertJointForNQ(bert_config)
    utils.torch_show_all_params(model)
    utils.torch_init_model(model, args.init_restore_dir)
    if args.float16:
        model.half()
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    results = evaluate(model, args, dev_features, device, 0)
