import torch
import argparse
from albert_modeling import AlBertJointForShort, AlbertConfig
from torch.utils.data import TensorDataset, DataLoader
import utils
from tqdm import tqdm
import os
import json
import albert_tokenization as tokenization
import collections
import pickle
from nq_eval import get_metrics_as_dict
from utils_nq import compute_short_pred, combine_long_short
from albert_short_preprocess import InputShortFeatures, read_nq_examples, convert_examples_to_features

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id",
                                    "short_start_logits",
                                    "short_end_logits",
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
            start_logits, end_logits, answer_type_logits = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = dev_features[example_index.item()]
            unique_id = str(eval_feature.unique_id)

            result = RawResult(unique_id=unique_id,
                               short_start_logits=start_logits[i].cpu().numpy(),
                               short_end_logits=end_logits[i].cpu().numpy(),
                               answer_type_logits=answer_type_logits[i].cpu().numpy())
            all_results.append(result)

    pickle.dump(all_results, open(os.path.join(args.output_dir, 'dev_RawResults.pkl'), 'wb'))
    # all_results = pickle.load(open(os.path.join(args.output_dir, 'RawResults.pkl'), 'rb'))

    nq_pred_dict = compute_short_pred(dev_features, all_results,
                                      args.n_best_size, args.max_answer_length)
    long_short_combined_pred = combine_long_short(nq_pred_dict, args.long_pred_file)

    output_prediction_file = os.path.join(args.output_dir, 'dev_predictions.json')
    with open(output_prediction_file, 'w') as f:
        json.dump({'predictions': long_short_combined_pred}, f)

    results = get_metrics_as_dict(args.predict_file, output_prediction_file)
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
    parser.add_argument("--gpu_ids", default="7", type=str)
    parser.add_argument("--eval_batch_size", default=64, type=int)
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

    parser.add_argument("--bert_config_file", default='albert_xxlarge/albert_config.json', type=str)
    parser.add_argument("--init_restore_dir", default='check_points/albert-xxlarge-short-V01/best_checkpoint.pth',
                        type=str)
    parser.add_argument("--output_dir", default='check_points/albert-xxlarge-short-V01', type=str)
    parser.add_argument("--predict_file", default='data/simplified-nq-dev.jsonl', type=str)
    parser.add_argument("--candidate_file",
                        default='check_points/albert-xxlarge-tfidf-600-top8-V0/long_cand_dict_predictions99998.json',
                        type=str)
    parser.add_argument("--long_pred_file",
                        default='check_points/albert-xxlarge-tfidf-600-top8-V0/predictions99998.json',
                        type=str)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print("device %s n_gpu %d" % (device, n_gpu))
    print("device: {} n_gpu: {} 16-bits training: {}".format(device, n_gpu, args.float16))

    # Loading data
    print('Loading data...')
    tokenizer = tokenization.FullTokenizer(
        vocab_file='albert_xxlarge/30k-clean.vocab', do_lower_case=True,
        spm_model_file='albert_xxlarge/30k-clean.model')

    dev_examples = read_nq_examples(args.predict_file, mode='test', args=args,
                                    test_cand_dict=json.load(open(args.candidate_file)))
    dev_features = convert_examples_to_features(dev_examples, tokenizer, is_training=False, args=args)

    all_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dev_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    eval_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.eval_batch_size)

    bert_config = AlbertConfig.from_json_file(args.bert_config_file)
    model = AlBertJointForShort(bert_config)
    utils.torch_show_all_params(model)
    utils.torch_init_model(model, args.init_restore_dir)
    if args.float16:
        model.half()
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    results = evaluate(model, args, dev_features, device)
