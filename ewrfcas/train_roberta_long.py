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
            start_logits, end_logits = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = dev_features[example_index.item()]
            unique_id = str(eval_feature.unique_id)

            result = RawResult(unique_id=unique_id,
                               long_start_logits=start_logits[i].cpu().numpy(),
                               long_end_logits=end_logits[i].cpu().numpy())
            all_results.append(result)

    pickle.dump(all_results, open(os.path.join(args.output_dir, 'RawResults.pkl'), 'wb'))
    # all_results = pickle.load(open(os.path.join(args.output_dir, 'RawResults.pkl'), 'rb'))

    ground_truth_dict = load_all_annotations_from_dev(args.predict_file)
    nq_pred_dict = compute_long_pred(ground_truth_dict, dev_features, all_results, args.n_best_size)

    output_prediction_file = os.path.join(args.output_dir, 'predictions' + str(global_steps) + '.json')
    with open(output_prediction_file, 'w') as f:
        json.dump({'predictions': list(nq_pred_dict.values())}, f)

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
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions)

    if output_features:
        return dataset, features
    return dataset


def to_list(tensor):
    return tensor.detach().cpu().tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", default="0,1,2,3,4,5,6,7", type=str)
    parser.add_argument("--train_epochs", default=2, type=int)
    parser.add_argument("--train_batch_size", default=48, type=int)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--n_best_size", default=20, type=int)
    parser.add_argument("--max_answer_length", default=30, type=int)
    parser.add_argument("--eval_steps", default=0.25, type=int)
    parser.add_argument('--seed', type=int, default=556)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--warmup_rate', type=float, default=0.1)
    parser.add_argument("--schedule", default='warmup_linear', type=str, help='schedule')
    parser.add_argument("--weight_decay_rate", default=0.01, type=float, help='weight_decay_rate')
    parser.add_argument("--float16", default=True, type=bool)

    parser.add_argument("--bert_config_file", default='roberta_large/config.json', type=str)
    parser.add_argument("--init_restore_dir", default='roberta_large/roberta_large_squad_extend.pth', type=str)
    parser.add_argument("--output_dir", default='check_points/roberta-large-long-V01', type=str)
    parser.add_argument("--log_file", default='log.txt', type=str)
    parser.add_argument("--setting_file", default='setting.txt', type=str)

    parser.add_argument("--predict_file", default='data/simplified-nq-dev.jsonl', type=str)
    # parser.add_argument("--train_feat_dir", default='dataset/train_data_maxlen512_roberta_tfidf_features.bin',
    #                     type=str)
    parser.add_argument("--train_feat_dir", default='dataset/train_data_maxlen512_includeunknowns0.138_roberta_tfidf_features.bin',
                        type=str)
    parser.add_argument("--dev_feat_dir", default='dataset/dev_data_maxlen512_roberta_tfidf_features.bin',
                        type=str)

    args = parser.parse_args()
    args = check_args(args)
    if os.path.exists(args.log_file):
        os.remove(args.log_file)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print("device %s n_gpu %d" % (device, n_gpu))
    print("device: {} n_gpu: {} 16-bits training: {}".format(device, n_gpu, args.float16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Loading data
    print('Loading data...')
    train_dataset = load_cached_data(feature_dir=args.train_feat_dir, output_features=False, evaluate=False)
    dev_dataset, dev_features = load_cached_data(feature_dir=args.dev_feat_dir, output_features=True, evaluate=True)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, drop_last=True)
    eval_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.eval_batch_size)

    steps_per_epoch = len(train_dataset) // args.train_batch_size
    dev_steps_per_epoch = len(dev_features) // args.eval_batch_size
    if len(train_dataset) % args.train_batch_size != 0:
        steps_per_epoch += 1
    if len(dev_dataset) % args.eval_batch_size != 0:
        dev_steps_per_epoch += 1
    total_steps = steps_per_epoch * args.train_epochs

    if args.eval_steps < 1:
        args.eval_steps = int(args.eval_steps * steps_per_epoch)
    print('steps per epoch:', steps_per_epoch)
    print('total steps:', total_steps)
    print('eval steps:', args.eval_steps)
    print('warmup steps:', int(args.warmup_rate * total_steps))

    bert_config = RobertaConfig.from_json_file(args.bert_config_file)
    model = RobertaJointForLong(RobertaModel(bert_config), bert_config)
    utils.torch_show_all_params(model)
    utils.torch_init_model(model, args.init_restore_dir)
    if args.float16:
        model.half()
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # get the optimizer
    optimizer = get_optimization(model=model,
                                 float16=args.float16,
                                 learning_rate=args.lr,
                                 total_steps=total_steps,
                                 schedule=args.schedule,
                                 warmup_rate=args.warmup_rate,
                                 max_grad_norm=args.clip_norm,
                                 weight_decay_rate=args.weight_decay_rate,
                                 opt_pooler=False)  # 只做长答案不需要做answer_type分类

    results = evaluate(model, args, dev_features, device, 0)

    # Train!
    print('***** Training *****')
    global_steps = 1
    best_f1 = 0
    for i in range(int(args.train_epochs)):
        print('Starting epoch %d' % (i + 1))
        total_loss = 0
        iteration = 1
        model.train()
        with tqdm(total=steps_per_epoch, desc='Epoch %d' % (i + 1)) as pbar:
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, \
                start_positions, end_positions = batch
                inputs = {'input_ids': input_ids,
                          'attention_mask': input_mask,
                          'token_type_ids': segment_ids,
                          'start_positions': start_positions,
                          'end_positions': end_positions}
                loss = model(**inputs)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                total_loss += loss.item()
                pbar.set_postfix({'loss': '{0:1.5f}'.format(total_loss / (iteration + 1e-5))})
                pbar.update(1)

                if args.float16:
                    optimizer.backward(loss)
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used and handles this automatically
                    lr_this_step = args.lr * warmup_linear(global_steps / total_steps, args.warmup_rate)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                else:
                    loss.backward()

                optimizer.step()
                model.zero_grad()
                global_steps += 1
                iteration += 1

                if global_steps % args.eval_steps == 0:
                    results = evaluate(model, args, dev_features, device, global_steps)
                    with open(args.log_file, 'a') as aw:
                        aw.write("--------------steps:{}--------------\n".format(global_steps))
                        aw.write(str(json.dumps(results, indent=2)) + '\n')

                    if results['long-f1'] >= best_f1:
                        best_f1 = results['long-f1']
                        print('Best f1:', best_f1)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save(model_to_save.state_dict(),
                                   os.path.join(args.output_dir, 'best_checkpoint.pth'))
