# @Time    : 1/19/2020 1:47 PM
# @Author  : mikelkl
import json
import os

from utils_nq import read_candidates_from_one_split

gold_file = 'data/simplified-nq-dev.jsonl'
model_dir = "check_points/albert-xxlarge-tfidf-600-top8-V0/"
prediction_file = "predictions99998.json"

long_prediction_file = "long_" + prediction_file
long_prediction_file = os.path.join(model_dir, long_prediction_file)

# Create dummy long prediction file
with open(os.path.join(model_dir, prediction_file), "r") as f:
    pred = json.load(f)
long_preds = []
for i in pred["predictions"]:
    example_id = i["example_id"]
    long_answer = i["long_answer"]
    long_answer_score = i["long_answer_score"]
    long_preds.append(dict(example_id=example_id, long_answer=long_answer, long_answer_score=long_answer_score))
with open(long_prediction_file, "w") as f:
    json.dump(long_preds, f)

# map long answer prediction span to its long candidate index
with open(long_prediction_file, "r") as f:
    long_preds = json.load(f)
cand_dict = {}

candidates_dict = read_candidates_from_one_split(gold_file)
for long_pred in long_preds:
    # example_id = str(long_pred["example_id"])
    example_id = long_pred["example_id"]
    start = long_pred["long_answer"]["start_token"]
    end = long_pred["long_answer"]["end_token"]
    cand_dict[example_id] = -1
    for idx, c in enumerate(candidates_dict[example_id]):
        if start == c["start_token"] and end == c["end_token"]:
            cand_dict[example_id] = idx
            break

with open(os.path.join(model_dir, "long_cand_dict_"+ prediction_file), "w") as f:
    json.dump(cand_dict, f)

# fit to short answer model
# example_id = "-1220107454853145579"
example_id = -1014091045618911654
long_answer_cand = [cand_dict[example_id]]
# output: [0]
