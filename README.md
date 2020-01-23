# Overview

Big thanks to my awesome teammates [@ewrfcas](https://github.com/ewrfcas) and [@leolemon214](https://github.com/leolemon214). In this repository you will find the code for the [21th place solution, puzzlingly shaking from public LB 3th](https://www.kaggle.com/c/tensorflow2-question-answering/discussion/128140). Our team finished as 3/1233 with a micro F1-score of 0.71 on the public test set and 21/1233 with a micro F1-score of 0.67 on the private test set. The challenge was to predict short and long answer responses to real questions about Wikipedia articles. 

# Get Data
```shell script
# Get train set and public test set
sudo pip install --upgrade kaggle
mkdir .kaggle
# Replace "MYUSER" and "MYKEY" with your credentials. You can create them on:
# `https://www.kaggle.com` -> `My Account` -> `Create New API Token`
echo '{"username":"MYUSER","key":"MYKEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle competitions download -c tensorflow2-question-answering
for f in *.zip; do unzip $f; done
rm *.zip

# Get dev set
wget https://storage.cloud.google.com/natural_questions/v1.0-simplified/nq-dev-all.jsonl.gz
gunzip *.gz


mkdir -p tf2qa/data
mv *.jsonl tf2qa/data
```

# Complete Repo Structure
```shell script
TF2-QA
├── LICENSE
├── README.md
├── input/
├── notebooks/
├── output/
├── refs/
├── tf2qa/                <- pipeline code base
│   ├── data/             <- data folder
│   ├── dataset/          <- preprocessed feature folder
│   ├── checkpoints/      <- saved model folder
│   ├── roberta_large/    <- roberta_large init folder, containing config, vocab, weight ...
│   ├── albert_xxlarge/   <- albert_xxlarge init folder, containing config, vocab, weight ...
└── visulization          <- visualization code base
```
# Usage
```shell script
# get the code
git clone https://github.com/mikelkl/TF2-QA.git

cd TF2-QA/tf2qa

# STEP1 将数据集去HTML化，然后按照字数阈值（600）划分存储
python build_splited_data.py
# STEP2 将分段后的语料选出top8
python build_tfidf.py
# STEP3 roberta_large LS preprocess
python roberta_preprocess.py
# STEP4 roberta_large LS train
python train_roberta_topk.py
# STEP5 albert short preprocess
python albert_short_preprocess.py
# STEP6 albert short train
python train_albert_short.py
# STEP7 ensemble pipeline
python pipline_roberta_albert.py
```

# Solution

## 1. Preprocessing

| No   | Technique                         | Pros                                                         | Cons                              | Effect                                |
| ---- | --------------------------------- | ------------------------------------------------------------ | --------------------------------- | ------------------------------------- |
| 1    | TF\-IDF paragraph selection       | Shorten doc resulting faster inference speed and better accuracy | May loss some context information | - dev f1 +1.8%,<br>- public LB f1 -1% |
| 2    | Sample negative features till 1:1 | Balance pos and neg                                          | Cause longer training time        | dev f1 +2.248%                        |
| 3    | Multi-process preprocessing       | Accelerate preprocessing, especially on training data        | Require multi-core CPU            | xN faster (with N processes)          |

## 2. Modeling

| No   | Model Architecture                                 | Idea                                                         | Performance          |
| ---- | -------------------------------------------------- | ------------------------------------------------------------ | -------------------- |
| 1    | Roberta-Large joint with long/short span extractor | 1. Jointly model:<br>- answer type<br>- long span<br>- short span<br>2. Output topk start/end logits/index | dev f1 63.986%       |
| 2    | Albert-xxlarge joint with short span extractor     | Jointly model:<br/>- answer type<br/>- short span            | def short-f1 69.364% |


All of above model architectures were pretrained on SQuAD dataset by ourselves.

## 3. Trick

| No   | Trick                                                        | Effect                             |
| ---- | ------------------------------------------------------------ | ---------------------------------- |
| 1    | If answer_type is yes/no, output yes/no rather than short span | public LB f1 +6%                   |
| 2    | 1. If answer_type is short, output long span and short span<br>2. If answer_type is long, output long span only<br>3. If answer_type is none, output neither long span nor short span | public LB f1 +8%                   |
| 3    | Choose the best long/short answer pair from topk * topk kind of long/short answer combinations | dev f1 +0.435%                     |
| 4    | `long_score = summary.long_span_score - summary.long_cls_score - summary.answer_type_logits[0]`<br>`short_score = summary.short_span_score - summary.short_cls_score - summary.answer_type_logits[0]` | - dev f1 +2.12%<br>- public LB +2% |
| 5    | Increase long  [CLS] logits multiplier threshold to increase null long answer | dev long-f1 +3.491%                |
| 6    | Decrease short answer_type logits divisor threshold to increase null short answer | dev short-f1 ?                     |

## 4. Ensemble

| No   | Idea                                                         | Effect                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | For long answer, We vote long answers of 2 `Roberta-Large joint with long/short span extractor` models | dev long-f1 +3.341%                                          |
| 2    | For short answer, use step 1 result to locate predicted long answer candidate as input,  We vote short answers of 2 `Roberta-Large joint with long/short span extractor` models and 4 `Albert-xxlarge joint with short span extractor` models | - dev short-f1 +2.842% <br/>- dev f1 67.569%,  +2.635% <br/>- public LB 71%, +5%<br/>- private LB 67% |
