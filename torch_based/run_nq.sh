#CUDA_VISIBLE_DEVICES=0,1,2,3,5,7 python run_nq.py \
#    --model_type bert \
#    --model_name_or_path ../input/pretrained_models/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin \
#    --config_name ../input/pretrained_models/bert-large-uncased-whole-word-masking-finetuned-squad-config.json \
#    --tokenizer_name ../input/bertjointbaseline/vocab-nq.txt \
#    --do_train \
#    --do_lower_case \
#    --fp16 \
#    --train_precomputed_file=../input/bertjointbaseline/nq-train.tfrecords-00000-of-00001 \
#    --learning_rate 3e-5 \
#    --num_train_epochs 2 \
#    --max_seq_length 512 \
#    --output_dir ../output/models/bert-large-uncased-whole-word-masking-finetuned-squad/ \
#    --per_gpu_train_batch_size=4 \
#    --save_steps 0.25 --overwrite_output_dir

# predict
CUDA_VISIBLE_DEVICES=7 python -m pdb run_nq.py \
    --model_type bert \
    --model_name_or_path ../output/models/bert-large-uncased-whole-word-masking-finetuned-squad/pytorch_model.bin \
    --config_name ../output/models/bert-large-uncased-whole-word-masking-finetuned-squad/config.json \
    --tokenizer_name ../output/models/bert-large-uncased-whole-word-masking-finetuned-squad/vocab.txt \
    --do_pred \
    --do_lower_case \
    --fp16 \
    --max_seq_length 512 \
    --output_dir ../output/models/bert-large-uncased-whole-word-masking-finetuned-squad/ \
    --per_gpu_eval_batch_size=28 \
    --overwrite_output_dir \
    --predict_file ../input/tensorflow2-question-answering/simplified-nq-test.jsonl \
    --output_prediction_file ../output/models/bert-large-uncased-whole-word-masking-finetuned-squad/predictions_pp.json