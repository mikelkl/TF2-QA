# @Time    : 11/9/2019 4:14 PM
# @Author  : mikelkl
import os
import tensorflow as tf

from transformers import BertConfig, TFBertForQuestionAnswering, BertTokenizer
from transformers.modeling_tf_utils import get_initializer

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
cache_dir = "input"
# cache_dir = "/users/liukanglong/.cache/torch/transformers"
config_path = os.path.join(cache_dir, "{}-config.json".format(model_name))
vocab_file = "../input/bertjointbaseline/vocab-nq.txt"
model_path = os.path.join(cache_dir, "{}-tf_model.h5".format(model_name))


class TFBertJointForNQ(TFBertForQuestionAnswering):
    r"""
    Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **start_scores**: ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``Numpy array`` or ``tf.Tensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``Numpy array`` or ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import BertTokenizer, TFBertForQuestionAnswering

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = TFBertForQuestionAnswering.from_pretrained('bert-base-uncased')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        start_scores, end_scores = outputs[:2]

    """

    def __init__(self, config, *inputs, **kwargs):
        super(TFBertJointForNQ, self).__init__(config, *inputs, **kwargs)

        self.answer_type_output = tf.keras.layers.Dense(config.num_answer_types,
                                                        kernel_initializer=get_initializer(config.initializer_range),
                                                        name='answer_type_output')

    def call(self, inputs, **kwargs):
        outputs = self.bert(inputs, **kwargs)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        answer_type_logits = self.answer_type_output(pooled_output)

        outputs = (start_logits, end_logits, answer_type_logits) + outputs[2:]

        return outputs  # start_logits, end_logits, answer_type_logits, (hidden_states), (attentions)


# Load tokenizer and model from pretrained model/vocabulary. Specify the number of labels to classify (2+: classification, 1: regression)
config = BertConfig.from_pretrained(config_path)
config.num_answer_types = 5
tokenizer = BertTokenizer.from_pretrained(vocab_file)
model = TFBertJointForNQ.from_pretrained(model_path, config=config)

train_filename = "input/bertjointbaseline/nq-train.tfrecords-00000-of-00001"
raw_dataset = tf.data.TFRecordDataset(train_filename)
eval_features = []
for raw_record in raw_dataset:
    eval_features.append(tf.train.Example.FromString(raw_record.numpy()))
    break
