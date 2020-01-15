# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""
from __future__ import print_function

import copy
import json
import math
import logging
import six

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

logger = logging.getLogger(__name__)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def fast_gelu(x):
    return x * torch.sigmoid(1.702 * x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": fast_gelu, "relu": torch.relu, "swish": swish}


class AlbertConfig(object):
    """Configuration for `AlbertModel`.
    The default settings match the configuration of model `albert_xxlarge`.
    """

    def __init__(self,
                 vocab_size,
                 embedding_size=128,
                 hidden_size=4096,
                 num_hidden_layers=12,
                 num_hidden_groups=1,
                 num_attention_heads=64,
                 intermediate_size=16384,
                 inner_group_num=1,
                 down_scale_factor=1,
                 hidden_act="gelu",
                 hidden_dropout_prob=0,
                 attention_probs_dropout_prob=0,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs AlbertConfig.
        Args:
          vocab_size: Vocabulary size of `inputs_ids` in `AlbertModel`.
          embedding_size: size of voc embeddings.
          hidden_size: Size of the encoder layers and the pooler layer.
          num_hidden_layers: Number of hidden layers in the Transformer encoder.
          num_hidden_groups: Number of group for the hidden layers, parameters in
            the same group are shared.
          num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
          intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
          inner_group_num: int, number of inner repetition of attention and ffn.
          down_scale_factor: float, the scale to apply
          hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler.
          hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
          attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
          max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
          type_vocab_size: The vocabulary size of the `token_type_ids` passed into
            `AlbertModel`.
          initializer_range: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_groups = num_hidden_groups
        self.num_attention_heads = num_attention_heads
        self.inner_group_num = inner_group_num
        self.down_scale_factor = down_scale_factor
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `AlbertConfig` from a Python dictionary of parameters."""
        config = AlbertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `AlbertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")


    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-5):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class AlbertEmbeddings(nn.Module):
    """ Albert embeddings. """

    def __init__(self, config):
        super(AlbertEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.embedding_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor

        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # if isinstance(config.hidden_act, str) else config.hidden_act
        self.intermediate_act_fn = ACT2FN[config.hidden_act]
        self.output = BertOutput(config)

    def forward(self, input_tensor):
        hidden_states = self.dense(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_output = self.output(hidden_states, input_tensor)
        return hidden_output


class BertFF(nn.Module):
    def __init__(self, config):
        super(BertFF, self).__init__()
        self.intermediate = BertIntermediate(config)

    def forward(self, hidden_states):
        hidden_states = self.intermediate(hidden_states)
        return hidden_states


class AlbertLayer(nn.Module):
    def __init__(self, config):
        super(AlbertLayer, self).__init__()
        self.attention_1 = BertAttention(config)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.ffn_1 = BertFF(config)
        self.LayerNorm_1 = BertLayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention_1(hidden_states, attention_mask)
        attention_output = self.LayerNorm(attention_output)
        attention_output = self.ffn_1(attention_output)
        attention_output = self.LayerNorm_1(attention_output)
        return attention_output


class AlbertEncoder(nn.Module):
    def __init__(self, config):
        super(AlbertEncoder, self).__init__()
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.num_hidden_layers = config.num_hidden_layers
        self.transformer = AlbertLayer(config)

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        all_encoder_layers = []
        for i in range(self.num_hidden_layers):
            hidden_states = self.transformer(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class AlbertModel(nn.Module):
    def __init__(self, config):
        super(AlbertModel, self).__init__()
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertEncoder(config)
        self.pooler = BertPooler(config)
        self.config = config
        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class AlbertForPreTraining(nn.Module):
    def __init__(self, config):
        super(AlbertForPreTraining, self).__init__()
        self.bert = AlbertModel(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        return self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers)


class MRC_finetune(nn.Module):
    def __init__(self, config):
        super(MRC_finetune, self).__init__()
        self.start_dense = nn.Linear(config.hidden_size, 1)
        self.end_dense = nn.Linear(config.hidden_size, 1)

    def forward(self, input_tensor):
        return self.start_dense(input_tensor), self.end_dense(input_tensor)


class AlBertJointForNQ(nn.Module):
    def __init__(self, config):
        super(AlBertJointForNQ, self).__init__()
        self.num_labels = config.num_labels
        self.config = config

        self.bert = AlbertModel(config)
        # long这里可以复用squad的权重，所以命名为finetune_mrc
        self.finetune_mrc = MRC_finetune(config)
        self.answer_types_outputs = nn.Linear(config.hidden_size, config.num_answer_types)

        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                start_positions=None, end_positions=None, answer_types=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_all_encoded_layers=False)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        start_logits, end_logits = self.finetune_mrc(sequence_output)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        answer_types_logits = self.answer_types_outputs(pooled_output)

        outputs = (start_logits, end_logits, answer_types_logits)
        if start_positions is not None and end_positions is not None and answer_types is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            if len(answer_types.size()) > 1:
                answer_types = answer_types.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            answer_type_loss = loss_fct(answer_types_logits, answer_types)
            total_loss = (start_loss + end_loss + answer_type_loss) / 3

            return total_loss

        else:
            return outputs


class AlBertJointForNQ2(nn.Module):
    def __init__(self, config, long_n_top=5, short_n_top=5):
        super(AlBertJointForNQ2, self).__init__()
        self.num_labels = config.num_labels
        self.long_n_top = long_n_top
        self.short_n_top = short_n_top
        self.config = config

        self.bert = AlbertModel(config)
        # long这里可以复用squad的权重，所以命名为finetune_mrc
        self.finetune_mrc = MRC_finetune(config)
        self.short_outputs = nn.Linear(config.hidden_size * 2, config.num_labels)
        self.answer_types_dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.cls_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.answer_types_outputs = nn.Linear(config.hidden_size, config.num_answer_types)

        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                long_start_positions=None, long_end_positions=None,
                short_start_positions=None, short_end_positions=None,
                answer_types=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_all_encoded_layers=False)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        long_start_logits, long_end_logits = self.finetune_mrc(sequence_output)
        long_start_logits = long_start_logits.squeeze(-1)
        long_end_logits = long_end_logits.squeeze(-1)

        # answer_type logits (only use the start feature)
        attention_mask = attention_mask.to(dtype=long_start_logits.dtype)
        long_start_logits_masked = long_start_logits + ((1 - attention_mask) * (-10000.0))
        long_end_logits_masked = long_end_logits + ((1 - attention_mask) * (-10000.0))
        long_start_softmax = torch.softmax(long_start_logits_masked, dim=1)  # [bs, len]
        long_end_softmax = torch.softmax(long_end_logits_masked, dim=1)  # [bs, len]
        # [bs,dim,len]x[bs,len,1]=[bs,dim,1]->[bs,dim]
        long_start_cls_feat = torch.matmul(sequence_output.transpose(1, 2),
                                           long_start_softmax.unsqueeze(-1)).squeeze(-1)
        long_end_cls_feat = torch.matmul(sequence_output.transpose(1, 2),
                                         long_end_softmax.unsqueeze(-1)).squeeze(-1)
        answer_type_feat = self.answer_types_dense(torch.cat([pooled_output,
                                                              long_start_cls_feat + long_end_cls_feat], dim=-1))
        answer_type_feat = self.cls_dropout(answer_type_feat)
        answer_type_logits = self.answer_types_outputs(answer_type_feat)

        # training process
        if long_start_positions is not None and long_end_positions is not None \
                and short_start_positions is not None and short_end_positions is not None \
                and answer_types is not None:
            # If we are on multi-GPU, split add a dimension
            if len(long_start_positions.size()) > 1:
                long_start_positions = long_start_positions.squeeze(-1)
            if len(long_end_positions.size()) > 1:
                long_end_positions = long_end_positions.squeeze(-1)
            if len(short_start_positions.size()) > 1:
                short_start_positions = short_start_positions.squeeze(-1)
            if len(short_end_positions.size()) > 1:
                short_end_positions = short_end_positions.squeeze(-1)
            if len(answer_types.size()) > 1:
                answer_types = answer_types.squeeze(-1)

            # loss setting
            ignored_index = long_start_logits.size(1)
            long_start_positions.clamp_(0, ignored_index)
            long_end_positions.clamp_(0, ignored_index)
            short_start_positions.clamp_(0, ignored_index)
            short_end_positions.clamp_(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

            # long loss
            long_start_loss = loss_fct(long_start_logits, long_start_positions)
            long_end_loss = loss_fct(long_end_logits, long_end_positions)

            # get short logits
            long_index = torch.zeros_like(attention_mask)  # [bs, len]
            long_index.scatter_add_(1, long_start_positions.unsqueeze(-1), torch.ones_like(long_index))
            long_index.scatter_add_(1, long_end_positions.unsqueeze(-1), torch.ones_like(long_index))  # [bs, len]
            # [bs, dim, len]x[bs, len, 1] = [bs, dim, 1]->[bs, 1, dim]
            long_features = torch.matmul(sequence_output.transpose(1, 2), long_index.unsqueeze(-1)).transpose(1, 2)
            long_features = long_features.repeat((1, sequence_output.shape[1], 1)) / 2.0  # [bs, len, dim]
            short_logits = self.short_outputs(torch.cat([sequence_output, long_features], dim=2))  # [bs, len, 2]
            short_start_logits, short_end_logits = short_logits.split(1, dim=-1)
            short_start_logits = short_start_logits.squeeze(-1)
            short_end_logits = short_end_logits.squeeze(-1)

            # short loss
            short_start_loss = loss_fct(short_start_logits, short_start_positions)
            short_end_loss = loss_fct(short_end_logits, short_end_positions)

            # answer_type_loss
            answer_type_loss = loss_fct(answer_type_logits, answer_types)

            total_loss = (long_start_loss + long_end_loss + short_start_loss + short_end_loss + answer_type_loss) / 3

            return total_loss

        else:  # test process
            # [bs, topk]
            long_start_topk_logits, long_start_topk_index = torch.topk(long_start_logits_masked,
                                                                       k=self.long_n_top, dim=1)
            long_end_topk_logits, long_end_topk_index = torch.topk(long_end_logits_masked,
                                                                   k=self.long_n_top, dim=1)
            long_topk_index = torch.zeros(size=(long_start_topk_index.size(0),
                                                long_start_topk_index.size(1),
                                                input_ids.size(1))).to(device=long_start_topk_logits.device,
                                                                       dtype=long_start_topk_logits.dtype)  # [bs, topk, len]
            # [bs, topk, len]
            long_topk_index.scatter_add_(2, long_start_topk_index.unsqueeze(-1), torch.ones_like(long_topk_index))
            long_topk_index.scatter_add_(2, long_end_topk_index.unsqueeze(-1), torch.ones_like(long_topk_index))
            # [bs, dim, len]x[bs, len, topk] = [bs, dim, topk]->[bs, topk, dim]
            long_features = torch.matmul(sequence_output.transpose(1, 2),
                                         long_topk_index.transpose(1, 2)).transpose(1, 2)
            long_features = long_features.unsqueeze(2)  # [bs, topk, 1, dim]
            long_features = long_features.repeat((1, 1, sequence_output.shape[1], 1)) / 2.0  # [bs, topk, len, dim]
            # [bs, topk, len, dim]
            sequence_topk_output = sequence_output.unsqueeze(1).repeat((1, self.long_n_top, 1, 1))
            # [bs, topk, len, 2]
            short_topk_logits = self.short_outputs(torch.cat([sequence_topk_output, long_features], dim=3))
            short_start_logits, short_end_logits = short_topk_logits.split(1, dim=-1)
            # [bs, topk, len]
            short_start_logits = short_start_logits.squeeze(-1)
            short_end_logits = short_end_logits.squeeze(-1)
            short_start_logits_masked = short_start_logits + ((1 - attention_mask.unsqueeze(1)) * (-10000.0))
            short_end_logits_masked = short_end_logits + ((1 - attention_mask.unsqueeze(1)) * (-10000.0))
            # [bs, topk, topk]
            short_start_topk_logits, short_start_topk_index = torch.topk(short_start_logits_masked,
                                                                         k=self.short_n_top, dim=2)
            short_end_topk_logits, short_end_topk_index = torch.topk(short_end_logits_masked,
                                                                     k=self.short_n_top, dim=2)

            outputs = {
                # [bs, topk]
                'long_start_topk_logits': long_start_topk_logits,
                'long_start_topk_index': long_start_topk_index,
                'long_end_topk_logits': long_end_topk_logits,
                'long_end_topk_index': long_end_topk_index,
                # [bs, topk, topk]
                'short_start_topk_logits': short_start_topk_logits,
                'short_start_topk_index': short_start_topk_index,
                'short_end_topk_logits': short_end_topk_logits,
                'short_end_topk_index': short_end_topk_index,
                # [bs, n_class]
                'answer_type_logits': answer_type_logits,
                # [bs,]
                'long_cls_logits': long_start_logits[:, 0] + long_end_logits[:, 0],
                # [bs, topk]
                'short_cls_logits': short_start_logits[:, :, 0] + short_end_logits[:, :, 0]
            }

            return outputs
