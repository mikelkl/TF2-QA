import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


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


class RobertaJointForLong(nn.Module):
    def __init__(self, bert, config):
        super(RobertaJointForLong, self).__init__()
        self.bert = bert
        self.config = config
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
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

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        sequence_output, _, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits

class RobertaJointForNQ2(nn.Module):
    def __init__(self, bert, config, long_n_top=5, short_n_top=5):
        super(RobertaJointForNQ2, self).__init__()
        self.num_labels = config.num_labels
        self.long_n_top = long_n_top
        self.short_n_top = short_n_top

        self.bert = bert
        self.config = config
        # long这里可以复用squad的权重，所以命名为qa_outputs
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
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

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        long_logits = self.qa_outputs(sequence_output)
        long_start_logits, long_end_logits = long_logits.split(1, dim=-1)
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