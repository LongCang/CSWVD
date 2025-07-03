"""PyTorch BERT model. """
import logging
import math
import os
import warnings
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.activations import gelu, gelu_new, swish
from transformers.configuration_bert import BertConfig
from transformers.file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_callable
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer

logger = logging.getLogger(__name__)  # 用于记录程序的运行状态、调试信息、警告、错误等信息。

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))  # 非线性激活函数，用于提升神经网络的性能。


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}
"""
ACT2FN 是一个激活函数映射字典，将激活函数的字符串名称映射到对应的函数对象。
这种设计允许根据字符串动态调用不同的激活函数，方便配置模型。
"""

BertLayerNorm = torch.nn.LayerNorm  # LayerNorm 是 Layer Normalization（层归一化），用于对隐藏状态（hidden states）或输入特征进行归一化处理。
"""
该行代码将 BertLayerNorm 变量指向 torch.nn.LayerNorm 类。
以后当使用 BertLayerNorm 时，实际上是在调用 torch.nn.LayerNorm。
这是为了在 BERT 模型中统一管理 LayerNorm，并与 HuggingFace 的 BERT 实现保持一致。
"""

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建 nn.Embedding 层，将 token ID 映射为嵌入向量。
        """
        config.vocab_size：词汇表的大小（即 token 的数量）。
        config.hidden_size：每个 token 嵌入向量的维度（如 768）。
        padding_idx=config.pad_token_id：pad token 的 ID，用于指定 padding 时的特殊处理。
        """
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        """
        为每个 token 添加位置信息，帮助模型理解序列中的位置关系。
        config.max_position_embeddings：最大位置嵌入维度，通常为 512。
        config.hidden_size：与 token 嵌入维度保持一致。
        """
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        """
        用于区分不同的句子（Segment Embedding），比如：0 表示第一个句子,1 表示第二个句子
        config.type_vocab_size：segment 的种类数，通常为 2（对应句子 A 和句子 B）。
        config.hidden_size：嵌入维度。
        """
        # 2022.10.8 variant_embeddings:  6 kinds of variant
        self.variant_embeddings = nn.Embedding(6, config.hidden_size)
        """
        用于处理敏感词变体的特定信息，并将其转换为嵌入向量。
        6 表示类别数量（6种敏感词的变体类型）。
        """
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        """
        LayerNorm 对嵌入进行层归一化，以加速训练并提高模型稳定性。
        通过规范化处理（zero mean, unit variance），防止梯度消失或梯度爆炸。
        """
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 防止模型过拟合，随机将部分神经元置为 0。
    # 2022.05.02 test token_tags
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, token_tags=None, variant_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        # 2022.10.8 variant_embeddings
        if variant_ids is not None:
            variant_embeddings = self.variant_embeddings(variant_ids)
            embeddings += variant_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        """
        将 query、key、value 从形状 [batch_size, seq_len, all_head_size] 转换为：
        [batch_size,num_heads,seq_len,head_size]
        """
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        """
        调用 transpose_for_scores()，将 query、key 和 value 转换为 [batch_size, num_heads, seq_len, head_size]。
        """

        # Take the dot product between "query" and "key" to get the raw attention scores.
        #  计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 添加注意力掩码
        """
        attention_mask 防止模型关注 padding 部分，通过添加一个非常小的负数来屏蔽 padding。
        attention_mask 通常形状为 [batch_size, 1, 1, seq_len]。
        """
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        # 重新排列上下文的维度，恢复为 [batch_size, seq_len, hidden_size] 的形状。
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs
        """
        返回结果
        context_layer：上下文表示 [batch_size, seq_len, hidden_size]。
        attention_probs：注意力分布（如果 output_attentions=True）。
        """


"""
负责对 BERT 自注意力模块 (Self-Attention) 的输出进行 线性变换、层归一化 (LayerNorm) 和 Dropout。
这一部分是 BERT 模型中残差连接 (Residual Connection)和层归一化 的核心部分。
"""
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        """
        dense 层用于对 self-attention 的输出进行线性变换。
        输入和输出维度均为 hidden_size（例如 768）。
        dense 层将 self-attention 生成的上下文向量映射回原始维度，准备与 input_tensor 进行残差连接。
        """
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        """
        LayerNorm 进行层归一化，保持模型的稳定性。
        归一化可以防止 梯度消失 或 梯度爆炸，加速模型收敛。
        """
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



"""
BertAttention 实现了 BERT 模型的注意力模块 (Attention Block)。
该模块包括：
Self-Attention (BertSelfAttention)：计算自注意力机制。
输出层 (BertSelfOutput)：对 Self-Attention 结果进行线性变换、残差连接、Dropout 和 LayerNorm。
还支持 注意力头裁剪 (Pruning Attention Heads)，减少计算量。
"""
class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs



"""
BertIntermediate 模块是 BERT 模型中 前馈神经网络 (FFN) 的 第一部分。
它执行 线性变换 和 激活函数，将 hidden_states 由 hidden_size 投影到 intermediate_size
"""
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

"""
BertOutput 是 BERT 模型 前馈神经网络 (FFN) 的 第二部分。
它对 BertIntermediate 的输出进行 线性变换 (dense)、Dropout、残差连接 (Residual Connection) 和 层归一化 (LayerNorm)。
"""
class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

"""
BertLayer 是 BERT 模型中的 Transformer 编码层，包含：
自注意力 (BertAttention)
交叉注意力 (BertAttention，仅解码器启用)
前馈神经网络 (BertIntermediate 和 BertOutput)
BertLayer 通过 注意力机制 和 前馈神经网络 对输入 hidden_states 进行编码，为 BERT 模型提供强大的特征表达能力。
"""
class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
         # 替换为 MoE
        # self.moe = MoE(config, num_experts=4)  # 可配置专家数量
        # self.output = BertOutput(config)  # 保留原有输出层

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

            # 替换前馈部分为 MoE
        # intermediate_output = self.moe(attention_output)
        # layer_output = self.output(intermediate_output, attention_output)
        # outputs = (layer_output,) + outputs
        # return outputs
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

"""
BertEncoder 是 BERT 模型中 Transformer 编码器 (Encoder) 的实现。
它包含 多个 BertLayer 层，用于逐层处理输入数据，通过自注意力机制和前馈神经网络捕获上下文信息。
BertEncoder 是 BERT 模型的 Transformer 编码器 (Encoder)，通过堆叠 BertLayer 逐层处理 hidden_states。
BertEncoder 支持返回所有层的隐藏状态 (all_hidden_states) 和注意力权重 (all_attentions)，用于分析模型性能。
BertEncoder 是 BERT 模型最核心的组成部分之一，理解它有助于掌握 Transformer 的编码机制。
"""
class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

"""
BertPooler 是 BERT 模型中用于 池化 (Pooling) 的模块。
它提取 [CLS] 标记的隐藏状态 (hidden_state) 作为整个输入序列的表示，并通过 线性变换 (dense layer) 和 Tanh 激活函数 生成池化后的输出。
1. 提取 [CLS] 标记的隐藏状态
BertPooler 仅取 [CLS] 标记作为整个序列的全局表示。
2. 进行线性变换和 Tanh 激活
通过 dense 层进行线性变换，并使用 Tanh 进行非线性映射。
3. 生成句子的固定长度向量
pooled_output 可用于句子分类、相似度计算、情感分析等下游任务。
"""
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

"""
BertPreTrainedModel 继承自 PreTrainedModel，是 BERT 预训练模型的基类。
该类定义了 BERT 模型的 权重初始化方法，用于为 BERT 模型的不同层设置初始参数。
_init_weights() 会在 BERT 模型初始化时自动被调用，为 BERT 模型的不同层赋予初始权重。
BertPreTrainedModel 作为 BERT、BERT-Base、BERT-Large 等模型的基类，适用于 BERT 体系结构的所有模块。
"""
class BertPreTrainedModel(PreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


"""
BertModel 是 BERT 模型的核心实现。
继承自 BertPreTrainedModel，提供了 BERT 的完整结构，包括：
嵌入层 (BertEmbeddings)
编码器 (BertEncoder)
池化层 (BertPooler)
该类定义了 BERT 模型的 前向传播逻辑 和 权重管理功能。
1. 处理 BERT 的完整前向传播流程
包括嵌入层、编码器层和池化层的前向传播。
2. 提供 hidden_states 和 attentions
可以选择返回每一层的 hidden_states 和注意力权重。
3. 适用于下游任务
pooled_output 可用于分类任务。
sequence_output 可用于命名实体识别、问答等任务。
"""
class BertModel(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        variant_ids=None, 
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)