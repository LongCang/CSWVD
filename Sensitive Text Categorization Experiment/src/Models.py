from transformers import BertModel
# from chinesebert import ChineseBertForMaskedLM, ChineseBertTokenizerFast, ChineseBertConfig
from src.BERT import BertModel
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# 2022.4.28 Glove + LSTM
class BiLSTM(nn.Module):
    def __init__(self, config, embedding_weight):
        super(BiLSTM, self).__init__()
        self.device = config.device
        self.vocab_size = embedding_weight.shape[0] # 词汇表大小，从词嵌入权重的第一维获取。
        self.embed_dim = embedding_weight.shape[1] # 词嵌入维度，从词嵌入权重的第二维获取。
        # Embedding Layer
        embedding_weight = torch.from_numpy(embedding_weight).float() # 将 NumPy 格式的词嵌入权重转换为 PyTorch 张量，并转为浮点型。
        embedding_weight = Variable(embedding_weight, requires_grad=config.if_grad)
        # 将张量包装为 Variable，并根据 config.if_grad 设置是否需要梯度（True 表示词嵌入可训练）。
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, _weight=embedding_weight)  # 使用 nn.Embedding 将输入 token ID 映射到词向量。
        # Encoder layer
        self.bi_lstm = nn.LSTM(self.embed_dim, config.lstm_hidden_dim, bidirectional=True, batch_first=True)

    def forward(self, **kwargs):
        emb = self.embedding(kwargs["title_text_token_ids"].to(self.device)) # [batch, len] --> [batch, len, embed_dim]
        lstm_out, _ = self.bi_lstm(emb)  # [batch, len, embed_dim] --> [batch, len, lstm_hidden_dim*2]
        lstm_out_pool = torch.mean(lstm_out, dim=1)  # [batch, lstm_hidden_dim*2]
        return lstm_out, lstm_out_pool


class Bert_Layer(torch.nn.Module):
    def __init__(self, config):
        super(Bert_Layer, self).__init__()
        # self.use_cuda = kwargs['use_cuda']
        self.device = config.device
        # BERT/Roberta
        self.bert_layer = BertModel.from_pretrained(config.model_name)
        # ChineseBERT
        # self.config = ChineseBertConfig.from_pretrained(config.model_name)
        # self.bert_layer = ChineseBertForMaskedLM.from_pretrained("ShannonAI/ChineseBERT-base", config=self.config)
        self.dim = config.vocab_dim

    def forward(self, **kwargs):
        bert_output = self.bert_layer(input_ids=kwargs['text_idx'].to(self.device),
                                 token_type_ids=kwargs['text_ids'].to(self.device),
                                 attention_mask=kwargs['text_mask'].to(self.device),
                                 variant_ids=kwargs["variant_ids"].to(self.device))
        return bert_output[0], bert_output[1]

class TwoLayerFFNNLayer(torch.nn.Module):
    '''
    2-layer FFNN with specified nonlinear function
    must be followed with some kind of prediction layer for actual prediction
    具有指定非线性函数的 2 层 FFNN，必须跟随着某种预测层才能进行实际预测
    '''
    def __init__(self, config):
        super(TwoLayerFFNNLayer, self).__init__()
        self.output = config.dropout
        self.input_dim = config.vocab_dim
        self.hidden_dim = config.fc_hidden_dim
        self.out_dim = config.num_classes
        self.dropout = nn.Dropout(config.dropout)
        self.model = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                   nn.Tanh(),
                                   nn.Linear(self.hidden_dim, self.out_dim))

    def forward(self, att_input, pooled_emb):
        att_input = self.dropout(att_input)
        pooled_emb = self.dropout(pooled_emb)
        return self.model(pooled_emb)

# 改进：加入序列信息的多模态门控
class EnhancedGate(nn.Module):
    def __init__(self, seq_dim, pooled_dim, num_experts):
        super().__init__()
        # 融合序列特征和池化特征
        self.seq_proj = nn.Linear(seq_dim, pooled_dim)
        # self.fc = nn.Sequential(
        #     nn.Linear(pooled_dim * 2, pooled_dim),
        #     nn.ReLU(),
        #     nn.Linear(pooled_dim, num_experts)
        # )
        self.attn = nn.MultiheadAttention(pooled_dim, num_heads=4, batch_first=True)
        self.fc = nn.Linear(pooled_dim, num_experts)
    # def forward(self, seq_output, pooled_output):
    #     seq_feat = self.seq_proj(seq_output.mean(dim=1))  # [batch, pooled_dim]
    #     combined = torch.cat([seq_feat, pooled_output], dim=1)
    #     return F.softmax(self.fc(combined), dim=-1)
    def forward(self, seq_output, pooled_output):
        # seq_feat = self.seq_proj(seq_output)
        # attn_out, _ = self.attn(pooled_output.unsqueeze(0), seq_feat, seq_feat)
        # return F.softmax(self.fc(attn_out.squeeze(0)), dim=-1)
        # seq_output: [batch, seq_len, pooled_dim]
        seq_feat = self.seq_proj(seq_output)
        # pooled_output: [batch, pooled_dim]，需要扩展成 [batch, 1, pooled_dim]作为 query
        query = pooled_output.unsqueeze(1)
        attn_out, _ = self.attn(query, seq_feat, seq_feat)
        # attn_out: [batch, 1, pooled_dim] → squeeze 得到 [batch, pooled_dim]
        return F.softmax(self.fc(attn_out.squeeze(1)), dim=-1)

class SensitiveExpert(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.variant_embed = nn.Embedding(6, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.proj = nn.Linear(hidden_size, num_classes)

    def forward(self, seq_output, variant_ids):
        # variant_ids: [batch, seq_len]
        variant_emb = self.variant_embed(variant_ids)  # [batch, seq_len, hidden_size]
        combined = seq_output + variant_emb  # [batch, seq_len, hidden_size]
        attn_out, _ = self.attention(combined, seq_output, seq_output)
        # attn_out: [batch, seq_len, hidden_size]，在 seq_len 维度上取平均 [batch, hidden_size]
        return self.proj(attn_out.mean(dim=1))


class EmotionExpert(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            bidirectional=True,
            batch_first=True
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, seq_output):
        lstm_out, _ = self.lstm(seq_output)
        return self.proj(lstm_out.mean(dim=1))


class SemanticExpert(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(hidden_size, nhead=4)
        self.proj = nn.Linear(hidden_size, num_classes)

    def forward(self, seq_output):
        trans_out = self.transformer(seq_output)
        return self.proj(trans_out.mean(dim=1))


class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.vocab_dim
        self.num_classes = config.num_classes

        self.experts = nn.ModuleDict({
            'variant': SensitiveExpert(self.hidden_size, self.num_classes),
            'emotion': EmotionExpert(self.hidden_size, self.num_classes),
            'semantic': SemanticExpert(self.hidden_size, self.num_classes)
        })

        self.gate = EnhancedGate(
            seq_dim=self.hidden_size,
            pooled_dim=self.hidden_size,
            num_experts=3
        )

        # self.dropout = nn.Dropout(config.dropout)
        # self.balance_loss_weight = 0.01
        self.balance_loss_weight = 0.01
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, seq_output, pooled_output, variant_ids):
        # print("seq_output shape:", seq_output.shape)  # 应为 [32, 80, 768]
        # print("pooled_output shape:", pooled_output.shape)  # 应为 [32, 768]
        # print("MoE: variant_ids shape:", variant_ids.shape)  # 预期 [32, 80]
        seq_output = self.dropout(seq_output)

        variant_out = self.experts['variant'](seq_output, variant_ids)
        emotion_out = self.experts['emotion'](seq_output)
        semantic_out = self.experts['semantic'](seq_output)

        gate_weights = self.gate(seq_output, pooled_output)
        # prob_mean = gate_weights.mean(dim=0)
        # balance_loss = -(prob_mean * torch.log(prob_mean + 1e-7)).sum()

        expert_outputs = torch.stack([variant_out, emotion_out, semantic_out], dim=1)
        moe_output = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)

        return moe_output