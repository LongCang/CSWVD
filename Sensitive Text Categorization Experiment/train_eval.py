import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import time
import json
import copy
from src.datasets import get_time_dif, to_tensor, convert_onehot
from src.Models import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train(config, train_iter, dev_iter, test_iter, task=1):
    embed_model = Bert_Layer(config).to(config.device)
    model = MoE(config).to(config.device)
    # variant_embeddings = embed_model.bert_layer.embeddings.variant_embeddings
    # model = TwoLayerFFNNLayer(config).to(config.device)
    # model = MoE(config, variant_embeddings=variant_embeddings).to(config.device)
    model_name = '{}-NN_ML-{}_D-{}_B-{}_E-{}_Lr-{}_aplha-{}'.format(
        config.model_name, config.pad_size, config.dropout,
        config.batch_size, config.num_epochs, config.learning_rate, config.alpha1
    )
    # embed_optimizer = optim.AdamW(embed_model.parameters(), lr=config.learning_rate)
    # model_optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    embed_optimizer = optim.AdamW(embed_model.parameters(), lr=config.learning_rate, weight_decay=1e-4)  # 增加权重衰减
    model_optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(model_optimizer, mode='max', factor=0.5, patience=1)  # 学习率调度
    loss_fn = nn.BCEWithLogitsLoss()
    max_score = 0

    for epoch in range(config.num_epochs):
        embed_model.train()
        model.train()
        start_time = time.time()
        print("Model is training in epoch {}".format(epoch))
        loss_all = 0.
        preds = []
        labels = []

        for batch in tqdm(train_iter, desc='Training', colour='MAGENTA'):
            embed_model.zero_grad()
            model.zero_grad()
            args = to_tensor(batch)
            att_input, pooled_emb = embed_model(**args)  # 通过Bert模型获取序列输出和池化输出
            # logit = model(att_input, pooled_emb).cpu()
            logit = model(att_input, pooled_emb, args['variant_ids'].to(config.device)).cpu()  # 更新为 MoE 的调用
            label = args['variant']
            loss = loss_fn(logit, label.float())
            pred = get_preds(config, logit)
            preds.extend(pred)
            labels.extend(label.detach().numpy())
            loss_all += loss.item()
            embed_optimizer.zero_grad()
            model_optimizer.zero_grad()
            loss.backward()
            embed_optimizer.step()
            model_optimizer.step()

        end_time = time.time()
        print(" took: {:.1f} min".format((end_time - start_time) / 60.))
        print("TRAINED for {} epochs".format(epoch))

        if epoch >= config.num_warm:
            trn_scores = get_scores(preds, labels, loss_all, len(train_iter), data_name="TRAIN")
            dev_scores, _ = eval(config, embed_model, model, loss_fn, dev_iter, data_name='DEV')
            scheduler.step(dev_scores['F1'])  # 根据验证集 F1 调整学习率
            f = open('{}/{}.all_scores.txt'.format(config.result_path, model_name), 'a')
            f.write(' ==================================================  Epoch: {}  ==================================================\n'.format(epoch))
            f.write('TrainScore: \n{}\nEvalScore: \n{}\n'.format(json.dumps(trn_scores), json.dumps(dev_scores)))
            max_score = save_best(config, epoch, model_name, embed_model, model, dev_scores, max_score)
        print("ALLTRAINED for {} epochs".format(epoch))

    path = '{}/ckp-{}-{}.tar'.format(config.checkpoint_path, model_name, 'BEST')
    checkpoint = torch.load(path)
    embed_model.load_state_dict(checkpoint['embed_model_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    test_scores, _ = eval(config, embed_model, model, loss_fn, test_iter, data_name='TEST')
    f = open('{}/{}.all_scores.txt'.format(config.result_path, model_name), 'a')
    f.write('Test: \n{}\n'.format(json.dumps(test_scores)))


def eval(config, embed_model, model, loss_fn, dev_iter, data_name='DEV'):
    loss_all = 0.
    preds = []
    labels = []
    for batch in tqdm(dev_iter, desc='Evaling', colour='CYAN'):
        with torch.no_grad():
            args = to_tensor(batch)
            att_input, pooled_emb = embed_model(**args)
            logit = model(att_input, pooled_emb, args['variant_ids'].to(config.device)).cpu()  # 传递 variant_ids
            # logit = model(att_input, pooled_emb).cpu()
            label = args['variant']
            loss = loss_fn(logit, label.float())
            pred = get_preds(config, logit)
            preds.extend(pred)
            labels.extend(label.detach().numpy())
            loss_all += loss.item()
    dev_scores = get_scores(preds, labels, loss_all, len(dev_iter), data_name=data_name)
    return dev_scores, preds

# For Multi Classfication
# 取最大概率的类别索引，转为 one-hot 编码
def get_preds(config, logit):
    results = torch.max(logit.data, 1)[1].cpu().numpy()
    new_results = []
    for result in results:
        result = convert_onehot(config, result)
        new_results.append(result)
    # return new_results
    return (torch.sigmoid(logit) > 0.5).int().cpu().numpy().tolist()

def get_preds_task2_4(config, logit):
    all_results = []
    logit_ = torch.sigmoid(logit)
    results_pred = torch.max(logit_.data, 1)[0].cpu().numpy()
    results = torch.max(logit_.data, 1)[1].cpu().numpy() # index for maximum probability
    for i in range(len(results)):
        if results_pred[i] < 0.5:
            result = [0 for i in range(config.num_classes)]
        else:
            result = convert_onehot(config, results[i])
        all_results.append(result)
    return all_results

# Task 3: 多标签分类 Targeted Group Detection
def get_preds_task3(config, logit):
    all_results = []
    logit_ = torch.sigmoid(logit)
    results_pred = torch.max(logit_.data, 1)[0].cpu().numpy()
    results = torch.max(logit_.data, 1)[1].cpu().numpy()
    logit_ = logit_.detach().cpu().numpy()
    for i in range(len(results)):
        if results_pred[i] < 0.5:
            result = [0 for i in range(config.num_classes)]
        else:
            result = get_pred_task3(logit_[i])
        all_results.append(result)
    return all_results

def get_pred_task3(logit):
    result = [0 for i in range(len(logit))]
    for i in range(len(logit)):
        if logit[i] >= 0.5:
            result[i] = 1
    return result

def get_scores(all_preds, all_lebels, loss_all, len, data_name):
    # 计算评估指标
    score_dict = dict()
    # 计算加权 F1、类别 F1、精确度和召回率。
    f1 = f1_score(all_preds, all_lebels, average='weighted')
    # acc = accuracy_score(all_preds, all_lebels)
    all_f1 = f1_score(all_preds, all_lebels, average=None)
    pre = precision_score(all_preds, all_lebels, average='weighted')
    recall = recall_score(all_preds, all_lebels, average='weighted')
    # 存储指标和平均损失。
    score_dict['F1'] = f1
    # score_dict['accuracy'] = acc
    score_dict['all_f1'] = all_f1.tolist()
    score_dict['precision'] = pre
    score_dict['recall'] = recall

    score_dict['all_loss'] = loss_all/len # 平均损失
    # 打印并返回分数字典
    print("Evaling on \"{}\" data".format(data_name))
    for s_name, s_val in score_dict.items(): 
        print("{}: {}".format(s_name, s_val)) 
    return score_dict

def save_best(config, epoch, model_name, embed_model, model, score, max_score):
    # 用于比较的关键指标（如 F1）。
    score_key = config.score_key
    curr_score = score[score_key]
    # 打印当前和历史最高得分
    print('The epoch_{} {}: {}\nCurrent max {}: {}'.format(epoch, score_key, curr_score, score_key, max_score))

    # 如果当前得分高于历史最高或为第 0 次迭代，保存模型并返回当前得分；否则返回历史最高得分。
    if curr_score > max_score or epoch == 0:
        torch.save({
        'epoch': config.num_epochs,
        'embed_model_state_dict': embed_model.state_dict(),
        'model_state_dict': model.state_dict(),
        }, '{}/ckp-{}-{}.tar'.format(config.checkpoint_path, model_name, 'BEST'))
        return curr_score
    else:
        return max_score
