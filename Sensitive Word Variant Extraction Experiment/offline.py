import os
import argparse
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
import json
import numpy as np
import torch
from transformers import set_seed
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
from vllm import SamplingParams
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from dataset import *
from preprocess import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
VLLM_GPU_NUMS = torch.cuda.device_count()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(VLLM_GPU_NUMS)))
# VLLM_GPU_NUMS = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embedding_model = SentenceTransformer('embedding_model/LaBSE', local_files_only=True).to(device)  # 🔺


# Utils #
def batch_generator(texts, batch_size):
    # texts: list[str]
    for i in range(0, len(texts), batch_size):
        yield texts[i:i + batch_size]


# check dirs
def check_dir_existence(model_name, dataset):
    dirs = [f'predictions/{model_name}/{dataset}/']
    for dir in dirs:
        if not os.path.exists(dir):
            raise ValueError('Prediction and prompt path not exist, please create them.')

def extract_complete_sentences(text, max_words=60):
    sentences = re.findall(r'[^。？！.!?]*[。？！.!?]', text)
    last_sentence = sentences[-1] if sentences else text
    is_complete = re.match(r'.*[。？！.!?]$', last_sentence) is not None
    output = ""
    if is_complete:
        output = sentences[0]
        words_num = len(output.split(" "))
        if len(sentences) > 1:
            for s in sentences[1:]:
                words_num += len(s.split(" "))
                if words_num > max_words:
                    break
                output += f" {s}"
    else:
        words = text.split(" ")
        output = " ".join(words[:100]) + "."
    return output

def evaluation(gold_samples, predictions, labels, threshold=0.5):
    """
    Soft-span-level evaluation by thresholding prediction scores.
    """
    TP = FP = FN = 0
    for gold_obj, pred_dict in zip(gold_samples, predictions):
        text = gold_obj["text"]
        gold_entities = {f"{e}:{t}" for e, t in gold_obj["label"]}

        # filter preds by threshold and valid
        pred_entities = set()
        for ent, (typ, score) in pred_dict.items():
            if score >= threshold and ent in text and typ in labels:
                pred_entities.add(f"{ent}:{typ}")

        # compute overlaps
        tp = len(gold_entities & pred_entities)
        fp = len(pred_entities - gold_entities)
        fn = len(gold_entities - pred_entities)

        TP += tp
        FP += fp
        FN += fn

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision,
            "recall": recall,
            "f1": f1,
            "support": TP + FN,
            "pred_count": TP + FP,
            "true_positive": TP}

def repair_string(s):
    s = s.replace('": "', '":"')
    pattern = r'"([^"]+)":"([^"]+)"'
    matches = re.findall(pattern, s)
    tmp = []
    for k, v in matches:
        tmp.append(f'"{k}": "{v}"')
    repaired_s = '{' + f'{", ".join(tmp)}' + '}'
    return repaired_s


def filter_entity_predictions(predictions, texts, labels):
    filtered_predictions = []
    for preds, text in zip(predictions, texts):
        tmp = {}
        for e, t in preds.items():
            if (e in text) and (t in labels):
                tmp[e] = t
        filtered_predictions.append(tmp)
    return filtered_predictions


def count_pred_entities(predictions):
    num = 0
    for preds in predictions:
        num += len(preds)
    return num


# #
def generate_predictions(prompt, texts, llm, batch_size, pred_path, labels):
    sampling_params = SamplingParams(temperature=0, top_p=0.5, max_tokens=64, stop='}', seed=0)
    # batch generate # # # # # #
    all_predictions = []
    total_batches = len(texts) // batch_size + (1 if len(texts) % batch_size != 0 else 0)
    for batch in tqdm(batch_generator(texts, batch_size), total=total_batches, desc=f"Generate predictions"):
        context = [str(prompt + f'Text: {text}.\tAnswer: ') for text in batch]
        response = llm.generate(context, sampling_params, use_tqdm=True)
        response = [str(response[i].outputs[0].text.replace("\n", "").strip() + '}') for i in range(len(response))]
        predictions = []
        for r in response:
            try:
                predictions.append(json.loads(r))
            except:
                try:
                    predictions.append(json.loads(repair_string(r)))
                except:
                    r_error = '{"Error":"Error"}'
                    predictions.append(json.loads(repair_string(r_error)))
        batch_texts = [text for text in batch]
        predictions = filter_entity_predictions(predictions, batch_texts, labels)
        all_predictions.extend(predictions)
        with open(pred_path, mode='a', encoding='utf-8') as f:
            for preds in predictions:
                # f.write(json.dumps(preds) + '\n')
                f.write(json.dumps(preds, ensure_ascii=False) + '\n')  # 添加 ensure_ascii=False
    return all_predictions


def post_process_preds(historical_preds, thresh_num):
    weighted_preds = {}
    counts_preds = {}
    for i, preds in enumerate(historical_preds):
        weight = i + 1
        # weight = 1
        for e, t in preds.items():
            # prediction nums of the entity
            if e not in counts_preds:
                counts_preds[e] = 1
            else:
                counts_preds[e] += 1
            # weights of each type of the entity
            if e not in weighted_preds:
                weighted_preds[e] = {t: weight}
            else:
                if t not in weighted_preds[e]:
                    weighted_preds[e][t] = weight
                else:
                    weighted_preds[e][t] += weight
    thresh_w = len(historical_preds)
    # thresh_n = len(historical_preds) // 2
    thresh_n = thresh_num
    intergrated_pred = {}
    for e in weighted_preds.keys():
        weighted_types = weighted_preds[e]
        max_weight_t = max(weighted_types, key=weighted_types.get)
        if (weighted_types[max_weight_t] > thresh_w) and (counts_preds[e] > thresh_n):
            # if weighted_types[max_weight_t] >  thresh_n:
            intergrated_pred[e] = max_weight_t
    return intergrated_pred


def post_process_predictions(args, historical_predictions, historical_pred_nums):
    if len(historical_predictions) == 1:
        return historical_predictions[0]
    else:
        current_step_num, historical_avg_num = historical_pred_nums[-1], sum(historical_pred_nums[:-1]) / (
                    len(historical_pred_nums) - 1)
        thresh = len(historical_predictions) // 2
        t = 0.1
        if (args.dataset == 'ACE05-E') or (args.dataset == 'CONLL'):
            t = 0.15
        if current_step_num >= historical_avg_num:
            thresh += ((current_step_num - historical_avg_num) / historical_avg_num) // t
            thresh = min(thresh, len(historical_predictions) - 1)  # 约束
        else:
            thresh -= ((historical_avg_num - current_step_num) / historical_avg_num) // t
            thresh = max(thresh, 2)
        intergrated_predictions = []
        for sample_id in range(len(historical_predictions[0])):
            historical_preds = [predictions[sample_id] for predictions in historical_predictions]
            intergrated_pred = post_process_preds(historical_preds, thresh)
            intergrated_predictions.append(intergrated_pred)
        return intergrated_predictions
def split_by_types(predictions, texts, labels):
    """
    # Return: { type: list[json_objs], ...}
    #   json_obj → {'text':..., 'entities':[...], 'text_id':..}
    """
    prediction_splits = {}
    for label in labels:
        subset = []
        for i in range(len(texts)):
            text, text_id, SensitiveWords = texts[i], i, []
            for pair in predictions[i].items():
                if pair[1] == label:
                    SensitiveWords.append(pair[0])
            if len(SensitiveWords) > 0:
                subset.append({"text":text, "SensitiveWords": SensitiveWords, "text_id": text_id})
            # save_path = f'tmp_files/tmp_{label}.json'
            # with open(save_path, mode='w', encoding='utf-8') as f:
            #     json.dump(subset, f, ensure_ascii=False, indent=2)
        prediction_splits[label] = subset
    return prediction_splits


# select representative samples # # # #
def get_cluster_center_ids(sentences, embedding_model, eps=0.5, min_samples=5):
    """
    使用 DBSCAN 聚类选择代表性样本的索引。
    参数:
        sentences: list[str]，输入的句子列表
        embedding_model: 嵌入模型，用于生成句子嵌入
        eps: float，DBSCAN 的距离阈值（默认 0.5）
        min_samples: int，DBSCAN 的最小样本数（默认 5）
    返回:
        list[int]，代表性样本的索引列表
    """
    # 如果样本数量少于 min_samples，直接返回所有索引
    if len(sentences) <= min_samples:
        return [i for i in range(len(sentences))]
    # 生成句子嵌入
    embeddings = embedding_model.encode(sentences, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
    # L2 归一化嵌入
    embeddings_normalized = normalize(embeddings, norm='l2')
    # 应用 DBSCAN 聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings_normalized)
    labels = dbscan.labels_  # 簇标签，-1 表示噪声点
    # 获取有效簇（排除噪声点）
    unique_labels = set(labels) - {-1}  # 移除噪声标签
    if not unique_labels:  # 如果没有有效簇，返回前 min_samples 个样本的索引
        return [i for i in range(min(len(sentences), min_samples))]
    # 从每个簇中选择代表性样本
    closest_embedding_ids = []
    for cluster_label in unique_labels:
        # 获取属于当前簇的样本索引
        cluster_indices = np.where(labels == cluster_label)[0]
        cluster_embeddings = embeddings_normalized[cluster_indices]
        # 计算簇内嵌入的平均值（类似质心）
        cluster_center = np.mean(cluster_embeddings, axis=0)
        # 找到距离质心最近的样本
        distances = cdist([cluster_center], cluster_embeddings, 'cosine')[0]
        closest_idx_in_cluster = np.argmin(distances)
        closest_embedding_ids.append(cluster_indices[closest_idx_in_cluster])

    return closest_embedding_ids


def select_typical_samples(type, num, predictions_type):
    sentences = []
    for obj in predictions_type:
        if len(obj['SensitiveWords']) == 1:
            # s = f'In text "{obj["text"]}", the {type} entity is: {obj["entities"][0]}.'
            s = f'在文本 "{obj["text"]}中",  {type}类型的敏感词是: {obj["SensitiveWords"][0]}.'
        else:
            # s = f'In text "{obj["text"]}", the {type} entities are: {", ".join(obj["entities"])}.'
            s = f'在文本"{obj["text"]}中",  {type}类型的敏感词有: {", ".join(obj["SensitiveWords"])}.'
        sentences.append(s)
    typical_sample_ids_subset = get_cluster_center_ids(sentences, embedding_model, num)
    typical_sample_ids = []
    for idx in typical_sample_ids_subset:
        typical_sample_ids.append(predictions_type[idx]['text_id'])
    return typical_sample_ids_subset, typical_sample_ids

# 生成敏感词类型的定义 # # # # # # # # # # # #
def generate_guideline_definition(type, typical_samples, llm, last_step_guide):
    if last_step_guide == '':
        prompt = f'以下是一些包含{type}敏感词变体的文本。请根据这些示例总结“{type}”类型的定义。.\n\n'
    else:
        prompt = f'目前，“{type}”类型定义为：“{last_step_guide}”。". '
        prompt = f'以下是一些包含 {type} 敏感词变体的文本。根据这些示例，补充或修改定义以使其更完整。\n\n'
    for obj in typical_samples:
        if len(obj['SensitiveWords']) == 1:
            s = f'在文本 "{obj["text"]}中",  {type}的敏感词是: {obj["SensitiveWords"][0]}.'
        else:
            s = f'在文本"{obj["text"]}中",  {type}类型的敏感词有: {", ".join(obj["SensitiveWords"])}.'
        prompt += s
    prompt += f'\n根据这些示例，“{type}”类型是指：'
    sampling_params = SamplingParams(temperature=0, top_p=0.5, max_tokens=64, seed=0)
    response = llm.generate([str(prompt)], sampling_params, use_tqdm=False)[0].outputs[0].text.replace("\n", "").strip()
    guideline_definition = extract_complete_sentences(text=str(response), max_words=64)
    return guideline_definition

def data_format_convert(predictions_type):
    e_texts = {}
    for obj in predictions_type:
        for SensitiveWord in obj['SensitiveWords']:
            if SensitiveWord in e_texts:
                e_texts[SensitiveWord].append(obj['text_id'])
            else:
                e_texts[SensitiveWord] = [obj['text_id']]
    return e_texts

def save_prompt(guidelines_definition, few_shot_samples, labels, prompt_path, step):
    s = "'" + "', '".join(labels) + "'"
    prompt = f"给定敏感词变体的类型：[{s}]。每种类型的定义为：\n"
    i = 1
    for type, definition in zip(labels, guidelines_definition):
        prompt += f'({i}) {type}: {definition}\n'
        i += 1
    # v1
    # prompt += "\nPlease recognize the named entities in the given text only belonging to the given types. Provide answer in the following JSON format: {\"entity\": \"type\"}. If there is no entity in the text, return the following empty object: {}.\n"
    # v2
    prompt += "\n请识别给定文本中仅属于给定类型的变体敏感词。以以下 JSON 格式提供答案：{“SensitiveWord”： “type”}。如果文本中没有变体敏感词，则返回以下空对象：{}。\n"
    prompt += '以下是一些示例：\n\n'
    for samples in few_shot_samples:
        prompt += f'{samples}\n\n'
    with open(prompt_path, mode='r', encoding='utf-8') as f:
        PROMPTS = json.load(f)
    PROMPTS[f'step{step}'] = str(prompt)
    with open(prompt_path, mode='w', encoding='utf-8') as f:
        PROMPTS = json.dump(PROMPTS, f, ensure_ascii=False, indent=2)
    return prompt


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def main():
    parser = argparse.ArgumentParser(description="Args for Auto Guides.")

    parser.add_argument("--dataset", type=str, default='data-debug')

    parser.add_argument("--model_name", type=str, default='Qwen2.5-7B-Instruct',
                        help="The name or path to the model.")

    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=4, help='The number of prediction iterations.')

    parser.add_argument("--shots_per_type", type=int, default=2, help='The few shot sample number for each type.')

    # for Debug # # # # # # # # # # # # #
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--start_sample_idx", type=int, default=0, help='only valid for the first(start) step')

    args = parser.parse_args()

    args.model_path = f'model/{args.model_name}'  # 🔺

    set_seed(0)
    check_dir_existence(args.model_name, args.dataset)

    test_data_path = f'dataset/{args.dataset}/test.json'  # 🔺
    label_data_path = f'dataset/{args.dataset}/labels.txt'  # 🔺
    with open(test_data_path, mode='r', encoding='utf-8') as f:
        gold_samples = json.load(f)
    texts = [str(obj['text']) for obj in gold_samples]
    labels = []
    with open(label_data_path, mode='r', encoding='utf-8') as f:
        for line in f:
            labels.append(str(line.strip()))
    print(f'{args.dataset}: {len(texts)} samples.\nLabels: {labels}')

    # Main process
    llm = LLM(model=args.model_path, trust_remote_code=True, tensor_parallel_size=VLLM_GPU_NUMS, max_model_len=2048,
              gpu_memory_utilization=0.85, enforce_eager=True)
    evaluation_path = f'predictions/{args.model_name}/{args.dataset}/evaluation_results.txt'  # ⭐
    # prompt_path = f'prompts/prompts_{args.dataset}.json'
    prompt_path = f'predictions/{args.model_name}/{args.dataset}/prompts_{args.dataset}.json'  # ⭐
    PROMPT = ''

    # seed guidelines   # ※notice when debug
    if args.dataset == 'CONLL':
        seed_guidelines = ['' for _ in labels]
    else:
        seed_guidelines = ['' for _ in labels]
    if args.dataset == 'CONLL' and args.start_step == 1 and args.model_name == 'demo_model':
        last_step_guides = ['' for _ in labels]
    else:
        last_step_guides = ['' for _ in labels]
    if args.start_step == 0:
        last_step_guides = seed_guidelines
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    for step in range(args.start_step, args.iterations):
        print(f'# Step {step} # # # # # # # # #  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
        print('----------------------------- generate current step predictions ----------------------------')
        if PROMPT == '':
            with open(prompt_path, mode='r', encoding='utf-8') as f:
                prompts = json.load(f)
            PROMPT = str(prompts[f'step{step}'])
        pred_path = f'predictions/{args.model_name}/{args.dataset}/step{step}.jsonl'
        if step == args.start_step and args.start_sample_idx != 0:  # for debug
            current_predictions = generate_predictions(PROMPT, texts[args.start_sample_idx:], llm,
                                                       batch_size=args.batch_size, pred_path=pred_path, labels=labels)
            all_predictions = []
            with open(pred_path, mode='r', encoding='utf-8') as f:
                for line in f:
                    try:
                        all_predictions.append(json.loads(line))
                    except:
                        import pdb;
                        pdb.set_trace()
            current_predictions = all_predictions
        else:
            current_predictions = generate_predictions(PROMPT, texts, llm, batch_size=args.batch_size,
                                                       pred_path=pred_path, labels=labels)

        eval_res = evaluation(gold_samples, current_predictions, labels)
        with open(evaluation_path, mode='a', encoding='utf-8') as fe:
            fe.write(
                f'Step{step}    : precision={eval_res["precision"]},  recall={eval_res["recall"]},  f1={eval_res["f1"]},  gold={eval_res["gold"]}, pred={eval_res["pred"]}, correct={eval_res["correct"]}\n')

        print('----------------------------- post process predictions ----------------------------')
        if step + 1 > 2:
            historical_predictions = []
            for i in range(0, step):
                tmp_path = f'predictions/{args.model_name}/{args.dataset}/step{i}.jsonl'
                tmp_predictions = []
                with open(tmp_path, mode='r', encoding='utf-8') as f:
                    for line in f:
                        tmp_predictions.append(json.loads(line))
                historical_predictions.append(tmp_predictions)
            historical_predictions.append(current_predictions)
            historical_pred_nums = []
            for predictions in historical_predictions:
                pred_nums = count_pred_entities(predictions)
                historical_pred_nums.append(pred_nums)
            processed_predictions = post_process_predictions(args, historical_predictions, historical_pred_nums)

            eval_res = evaluation(gold_samples, processed_predictions, labels)
            with open(evaluation_path, mode='a', encoding='utf-8') as fe:
                fe.write(
                    f'Processed: precision={eval_res["precision"]},  recall={eval_res["recall"]},  f1={eval_res["f1"]},  gold={eval_res["gold"]}, pred={eval_res["pred"]}, correct={eval_res["correct"]}\n')
        else:
            processed_predictions = current_predictions

        print('----------------------------- split predictions by types ----------------------------')
        predictions_split = split_by_types(processed_predictions, texts, labels)

        print('----------------------------- select typical samples ----------------------------')
        typical_num = 8
        typical_sample_ids_subset, typical_sample_ids_whole = [], []
        for label in tqdm(labels, desc='Select typical samples '):
            ids_subset, ids_whole = select_typical_samples(label, typical_num, predictions_split[label])
            typical_sample_ids_subset.append(ids_subset)
            typical_sample_ids_whole.append(ids_whole)

        print('----------------------------- generate 敏感词类型定义 ----------------------------')
        guideline_definitions = []
        for t_id, label in tqdm(enumerate(labels), total=len(labels)):
            typical_samples = [predictions_split[label][i] for i in typical_sample_ids_subset[t_id]]
            definition = generate_guideline_definition(label, typical_samples, llm, last_step_guides[t_id])
            guideline_definitions.append(definition)
        last_step_guides = guideline_definitions
        print('----------------------------- update and save new prompts ----------------------------')
        few_shot_samples = []
        next_step = step + 1
        if next_step > 2:
            num_per_type = args.shots_per_type
            for t_id in range(len(labels)):
                if len(typical_sample_ids_whole[t_id]) > 0:
                    sentences = []
                    for idx in typical_sample_ids_whole[t_id]:
                        sentences.append(texts[idx])
                    center_ids_subset = get_cluster_center_ids(sentences, embedding_model, num_per_type)
                    for idx_subset in center_ids_subset:
                        idx = typical_sample_ids_whole[t_id][idx_subset]  # idx in the whole_set
                        text, answer = texts[idx], json.dumps(processed_predictions[idx], ensure_ascii=False)
                        sample = f'Text: {text}\nAnswer: {answer}'
                        few_shot_samples.append(sample)
        else:
            for obj in gold_samples:
                if len(obj["label"]) > 0:
                    text = obj["text"]
                    tmp = {}
                    for pair in obj["label"]:
                        tmp[pair[0]] = pair[1]
                    answer = json.dumps(tmp, ensure_ascii=False)
                    break
            sample = f'Text: {text}\nAnswer: {answer}'
            few_shot_samples.append(sample)
        PROMPT = save_prompt(guideline_definitions, few_shot_samples, labels, prompt_path,
                             next_step)


if __name__ == '__main__':
    pass

    main()
