# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

from torch.utils.data import TensorDataset
import numpy as np
import os
import torch
import json
from .dataset import COCODetectionDataset, CityscapesSegmentationDataset, TrafficSignClassificationDataset
import glob


def read_data(dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join('../dataset', dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join('../dataset', dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, idx, is_train=True):
    if "detection" in dataset:
        split = "train" if is_train else "test"
        root = f"/home/zhengbk/PFLoRA-lib/dataset/{dataset}"
        full_dataset = COCODetectionDataset(
            root=root,
            split=split
        )

    elif "segmentation" in dataset:
        split = "train" if is_train else "validation"
        root = f"/home/zhengbk/PFLoRA-lib/dataset/{dataset}"

        parquet_files = sorted(
            glob.glob(f"{root}/{split}-*.parquet")
        )

        full_dataset = CityscapesSegmentationDataset(
            parquet_files=parquet_files
        )

    elif "classification" in dataset:
        split = "train" if is_train else "test"
        root = f"/home/zhengbk/PFLoRA-lib/dataset/{dataset}"

        full_dataset = TrafficSignClassificationDataset(
            root=root,
            split=split
        )

    print(f"[{dataset}] {split} samples:", len(full_dataset))

    return full_dataset
    # if "News" in dataset:
    #     return read_client_data_text(dataset, idx, is_train)
    # elif "Shakespeare" in dataset:
    #     return read_client_data_Shakespeare(dataset, idx)

    # elif "Seq" in dataset:
    #     # return read_client_data_glue(dataset, idx, is_train)
    #     return read_client_data_sst2(dataset, idx, is_train)
    # elif "Token" in dataset:
    #     return read_client_data_conll(dataset, idx, is_train)
    #     # return read_client_data_ratener(dataset, idx, is_train)
    #     # return read_client_data_wiki(dataset, idx, is_train)
    # elif "Choice" in dataset:
    #     # return read_client_data_swag(dataset, idx, is_train)
    #     # return read_client_data_copa(dataset, idx, is_train)
    #     return read_client_data_sct(dataset, idx, is_train)


    # if is_train:
    #     train_data = read_data(dataset, idx, is_train)
    #     X_train = torch.Tensor(train_data['x']).type(torch.float32)
    #     y_train = torch.Tensor(train_data['y']).type(torch.int64)

    #     train_data = [(x, y) for x, y in zip(X_train, y_train)]
    #     return train_data
    # else:
    #     test_data = read_data(dataset, idx, is_train)
    #     X_test = torch.Tensor(test_data['x']).type(torch.float32)
    #     y_test = torch.Tensor(test_data['y']).type(torch.int64)
    #     test_data = [(x, y) for x, y in zip(X_test, y_test)]
    #     return test_data


def read_client_data_text(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [((x, lens), y)
                      for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [((x, lens), y)
                     for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data


def read_client_data_Shakespeare(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def read_glue_data(dataset, idx, is_train=True):
    data_dir = os.path.join('../dataset', dataset,
                            'train' if is_train else 'test')
    file_path = os.path.join(data_dir, f'{idx}.npz')
    with open(file_path, 'rb') as f:
        data = np.load(f, allow_pickle=True)
        return {key: data[key] for key in data}


def read_client_data_glue(dataset, idx, is_train=True):
    data = read_glue_data(dataset, idx, is_train)
    input_ids = torch.tensor(data['input_ids'], dtype=torch.long)
    attention_mask = torch.tensor(data['attention_mask'], dtype=torch.long)
    labels = torch.tensor(data['labels'], dtype=torch.long)
    return TensorDataset(input_ids, attention_mask, labels)


from transformers import BertTokenizer
import pandas as pd


def read_sst2_data(data_dir, filename):
    # 读取SST-2数据集的TSV文件
    file_path = os.path.join(data_dir, filename)
    data = pd.read_csv(file_path, delimiter='\t')  # 使用tab作为分隔符
    return data


def preprocess_conll_data(data, tokenizer, label2id, max_length=512, save_path=None):
    # 使用tokenizer进行文本的编码处理
    input_ids = []
    attention_masks = []
    labels = []

    for words, label_seq in data:
        # 使用BERT tokenizer将文本转换为input_ids和attention_mask
        encoding = tokenizer(words, is_split_into_words=True, padding='max_length', 
                             truncation=True, max_length=max_length, 
                             return_attention_mask=True, return_tensors='pt')

        # 处理标签
        label_ids = [label2id[label] for label in label_seq]
        
        # 处理不匹配的长度
        while len(label_ids) < len(encoding['input_ids'][0]):
            label_ids.append(label2id['O'])  # 使用'O'作为默认标签

        # 保存编码后的数据
        input_ids.append(encoding['input_ids'].squeeze(0))  # [seq_len, ]
        attention_masks.append(encoding['attention_mask'].squeeze(0))  # [seq_len, ]
        labels.append(label_ids)  # [seq_len, ]

    # 转换为tensor
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.tensor(labels, dtype=torch.long)

    if save_path:
        # 保存数据为 .npz 文件
        data_dict = {
            'input_ids': input_ids.numpy(),
            'attention_mask': attention_masks.numpy(),
            'labels': labels.numpy()
        }
        save_data_as_npz(data_dict, save_path)

    return TensorDataset(input_ids, attention_masks, labels)


def preprocess_sst2_data(data, tokenizer, max_length=512):
    # 使用tokenizer进行文本的编码处理
    input_ids = []
    attention_masks = []
    labels = []

    for _, row in data.iterrows():
        sentence = row['sentence']
        label = row['label']
        
        # 使用BERT tokenizer将文本转换为input_ids和attention_mask
        encoding = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,  # 添加[CLS]和[SEP]符号
            padding='max_length',     # 填充到最大长度
            truncation=True,          # 截断超出最大长度的文本
            max_length=max_length,    # 最大长度
            return_attention_mask=True,
            return_tensors='pt'       # 返回PyTorch tensors
        )

        input_ids.append(encoding['input_ids'].squeeze(0))  # 由于返回的input_ids是二维的，我们取掉第一维
        attention_masks.append(encoding['attention_mask'].squeeze(0))
        labels.append(label)

    # 转换为tensor
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.tensor(labels, dtype=torch.long)
    save_path = '../GLUE-baselines/glue_data/SST-2/dev.npz'
    # 保存数据为 npz 文件
    data_dict = {
        'input_ids': input_ids.numpy(),
        'attention_mask': attention_masks.numpy(),
        'labels': labels.numpy()
    }
    save_data_as_npz(data_dict, save_path)

    return TensorDataset(input_ids, attention_masks, labels)

def preprocess_swag_data(file_path, tokenizer, max_length=512, save_path=None):
    # 读取 parquet 数据
    df = pd.read_parquet(file_path)

    input_ids = []
    attention_masks = []
    labels = []

    for idx, row in df.iterrows():
        # print(row)
        sent1 = row["sent1"]  # 上下文
        sent2 = row["sent2"]  # 前缀（可拼接）
        
        label = int(row["label"])  # 正确答案索引
        assert 0 <= label < 4, f"Invalid label: {label} at index {idx}"
        # 生成 4 个候选句子
        choices = [row[f"ending{i}"] for i in range(4)]
        contexts = [f"{sent1} {sent2}"] * 4  # 复制 4 次上下文

        # 使用 tokenizer 处理
        encoding = tokenizer(
            contexts, choices, padding="max_length", truncation=True,
            max_length=max_length, return_attention_mask=True, return_tensors="pt"
        )

        input_ids.append(encoding["input_ids"])  # [4, max_length]
        attention_masks.append(encoding["attention_mask"])  # [4, max_length]
        labels.append(label)  # 单个整数标签

    # 转换为 Tensor
    input_ids = torch.stack(input_ids)  # [num_samples, 4, max_length]
    attention_masks = torch.stack(attention_masks)  # [num_samples, 4, max_length]
    labels = torch.tensor(labels, dtype=torch.long)  # [num_samples]

    if save_path:
        data_dict = {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_masks.numpy(),
            "labels": labels.numpy()
        }
    save_data_as_npz(data_dict, save_path)

    return TensorDataset(input_ids, attention_masks, labels)


def save_data_as_npz(data, save_path):
    # 将数据保存为 .npz 文件
    np.savez(save_path,
             input_ids=data['input_ids'],
             attention_mask=data['attention_mask'],
             labels=data['labels'])

# def read_client_data_swag(dataset, idx, is_train=True):
#     data_dir = os.path.join('../dataset/SWAG')
#     # file_type = 'test.parquet'
#     file_type = 'train.parquet' if is_train else 'validation.parquet'
#     file_path = os.path.join(data_dir, file_type)
#     # 调用 preprocessor 将数据编码并保存为 npz 文件
#     tokenizer =  BertTokenizer.from_pretrained("/home/zhengbk/PFLoRA-lib/bert_base_uncased")
#     save_path = f"../dataset/SWAG/preprocessed_train.npz" if is_train else f"../dataset/SWAG/preprocessed_test.npz"
#     data = preprocess_swag_data(file_path, tokenizer, save_path=save_path)
#     return data


def read_client_data_swag(dataset, idx, is_train=True):
    data_dir = os.path.join('../dataset/SWAG')
    file_type = 'preprocessed_train.npz' if is_train else 'preprocessed_test.npz'
    file_path = os.path.join(data_dir, file_type)
    return load_data_from_npz(file_path)


import xml.etree.ElementTree as ET

def preprocess_copa_data(file_path, tokenizer, max_length=512, save_path=None):
    # 解析 XML 文件
    tree = ET.parse(file_path)
    root = tree.getroot()
    input_ids = []
    attention_masks = []
    labels = []

    for item in root.findall('item'):
        premise = item.find('p').text
        choice1 = item.find('a1').text
        choice2 = item.find('a2').text
        label = int(item.get('most-plausible-alternative')) - 1  # 转换为 0/1
        question_type = item.get('asks-for')  # "cause" 或 "effect"

        # 生成 2 个候选句子
        choices = [choice1, choice2]
        contexts = [premise] * 2  # 复制 2 次 premise

        # 使用 tokenizer 处理
        encoding = tokenizer(
            contexts, choices, padding="max_length", truncation=True,
            max_length=max_length, return_attention_mask=True, return_tensors="pt"
        )

        input_ids.append(encoding["input_ids"])  # [2, max_length]
        attention_masks.append(encoding["attention_mask"])  # [2, max_length]
        labels.append(label)  # 单个整数标签
    
    # 转换为 Tensor
    input_ids = torch.stack(input_ids)  # [num_samples, 2, max_length]
    attention_masks = torch.stack(attention_masks)  # [num_samples, 2, max_length]
    labels = torch.tensor(labels, dtype=torch.long)  # [num_samples]

    if save_path:
        data_dict = {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_masks.numpy(),
            "labels": labels.numpy()
        }
    save_data_as_npz(data_dict, save_path)

    return TensorDataset(input_ids, attention_masks, labels)

def preprocess_sct_data(file_path, tokenizer, max_length=512, save_path=None):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    input_ids = []
    attention_masks = []
    labels = []
    
    for _, row in df.iterrows():
        # 构建上下文：将前4个句子合并
        context = " ".join([
            row['InputSentence1'],
            row['InputSentence2'],
            row['InputSentence3'],
            row['InputSentence4']
        ])
        
        # 获取两个候选结尾
        choice1 = row['RandomFifthSentenceQuiz1']
        choice2 = row['RandomFifthSentenceQuiz2']
        choices = [choice1, choice2]
        
        # 获取正确标签 (1或2转换为0或1)
        label = int(row['AnswerRightEnding']) - 1
        
        # 复制上下文以匹配两个选择
        contexts = [context] * 2
        
        # 使用tokenizer处理
        encoding = tokenizer(
            contexts, choices, padding="max_length", truncation=True,
            max_length=max_length, return_attention_mask=True, return_tensors="pt"
        )
        
        input_ids.append(encoding["input_ids"])  # [2, max_length]
        attention_masks.append(encoding["attention_mask"])  # [2, max_length]
        labels.append(label)  # 单个整数标签
    
    # 转换为Tensor
    input_ids = torch.stack(input_ids)  # [num_samples, 2, max_length]
    attention_masks = torch.stack(attention_masks)  # [num_samples, 2, max_length]
    labels = torch.tensor(labels, dtype=torch.long)  # [num_samples]
    
    if save_path:
        data_dict = {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_masks.numpy(),
            "labels": labels.numpy()
        }
        save_data_as_npz(data_dict, save_path)
    
    return TensorDataset(input_ids, attention_masks, labels)


def read_client_data_sct(dataset, idx, is_train=True):
    data_dir = os.path.join('../dataset/SCT')
    # file_type = 'train.csv' if is_train else 'test.csv'
    # file_path = os.path.join(data_dir, file_type)
    # # 调用 preprocessor 将数据编码并保存为 npz 文件
    # tokenizer =  BertTokenizer.from_pretrained("/home/zhengbk/PFLoRA-lib/bert_base_uncased")
    # save_path = f"../dataset/SCT/preprocessed_train.npz" if is_train else f"../dataset/SCT/preprocessed_test.npz"
    # data = preprocess_sct_data(file_path, tokenizer, save_path=save_path)
    # return data
    file_type = 'preprocessed_train.npz' if is_train else 'preprocessed_test.npz'
    file_path = os.path.join(data_dir, file_type)
    return load_data_from_npz(file_path)

def read_client_data_copa(dataset, idx, is_train=True):
    data_dir = os.path.join('../dataset/COPA')
    # file_type = 'copa-dev.xml' if is_train else 'copa-test.xml'
    # file_path = os.path.join(data_dir, file_type)
    # # 调用 preprocessor 将数据编码并保存为 npz 文件
    # tokenizer =  BertTokenizer.from_pretrained("/home/zhengbk/PFLoRA-lib/bert_base_uncased")
    # save_path = f"../dataset/COPA/preprocessed_train.npz" if is_train else f"../dataset/COPA/preprocessed_test.npz"
    # data = preprocess_copa_data(file_path, tokenizer, save_path=save_path)
    # return data
    file_type = 'preprocessed_train.npz' if is_train else 'preprocessed_test.npz'
    file_path = os.path.join(data_dir, file_type)
    return load_data_from_npz(file_path)


def read_client_data_sst2(dataset, idx, is_train=True):
    # 根据是否是训练集选择文件
    data_dir = os.path.join('../GLUE-baselines/glue_data/SST-2')
    file_type = 'train_data.npz' if is_train else 'dev.npz'  # 'dev.tsv' 作为验证集
    # data = read_sst2_data(data_dir, file_type)
    # tokenizer =  BertTokenizer.from_pretrained("/home/zhengbk/PFLoRA-lib/bert_base_uncased")

    # 对数据进行预处理，转换为input_ids, attention_mask, labels
    # return preprocess_sst2_data(data, tokenizer)
    file_path = os.path.join(data_dir, file_type)
    return load_data_from_npz(file_path)

def read_client_data_wiki(dataset, idx, is_train=True):
    # if is_train:
    #     data_file = f"../dataset/Token/wiki/train.conllu"
    # else:
    #     data_file = f"../dataset/Token/wiki/test.conllu"
    
    # label2id = {
    #     'O': 0,         # 非命名实体
    #     'B-PER': 1,     # 人名（开始）
    #     'I-PER': 2,     # 人名（内部）
    #     'B-ORG': 3,     # 组织名（开始）
    #     'I-ORG': 4,     # 组织名（内部）
    #     'B-LOC': 5,     # 地点名（开始）
    #     'I-LOC': 6,     # 地点名（内部）
    #     'B-MISC': 7,    # 其他类别（开始）
    #     'I-MISC': 8     # 其他类别（内部）
    # }

    # data = []
    # sentences = []
    # labels = []
    
    # # 读取数据文件
    # with open(data_file, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         line = line.strip()
    #         if line == '':
    #             if sentences:
    #                 # 当遇到空行时，认为是句子的结束
    #                 data.append((sentences, labels))
    #                 sentences = []
    #                 labels = []
    #         else:
    #             parts = line.split('\t')
    #             word = parts[1]  # 单词位于第二列
    #             label = parts[-1]  # 标签位于第四列
                
    #             sentences.append(word)
    #             labels.append(label)
        
    #     # 处理文件结尾的情况
    #     if sentences:
    #         data.append((sentences, labels))

    # # 调用 preprocessor 将数据编码并保存为 npz 文件
    # tokenizer =  BertTokenizer.from_pretrained("/home/zhengbk/PFLoRA-lib/bert_base_uncased")
    # save_path = f"../dataset/Token/wiki/preprocessed_train.npz" if is_train else f"../dataset/Token/wiki/preprocessed_test.npz"
    # tokenized_data = preprocess_conll_data(data, tokenizer, label2id, max_length=512, save_path=save_path)
    file_path = "../dataset/Token/wiki/preprocessed_train.npz" if is_train else "../dataset/Token/wiki/preprocessed_test.npz"
    tokenized_data = load_data_from_npz(file_path)

    return tokenized_data


def read_client_data_conll(dataset, idx, is_train=True):
    # if is_train:
    #     data_file = f"../dataset/Token/conll2023/train.txt"
    # else:
    #     data_file = f"../dataset/Token/conll2023/test.txt"

    # data = []
    # sentences = []
    # labels = []
    # label2id = {
    #     'O': 0,         # 非命名实体
    #     'B-PER': 1,     # 人名（开始）
    #     'I-PER': 2,     # 人名（内部）
    #     'B-ORG': 3,     # 组织名（开始）
    #     'I-ORG': 4,     # 组织名（内部）
    #     'B-LOC': 5,     # 地点名（开始）
    #     'I-LOC': 6,     # 地点名（内部）
    #     'B-MISC': 7,    # 其他类别（开始）
    #     'I-MISC': 8     # 其他类别（内部）
    # }
    # # 读取数据文件
    # with open(data_file, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         line = line.strip()
    #         if line == '':
    #             if sentences:
    #                 data.append((sentences, labels))
    #                 sentences = []
    #                 labels = []
    #         elif line.startswith('-DOCSTART-'):
    #             continue  # Skip document start lines
    #         else:
    #             parts = line.split()
    #             word = parts[0]
    #             label = parts[-1]  # 标签在最后一列
                
    #             sentences.append(word)
    #             labels.append(label)
        
    #     # 处理文件结尾的情况
    #     if sentences:
    #         data.append((sentences, labels))
    # tokenizer =  BertTokenizer.from_pretrained("/home/zhengbk/PFLoRA-lib/bert_base_uncased")
    # # 调用 preprocessor 将数据编码并保存为 npz 文件
    # save_path = "../dataset/Token/conll2023/preprocessed_train.npz" if is_train else "../dataset/Token/conll2023/preprocessed_test.npz"
    # tokenized_data = preprocess_conll_data(data, tokenizer, label2id, max_length=512, save_path=save_path)
    file_path = "../dataset/Token/conll2023/preprocessed_train.npz" if is_train else "../dataset/Token/conll2023/preprocessed_test.npz"
    # file_path = "../dataset/Token/raten/preprocessed_train.npz" if is_train else "../dataset/Token/raten/preprocessed_test.npz"
    tokenized_data = load_data_from_npz(file_path)

    return tokenized_data


def load_data_from_npz(file_path):
    # 加载保存的 .npz 文件
    with np.load(file_path) as data:
        input_ids = torch.tensor(data['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(data['attention_mask'], dtype=torch.long)
        labels = torch.tensor(data['labels'], dtype=torch.long)
    return TensorDataset(input_ids, attention_mask, labels)



def read_client_data_ratener(dataset, idx, is_train=True):
    # 加载标签映射（需要根据您的数据定义）
    label2id = {
        "0": "B-ABNORMALITY", 
        "1": "I-ABNORMALITY", 
        "2": "B-NON-ABNORMALITY", 
        "3": "I-NON-ABNORMALITY", 
        "4": "B-DISEASE", 
        "5": "I-DISEASE", 
        "6": "B-NON-DISEASE", 
        "7": "I-NON-DISEASE", 
        "8": "B-ANATOMY", 
        "9": "I-ANATOMY", 
        "10": "O"
    }

    
    # 读取JSON文件
    data_type = "train" if is_train else "test"
    json_path = f"../dataset/Token/raten/{data_type}.json"
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 调用预处理函数
    save_path = f"../dataset/Token/raten/preprocessed_{data_type}.npz"
    tokenized_data = preprocess_ratener_data(
        data=data,
        tokenizer=BertTokenizer.from_pretrained("/home/zhengbk/PFLoRA-lib/bert_base_uncased"),
        label2id=label2id,
        max_length=512,
        save_path=save_path
    )
    
    return tokenized_data


def preprocess_ratener_data(data, tokenizer, label2id, max_length=512, save_path=None):
    input_ids = []
    attention_masks = []
    labels = []

    for example in data:
        words = example["tokens"]
        label_seq = example["ner_tags"]

        # 使用BERT tokenizer将文本转换为input_ids和attention_mask
        encoding = tokenizer(words, is_split_into_words=True, padding='max_length', 
                             truncation=True, max_length=max_length, 
                             return_attention_mask=True, return_tensors='pt')

        # 处理标签
        # label_ids = [label2id[label] for label in label_seq]
        
        # 处理不匹配的长度
        while len(label_seq) < len(encoding['input_ids'][0]):
            label_seq.append(10)  # 使用'O'作为默认标签
    
        
        input_ids.append(encoding['input_ids'].squeeze(0))  # [seq_len, ]
        attention_masks.append(encoding['attention_mask'].squeeze(0))  # [seq_len, ]
        labels.append(torch.tensor(label_seq, dtype=torch.long))  # [seq_len, ]
    
    # 转换为tensor
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.stack(labels)
    
    if save_path:
        data_dict = {
            'input_ids': input_ids.numpy(),
            'attention_mask': attention_masks.numpy(),
            'labels': labels.numpy()
        }
        save_data_as_npz(data_dict, save_path)
    
    return TensorDataset(input_ids, attention_masks, labels)

