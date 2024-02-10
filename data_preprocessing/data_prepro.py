import os
import sys
sys.path.append('./')
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
##from utils.arguments_parse import args
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import unicodedata, re
from data_preprocessing import tools
from tqdm import tqdm

tokenizer=tools.get_tokenizer()
label2id,id2label,num_labels = tools.load_schema()

# -*- coding:utf-8 -*-
# @Time  : 2020/11/3 14:22
# @Author: yangping

import argparse

parser = argparse.ArgumentParser(description="train")
parser.add_argument("--train_path", type=str, default="./data/train.json",help="train file")
parser.add_argument("--test_path", type=str, default="./data/test.json",help="test file")
parser.add_argument("--schema_path", type=str, default="./event_schema/event_schema.json",help="schema")
parser.add_argument("--checkpoints", type=str, default="./checkpoints/multilabel_cls.pth",help="output_dir")
parser.add_argument("--bert_mrc_checkpoints", type=str, default="./checkpoints/bert_mrc.pth",help="output_dir")
parser.add_argument("--vocab_file", type=str, default="./data/vocab.txt",help="vocab_file")
parser.add_argument("--tag_file", type=str, default="./data/tags.txt",help="tag_file")
parser.add_argument("--batch_size", type=int, default=8,help="batch_size")
parser.add_argument("--hidden_num", type=int, default=512,help="hidden_num")
parser.add_argument("--max_length", type=int, default=128,help="max_length")
parser.add_argument("--embedding_file", type=str, default=None,help="embedding_file")
parser.add_argument("--epoch", type=int, default=400,help="epoch")
parser.add_argument("--learning_rate", type=float, default=1e-4,help="learning_rate")
parser.add_argument("--require_improvement", type=int, default=100,help="require_improvement")
parser.add_argument("--pretrained_model_path", type=str, default="./pretrained_model/chinese_roberta_wwm_ext",help="pretrained_model_path")
parser.add_argument("--clip_norm", type=str, default=0.25,help="clip_norm")
parser.add_argument("--warm_up_epoch", type=str, default=1,help="warm_up_steps")
parser.add_argument("--decay_epoch", type=str, default=80,help="decay_steps")
parser.add_argument("--output", type=str, default="./output/result.json",help="output")

args = parser.parse_args()


def load_data(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        sentences = []
        arguments = []
        for line in lines:
            data = json.loads(line)
            text = data['text']
            entity_list = data['entity_list']
            args_dict={}
            if entity_list != []:
                for entity in entity_list:
                    entity_type = entity['type']
                    entity_argument=entity['argument']
                    if entity_type not in args_dict.keys():
                        args_dict[entity_type] = [entity_argument]
                    else:
                        args_dict[entity_type].append(entity_argument)
                sentences.append(text)
                arguments.append(args_dict)
        return sentences, arguments


def encoder(sentence, argument,args):
    encode_dict = tokenizer.encode_plus(sentence,
                                        max_length=args.max_length,
                                        pad_to_max_length=True)
    encode_sent = encode_dict['input_ids']
    token_type_ids = encode_dict['token_type_ids']
    attention_mask = encode_dict['attention_mask']

    zero = [0 for i in range(args.max_length)]
    span_mask=[ attention_mask for i in range(sum(attention_mask))]
    span_mask.extend([ zero for i in range(sum(attention_mask),args.max_length)])
    
    span_label = [0 for i in range(args.max_length)]
    span_label = [span_label for i in range(args.max_length)]
    span_label = np.array(span_label)
    for entity_type,args in argument.items():
        for arg in args:
            encode_arg = tokenizer.encode(arg)
            start_idx = tools.search(encode_arg[1:-1], encode_sent)
            end_idx = start_idx + len(encode_arg[1:-1]) - 1
            span_label[start_idx, end_idx] = label2id[entity_type]+1

    return encode_sent, token_type_ids, attention_mask, span_label, span_mask



def data_pre(file_path):
    sentences, arguments = load_data(file_path)
    data = []
    for i in tqdm(range(len(sentences))):
        encode_sent, token_type_ids, attention_mask, span_label, span_mask = encoder(
            sentences[i], arguments[i], args  # Pass args here
        )
        tmp = {
            'input_ids': encode_sent,
            'input_seg': token_type_ids,
            'input_mask': attention_mask,
            'span_label': span_label,
            'span_mask': span_mask
        }
        data.append(tmp)

    return data



class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        one_data = {
            "input_ids": torch.tensor(item['input_ids']).long(),
            "input_seg": torch.tensor(item['input_seg']).long(),
            "input_mask": torch.tensor(item['input_mask']).float(),
            "span_label": torch.tensor(item['span_label']).long(),
            "span_mask": torch.tensor(item['span_mask']).long()
        }
        return one_data

def yield_data(file_path):
    tmp = MyDataset(data_pre(file_path))
    return DataLoader(tmp, batch_size=args.batch_size, shuffle=True)


if __name__ == '__main__':
    # Create DataLoader
    data = data_pre(args.train_path)


    print(data[0]['span_label'])

    print(data[0]['span_mask'])



