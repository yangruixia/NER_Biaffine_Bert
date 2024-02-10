import os
import sys
sys.path.append('./')
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
#from utils.arguments_parse import args
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import unicodedata, re

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

def get_tokenizer():
    """添加特殊中文字符和未使用的token【unused1】"""
    added_token=['[unused1]','[unused1]']
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path,additional_special_tokens=added_token)
    special_tokens_dict = {'additional_special_tokens':['”','“']}
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer

tokenizer=get_tokenizer()

class token_rematch:
    def __init__(self):
        self._do_lower_case = True


    @staticmethod
    def stem(token):
            """获取token的“词干”（如果是##开头，则自动去掉##）
            """
            if token[:2] == '##':
                return token[2:]
            else:
                return token
    @staticmethod
    def _is_control(ch):
            """控制类字符判断
            """
            return unicodedata.category(ch) in ('Cc', 'Cf')
    @staticmethod
    def _is_special(ch):
            """判断是不是有特殊含义的符号
            """
            return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    def rematch(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系
        """
        if self._do_lower_case:
            text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


def load_schema():
    # 读取schema
    
    label2id={'nr':0,'ns':1,'nt':2}
    id2label={0:'nr',1:'ns',2:'nt'}
    num_labels=3

    return label2id,id2label,num_labels

if __name__=='__main__':

    schema=load_schema()
    print(len(schema))
    # cl=token_rematch()
    # d=cl.rematch(s,c)
    # print(d)

