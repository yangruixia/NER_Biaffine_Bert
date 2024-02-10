import os
import sys
from typing import Any
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
from torch import optim
import numpy as np
#from utils.arguments_parse import args
from data_preprocessing import data_prepro
from model.model import myModel
from model.loss_function import cross_entropy_loss
from model.loss_function import span_loss
from model.loss_function import focal_loss
from model.metrics import metrics
from utils.logger import logger

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

device = torch.device('cuda')
sentences,_=data_prepro.load_data(args.train_path)
train_data_length=len(sentences)


class WarmUp_LinearDecay:
    def __init__(self, optimizer: optim.AdamW, init_rate, warm_up_epoch, decay_epoch, min_lr_rate=1e-8):
        self.optimizer = optimizer
        self.init_rate = init_rate
        self.epoch_step = train_data_length / args.batch_size
        self.warm_up_steps = self.epoch_step * warm_up_epoch
        self.decay_steps = self.epoch_step * decay_epoch
        self.min_lr_rate = min_lr_rate
        self.optimizer_step = 0
        self.all_steps = args.epoch*(train_data_length/args.batch_size)

    def step(self):
        self.optimizer_step += 1
        if self.optimizer_step <= self.warm_up_steps:
            rate = (self.optimizer_step / self.warm_up_steps) * self.init_rate
        elif self.warm_up_steps < self.optimizer_step <= self.decay_steps:
            rate = self.init_rate
        else:
            rate = (1.0 - ((self.optimizer_step - self.decay_steps) / (self.all_steps-self.decay_steps))) * self.init_rate
            if rate < self.min_lr_rate:
                rate = self.min_lr_rate
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self.optimizer.step()

def train():
    train_data = data_prepro.yield_data(args.train_path)
    test_data = data_prepro.yield_data(args.test_path)

    model = myModel(pre_train_dir=args.pretrained_model_path, dropout_rate=0.5).to(device)
    # model.load_state_dict(torch.load(args.checkpoints))
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
    ]
    optimizer = optim.AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate)

    schedule = WarmUp_LinearDecay(
                optimizer = optimizer, 
                init_rate = args.learning_rate,
                warm_up_epoch = args.warm_up_epoch,
                decay_epoch = args.decay_epoch
            )
    
    span_loss_func = span_loss.Span_loss().to(device)
    span_acc = metrics.metrics_span().to(device)

    step=0
    best=0
    for epoch in range(args.epoch):
        for item in train_data:
            step+=1
            input_ids, input_mask, input_seg = item["input_ids"], item["input_mask"], item["input_seg"]
            span_label,span_mask = item['span_label'],item["span_mask"]
            optimizer.zero_grad()
            span_logits = model( 
                input_ids=input_ids.to(device), 
                input_mask=input_mask.to(device),
                input_seg=input_seg.to(device),
                is_training=True
            )
            span_loss_v = span_loss_func(span_logits,span_label.to(device),span_mask.to(device))
            loss = span_loss_v
            loss = loss.float().mean().type_as(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
            schedule.step()
            # optimizer.step()
            if step%100 == 0:
                span_logits = torch.nn.functional.softmax(span_logits, dim=-1)
                recall,precise,span_f1=span_acc(span_logits,span_label.to(device))
                logger.info('epoch %d, step %d, loss %.4f, recall %.4f, precise %.4f, span_f1 %.4f'% (epoch,step,loss,recall,precise,span_f1))
        with torch.no_grad():
            count=0
            span_f1=0
            recall=0
            precise=0

            for item in test_data:
                count+=1
                input_ids, input_mask, input_seg = item["input_ids"], item["input_mask"], item["input_seg"]
                span_label,span_mask = item['span_label'],item["span_mask"]

                optimizer.zero_grad()
                span_logits = model( 
                    input_ids=input_ids.to(device), 
                    input_mask=input_mask.to(device),
                    input_seg=input_seg.to(device),
                    is_training=False
                    ) 
                tmp_recall,tmp_precise,tmp_span_f1=span_acc(span_logits,span_label.to(device))
                span_f1+=tmp_span_f1
                recall+=tmp_recall
                precise+=tmp_precise
                
            span_f1 = span_f1/count
            recall=recall/count
            precise=precise/count

            logger.info('-----eval----')
            logger.info('epoch %d, step %d, loss %.4f, recall %.4f, precise %.4f, span_f1 %.4f'% (epoch,step,loss,recall,precise,span_f1))
            logger.info('-----eval----')
            if best < span_f1:
                best=span_f1
                torch.save(model.state_dict(), f=args.checkpoints)
                logger.info('-----save the best model----')

        
if __name__=='__main__':
    train()