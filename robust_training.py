import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import pickle
import os

from src.attack.text_grad import PGDAttack
from src.substitution.bert_sub import BertSubstitutor
from src.models.bert_model import BertVictimModel
from transformers import BertTokenizer, BertForSequenceClassification, BertForMaskedLM
from src.attack.context import ctx_noparamgrad
from src.data_util.dataloader import (LocalSSTDataset, 
                                        remove_unused_columns,
                                        get_class_num, get_task_type
                                        )

import time
import tqdm

MODEL_CACHE_DIR = './model_cache/'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--norm', action = 'store_true')
parser.add_argument('--ste', action = 'store_true')
parser.add_argument('--rand', action = 'store_true')
parser.add_argument('--no_subword', action = 'store_true')
parser.add_argument('--multi_sample', action = 'store_true')
parser.add_argument('--use_lm', action = 'store_true')
parser.add_argument('--cw', action = 'store_true')
parser.add_argument('--use_cache', action = 'store_true')

parser.add_argument('--eta_z', type = float, default = 0.8)
parser.add_argument('--eta_u', type = float, default = 0.8)
parser.add_argument('--iter_time', type = int, default = 5)
parser.add_argument('--sample_num', type = int, default = 20)
parser.add_argument('--final_sample', type = int, default = 20)
parser.add_argument('--modif', type = float, default = 0.25)
parser.add_argument('--patience', type = int, default = 1)
parser.add_argument('--lm_beta', type = float, default = 0.1)

parser.add_argument('--trades', action = 'store_true')

parser.add_argument("--victim", type = str, default = 'bert')
parser.add_argument("--dataset", type = str, default = 'sst')
parser.add_argument('--trades_beta', type = float, default = 1.0)

args = parser.parse_args()

batch_size = 16
epoch_num = 5
use_trades = args.trades
trades_beta = args.trades_beta
criterion_kl = nn.KLDivLoss(reduction = 'mean')

def evaluate_accuracy(model, dataloader):
    total = 0
    correct = 0
    for idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids'] if 'token_type_ids' in batch else None
        outputs = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        logits = outputs.logits
        pred_labels = torch.argmax(logits, dim = 1)
        ys = batch['label']
        correct += pred_labels.eq(ys).sum().item()
        total += input_ids.size(0)
    return correct / total


if __name__ == '__main__':
    device = torch.device("cuda")
    if args.victim == 'bert':
        clsf = BertVictimModel(model_name_or_path = 'bert-base-uncased', cache_dir = MODEL_CACHE_DIR + 'bert_model/bert-base-uncased', max_len = 100, device = device)
        output_dir = './checkpoints/bert-sst-robust/',
        epoch_save_name = './checkpoints/bert-sst-robust-epoch'
        print(epoch_save_name)
        substitutor = BertSubstitutor(model_type = 'bert-base-uncased', model_dir = MODEL_CACHE_DIR + 'bert_model/bert-base-uncased/masklm/')

    all_dataset = LocalSSTDataset(tokenizer = clsf.tokenizer)
    train_dataset = all_dataset.train_dataset
    valid_dataset = all_dataset.valid_dataset
    test_dataset = all_dataset.test_dataset

    train_dataset = remove_unused_columns(clsf.model, train_dataset)
    valid_dataset = remove_unused_columns(clsf.model, valid_dataset)
    test_dataset = remove_unused_columns(clsf.model, test_dataset)
    num_classes = get_class_num(args.dataset)
    sentence_pair = get_task_type(args.dataset)

    attacker = PGDAttack(victim_model = clsf, tokenizer = clsf.tokenizer, substitutor = substitutor, device = device, modification_rate = args.modif,
                        eta_z = args.eta_z, eta_u = args.eta_u, iter_time = args.iter_time, ste = args.ste,
                        norm = args.norm, rand_init = args.rand,  multi_sample = args.multi_sample,
                        discrete_sample_num = args.sample_num, final_sample_time = args.final_sample,
                        use_lm = args.use_lm, lm_loss_beta= args.lm_beta, use_cw_loss = args.cw,
                        victim = args.victim, num_classes = num_classes, sentence_pair = sentence_pair)

    optimizer = AdamW(params = clsf.model.parameters(), lr = 2e-5)

    train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True, collate_fn = all_dataset.data_collator)
    valid_loader = DataLoader(train_dataset, batch_size = 16, shuffle = False, collate_fn = all_dataset.data_collator)
    test_loader = DataLoader(train_dataset, batch_size = 16, shuffle = False, collate_fn = all_dataset.data_collator)
    

    global_acc = 0
    batches_per_epoch = len(train_loader)
    for epoch_idx in range(epoch_num):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        progress_bar = tqdm.auto.tqdm(range(len(train_loader)))
        for idx, batch in enumerate(train_loader):
            progress_bar.update(1)
            input_ids = batch['input_ids'].tolist()
            labels = batch['labels'].tolist()
            batch_size = len(input_ids)

            adv_sentences = []
            clsf.model.eval()
            for sen_idx in range(len(input_ids)):
                orig_sentence_ids = input_ids[sen_idx]
                orig_sentence = clsf.tokenizer.decode(orig_sentence_ids)
                orig_label = labels[sen_idx]
                with ctx_noparamgrad(clsf.model):
                    succ_examples, succ_pred_scores, succ_modif_rates,flag = attacker.attack(orig_sentence, orig_label, restart_num = 1)
                if flag:
                    if len(succ_examples) == 1:
                        adv_sentences.append(succ_examples[0])
                    elif len(succ_examples) > 1:
                        best_idx = np.argmin(succ_pred_scores[:, orig_label])
                        adv_sentences.append(succ_examples[best_idx])
                    else:
                        raise NotImplementedError
                else:
                    if len(succ_examples) == 0:
                        adv_sentences.append(orig_sentence)
                    else:
                        adv_sentences.append(succ_examples[0])
            if not use_trades:
                tokenized = clsf.tokenizer(adv_sentences, padding = 'longest', return_tensors = 'pt').to(device)
                ys = batch['labels'].to(device)
                optimizer.zero_grad()
            
                clsf.model.train()
                result = clsf.model(labels = ys, **tokenized)
                loss = result.loss
                logits = result.logits
                epoch_loss += loss.item()
                epoch_accuracy += torch.argmax(logits,dim = 1).eq(ys).sum().item()/batch_size
                loss.backward()
                optimizer.step()
            else:
                optimizer.zero_grad()
                batch = batch.to(device)
                result = clsf.model(**batch)
                natural_loss = result.loss

                natural_logits = result.logits

                tokenized = clsf.tokenizer(adv_sentences, padding = 'longest', return_tensors = 'pt').to(device)
                ys = batch['labels'].to(device)
                result = clsf.model(**tokenized)
                adv_logits = result.logits
                adv_loss = criterion_kl(F.log_softmax(adv_logits, dim = 1), F.softmax(natural_logits, dim = 1))
                loss = natural_loss + trades_beta * adv_loss
                epoch_loss += loss.item()
                epoch_accuracy += torch.argmax(natural_logits,dim = 1).eq(ys).sum().item()/batch_size
                loss.backward()
                optimizer.step()
        epoch_loss /= batches_per_epoch
        epoch_accuracy /= batches_per_epoch
        
        print(epoch_idx,' ',epoch_loss, ' ',epoch_accuracy)    
        print('Train accuracy = ', evaluate_accuracy(clsf.model, train_loader))
        local_acc = evaluate_accuracy(clsf.model, valid_loader)
        print("valid accuracy = ", local_acc)
        
        clsf.model.save_pretrained(epoch_save_name + str(epoch_idx))
        clsf.tokenizer.save_pretrained(epoch_save_name + str(epoch_idx))        


        if local_acc > global_acc:
            global_acc = local_acc
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            clsf.model.save_pretrained(output_dir)
            clsf.tokenizer.save_pretrained(output_dir)

    clsf.model = BertForSequenceClassification.from_pretrained(output_dir)
    clsf.model.to(device)
    print("Test accuracy = ", evaluate_accuracy(clsf.model, test_loader))
    print("All done")      
    clsf.model.eval() 

