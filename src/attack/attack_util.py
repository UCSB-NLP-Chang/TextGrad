import numpy as np 
import nltk
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM


def match_subword_with_word(subword_list, orig_word_list):
    '''
    subword_list: tokenized by transformers.models.bert.BertTokenizer, for example,  ["a", "great", "word", "##piece", "token","##izer"]
    orig_word_list: the original word list generated from the str.split() function, for example, ["a", "great", "wordpiece", "tokenizer"]

    to generate the match relationship before and after a sentence is tokenized by a word-piece tokenizer. Because some words should not be substituted, for example, "a", "the",
    (we filter words with two criterions: 1. stopwords  2. part-of-speech(pos), only noun, verb, adj, adv can be substituted). As a result, we need to generate a mask to represent
    whether a word in a sentence should be substituted. 
    But becuase of the fuck word piece, some words may be transformed into more than one sub-words. So, we need to know the mach relationship between them, and mask all subwords of 
    a filtered word.

    **both orig word list and subword list should have [CLS] at the begining of them and [SEP] at the end of them**

    use the same example, the result will be:  [0, 1, 3, 5]

    '''
    sentence_length = len(orig_word_list)
    index_list = np.zeros([sentence_length],dtype = int)


    new_tokens = []
    for i in range(len(subword_list)):
        token = subword_list[i]
        if len(token) >= 2 and token[:2] == '##':
            new_tokens.append(token[2:])
        else:
            new_tokens.append(token)

    tokenized_pos = 0
    sum_len1 = 0
    sum_len2 = 0
    for init_pos in range(len(orig_word_list)):
        curr_init_word = orig_word_list[init_pos]
        curr_tokenized_word = new_tokens[tokenized_pos]
        sum_len1 += len(curr_init_word)
        sum_len2 += len(curr_tokenized_word)
        if sum_len1 == sum_len2:
            index_list[init_pos] = tokenized_pos
            tokenized_pos += 1
        else:
            while 1:
                tokenized_pos += 1
                new_token = new_tokens[tokenized_pos]
                sum_len2 += len(new_token)
                if sum_len1 == sum_len2:
                    index_list[init_pos] = tokenized_pos
                    tokenized_pos += 1
                    break
    return index_list

def match_subword_with_word_albert(subword_list, orig_word_list):
    '''
    for albert tokenizer; there is some difference between bert and albert

    '''
    sentence_length = len(orig_word_list)
    index_list = np.zeros([sentence_length],dtype = int)

    orig_count = 0
    subword_count = 0
    subword_str = ""
    while True:
        if orig_count >= sentence_length:
            break
        curr_to_match = orig_word_list[orig_count]
        if subword_list[subword_count][0] == '▁' and len(subword_list[subword_count]) > 1:
            curr_subword = subword_list[subword_count][1:]
        elif subword_list[subword_count][0] == '▁':
            curr_subword = ''
        else:
            curr_subword = subword_list[subword_count]
        subword_str = subword_str + curr_subword


        # if subword_str != curr_to_match:
        if len(subword_str) != len(curr_to_match):
            subword_count += 1
            continue
        else:
            while True:
                if subword_count == len(subword_list) - 1:
                    break
                if subword_list[subword_count + 1] == '▁':
                    subword_count += 1
                    continue
                break
            index_list[orig_count] = subword_count
            orig_count += 1
            subword_count += 1
            subword_str = ""

    return index_list

def match_subword_with_word_roberta(subword_list, orig_word_list, tokenizer: RobertaTokenizer):
    '''
    for roberta tokenizer; compeletely different from BERT
    '''
    sentence_length = len(orig_word_list)
    index_list = np.zeros([sentence_length],dtype = int)

    orig_count = 0
    subword_count = 0
    subword_str = ""
    orig_str = ""
    update_orig_str = True
    while True:
        if orig_count >= sentence_length:
            break
        curr_to_match = orig_word_list[orig_count]
        if update_orig_str:
            orig_str = orig_str + curr_to_match
            update_orig_str = False

        if subword_list[subword_count][0] == 'Ġ' and len(subword_list[subword_count]) > 1:
            curr_subword = subword_list[subword_count][1:]
            curr_subword = tokenizer.convert_tokens_to_string([curr_subword])
        elif subword_list[subword_count][0] == 'Ġ':
            curr_subword = ''
        else:
            curr_subword = subword_list[subword_count]
            curr_subword = tokenizer.convert_tokens_to_string([curr_subword])
        subword_str = subword_str + curr_subword
        
        # print(curr_subword, subword_str, orig_str)
        if subword_str != orig_str:
            subword_count += 1
            continue
        else:
            while True:
                if subword_count == len(subword_list) - 1:
                    break
                if subword_list[subword_count + 1] == 'Ġ':
                    subword_count += 1
                    continue
                break
            index_list[orig_count] = subword_count
            orig_count += 1
            subword_count += 1
            # subword_str = ""
            update_orig_str = True

    return index_list


def pos_tag(word_list):
    '''
    label the part-of-speech of each word in the sentence. The word list should contain the original words rather than word piece.
    '''
    pos_tags = nltk.pos_tag(word_list)
    pos_tags = [x[1] for x in pos_tags]
    valid_pos_list = ['JJ','NN','RB','VB']

    transformed_pos_list = []
    for pos in pos_tags:
        if pos[:2] not in valid_pos_list:
            transformed_pos_list.append("o")
            continue
        if pos[:2] == 'JJ':
            pos = 'a'
        elif pos[:2] == 'NN':
            pos = 'n'
        elif pos[:2] == 'RB':
            pos = 'r'
        else:
            pos = 'v'
        transformed_pos_list.append(pos)    

    return transformed_pos_list

def expand_with_match_index(item_list, match_index):
    new_item_list = [None for _ in range(match_index[-1] + 1)]
    prev = 0
    for i in range(len(match_index)):
        new_item_list[prev: match_index[i] + 1] = [item_list[i] for _ in range(match_index[i] + 1 - prev)]
        prev = match_index[i] + 1
    return new_item_list

class STESample(torch.autograd.Function): 
    @staticmethod                          
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class STERandSelect(torch.autograd.Function):  
    @staticmethod                               
    def forward(ctx, input):
        num_classes = input.size(-1)
        res = torch.multinomial(input,1)
        out = res.view(-1)
        out = F.one_hot(out, num_classes = num_classes).double()
        return out


    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

def adjust_discrete_num(victim, length, orig_value):
    if victim == 'bert':
        return min([20, orig_value])
    elif victim == 'roberta':
        if length > 75:
            return min([15,orig_value])
        else:
            return min([20, orig_value])
    elif victim == 'albert':
        if length > 50:
            return min([5, orig_value])
        elif length > 30:
            return min([10, orig_value])
        else:
            return min([20, orig_value])


def strip_pad(idx_list, pad_id):
    new_idx_list = [x for x in idx_list if x != pad_id]
    return new_idx_list

