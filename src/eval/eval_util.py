import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F


class GPT2_processor():
    def __init__(self, model_type = 'gpt2-xl', cache_dir = './model_cache/gpt2_model/gpt2_xl/'):
        self.gpt_model = GPT2LMHeadModel.from_pretrained(model_type, cache_dir = cache_dir)
        self.gpt_model.eval()
        self.gpt_model.to('cuda')
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_type,cache_dir = cache_dir)

    def predict_ppl(self,sentence, return_raw = False):
        '''
        sentence: word list
        '''
        if type(sentence) == list:
            str_sentence = ' '.join(sentence)
        else:
            str_sentence = sentence
        tokenize_input = self.tokenizer.tokenize(str_sentence)
        tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenize_input)])
        tensor_input = tensor_input.to('cuda')
        outputs = self.gpt_model(tensor_input, labels=tensor_input)
        loss, logits = outputs[:2]
        sentence_prob = loss.item()
        ppl = np.exp(sentence_prob)
        if return_raw:
            return sentence_prob,ppl
        return ppl
