import numpy as np
import torch
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AdamW
from transformers.modeling_outputs import SequenceClassifierOutput
import os
import datasets
from .base_model import BaseModel
from tqdm import tqdm

class ALBERTVictimModel(BaseModel):
    def __init__(self,model_name_or_path = 'albert-xxlarge-v2',
                       cache_dir = './model_cache/albert_model/albert-xxlarge-v2/',
                       max_len = 100, 
                       device = torch.device('cuda'),
                       num_labels = 2):
        self.tokenizer = AlbertTokenizer.from_pretrained(model_name_or_path, cache_dir = cache_dir)
        self.model = AlbertForSequenceClassification.from_pretrained(model_name_or_path,cache_dir = cache_dir, num_labels = num_labels)
        self.device = device
        self.model = self.model.to(self.device)
        self.max_len = max_len
        self.max_batch_size = 32
    
    def get_input_embedding(self,):
        return self.model.albert.embeddings.word_embeddings

    def predict(self,sentence1_list, sentence2_list = None):
        '''
        sentence1_list: list of str. In NLI tasks, sentence1_list should contain a list of premises
        sentence2_list (optional, only for NLI tasks): list of str. sentence1_list should contain a list of hypotheses
        '''
        self.model.eval()
        tokenized = self.tokenizer(sentence1_list, sentence2_list, padding = 'longest', return_tensors = 'pt', truncation = True, max_length = self.max_len).to(self.device)
        xs = tokenized['input_ids']
        masks = tokenized['attention_mask']
        token_types = tokenized['token_type_ids']
        result = []
        if len(sentence1_list) <= self.max_batch_size:
            with torch.no_grad():
                res = self.model(input_ids = xs,attention_mask = masks, token_type_ids = token_types)
                logits = res.logits
                logits = torch.nn.functional.softmax(logits,dim = 1)
                result = logits.cpu().detach().numpy()
        else:
            batches = len(sentence1_list) // self.max_batch_size
            with torch.no_grad():
                for i in range(batches):
                    res = self.model(input_ids = xs[i*self.max_batch_size:(i+1) * self.max_batch_size],
                                     attention_mask = masks[i*self.max_batch_size:(i+1) * self.max_batch_size],
                                     token_type_ids = token_types[i*self.max_batch_size:(i+1) * self.max_batch_size])
                    logits = res.logits
                    logits = torch.nn.functional.softmax(logits,dim = 1)
                    result.append(logits.cpu().detach().numpy())
                if batches*self.max_batch_size < len(sentence1_list):
                    res = self.model(input_ids = xs[batches*self.max_batch_size:],
                                     attention_mask = masks[batches*self.max_batch_size:],
                                     token_type_ids = token_types[batches*self.max_batch_size:])
                    logits = res.logits
                    logits = torch.nn.functional.softmax(logits,dim = 1)
                    result.append(logits.cpu().detach().numpy())
                result = np.concatenate(result,axis = 0)
        assert len(result) == len(sentence1_list)
        return result

    def predict_via_embedding(self, embedding_mat, attention_mask, token_type_ids = None, labels = None):
        self.model.eval()
        batch_size = embedding_mat.size(0)
        if batch_size <= self.max_batch_size:
            res = self.model(inputs_embeds = embedding_mat, attention_mask = attention_mask, token_type_ids = token_type_ids, labels = labels)
            return res
        batches = batch_size // self.max_batch_size
        logits = []
        for i in range(batches):
            res = self.model(inputs_embeds = embedding_mat[i * self.max_batch_size: (i + 1) * self.max_batch_size],
                            attention_mask = attention_mask[i * self.max_batch_size: (i + 1) * self.max_batch_size],
                            token_type_ids = token_type_ids[i * self.max_batch_size: (i + 1) * self.max_batch_size],
                            )
            logits.append(res.logits)
        if batches * self.max_batch_size < batch_size:
            res = self.model(inputs_embeds = embedding_mat[batches * self.max_batch_size: ],
                            attention_mask = attention_mask[batches * self.max_batch_size: ],
                            token_type_ids = token_type_ids[i * self.max_batch_size: (i + 1) * self.max_batch_size],
                            )
            logits.append(res.logits)
        logits = torch.cat(logits, dim = 0)
        assert logits.size(0) == batch_size
        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
        )

