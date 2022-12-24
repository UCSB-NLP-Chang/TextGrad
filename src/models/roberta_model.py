import numpy as np
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from transformers.modeling_outputs import SequenceClassifierOutput
import os
import datasets
from .base_model import BaseModel
import tqdm

class RoBERTaVictimModel(BaseModel):
    def __init__(self,model_name_or_path = 'roberta-large',
                       cache_dir = './model_cache/roberta_model/roberta-large/',
                       max_len = 100,
                       device = torch.device("cuda"),
                       num_labels = 2,
                       ):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path, cache_dir = cache_dir)
        self.model = RobertaForSequenceClassification.from_pretrained(model_name_or_path, cache_dir = cache_dir, num_labels = num_labels)
        self.device = device
        self.model = self.model.to(self.device)
        # self.model.cuda()
        self.max_len = max_len
        self.optimizer = AdamW(self.model.parameters(), lr = 1e-5)
        self.max_batch_size = 128

    def get_input_embedding(self,):
        return self.model.roberta.embeddings.word_embeddings

    def predict(self,sentence1_list, sentence2_list = None):
        '''
        sentence1_list: list of str. In NLI tasks, sentence1_list should contain a list of premises
        sentence2_list (optional, only for NLI tasks): list of str. sentence1_list should contain a list of hypotheses
        '''
        self.model.eval()
        tokenized = self.tokenizer(sentence1_list, sentence2_list, padding = 'longest', return_tensors = 'pt', truncation = True, max_length = self.max_len).to(self.device)
        xs = tokenized['input_ids']
        masks = tokenized['attention_mask']
        result = []
        if len(sentence1_list) <= self.max_batch_size:
            with torch.no_grad():
                res = self.model(input_ids = xs,attention_mask = masks)
                logits = res.logits
                logits = torch.nn.functional.softmax(logits,dim = 1)
                result = logits.cpu().detach().numpy()
        else:
            batches = len(sentence1_list) // self.max_batch_size
            with torch.no_grad():
                for i in range(batches):
                    res = self.model(input_ids = xs[i*self.max_batch_size:(i+1) * self.max_batch_size],
                                     attention_mask = masks[i*self.max_batch_size:(i+1) * self.max_batch_size],
                                     )
                    logits = res.logits
                    logits = torch.nn.functional.softmax(logits,dim = 1)
                    result.append(logits.cpu().detach().numpy())
                if batches*self.max_batch_size < len(sentence1_list):
                    res = self.model(input_ids = xs[batches*self.max_batch_size:],
                                     attention_mask = masks[batches*self.max_batch_size:],
                                     )
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
            res = self.model(inputs_embeds = embedding_mat, attention_mask = attention_mask, labels = labels)
            return res
        batches = batch_size // self.max_batch_size
        logits = []
        for i in range(batches):
            res = self.model(inputs_embeds = embedding_mat[i * self.max_batch_size: (i + 1) * self.max_batch_size],
                            attention_mask = attention_mask[i * self.max_batch_size: (i + 1) * self.max_batch_size],
                            )
            logits.append(res.logits)
        if batches * self.max_batch_size < batch_size:
            res = self.model(inputs_embeds = embedding_mat[batches * self.max_batch_size: ],
                            attention_mask = attention_mask[batches * self.max_batch_size: ],
                            )
            logits.append(res.logits)
        logits = torch.cat(logits, dim = 0)
        assert logits.size(0) == batch_size
        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
        )
